#pragma once
#include <omp.h>

#include <algorithm>
#include <cassert>
#include <immintrin.h>  // AVX2/SIMD intrinsics

#include "graph.h"
#include "hashbag.h"
#include "parlay/internal/get_time.h"
using namespace std;
using namespace parlay;

constexpr int NUM_SRC   = 3;
constexpr int NUM_ROUND = 1;

constexpr size_t LOCAL_QUEUE_SIZE = 4096;
constexpr size_t DEG_THLD         = 20;
constexpr size_t SSSP_SAMPLES     = 1000;
constexpr size_t LOCAL_DISCOVERY_CACHE = 8192;

// ---------------------------------------------------------------------------
// Optimization 3: Cache-line padding to eliminate false sharing.
// ---------------------------------------------------------------------------
constexpr int CACHE_LINE = 64;
struct alignas(CACHE_LINE) PaddedCounter {
  size_t value{0};
  char   _pad[CACHE_LINE - sizeof(size_t)];
};

enum Algorithm { rho_stepping = 0, delta_stepping, bellman_ford };

class SSSP {
 protected:
  const Graph& G;
  bool         sparse;
  int          sd_scale;
  size_t       frontier_size;
  hashbag<NodeId> bag;
  sequence<EdgeTy>           dist;
  sequence<NodeId>           frontier;
  // Bit-packed frontier arrays (Optimization 3)
  sequence<atomic<uint64_t>> in_frontier;
  sequence<atomic<uint64_t>> in_next_frontier;
  vector<PaddedCounter>      block_sum;
  size_t                     total_work_shared;

  // Threshold computed once per round in omp single, shared with all threads.
  EdgeTy round_threshold;

  vector<vector<NodeId>> local_buffers;      // merged frontier buffers
  vector<vector<NodeId>> discovery_cache;    // thread-local pending discoveries

  vector<PaddedCounter>  thread_offsets;
  vector<size_t>         prefix_work;
  vector<size_t>         merge_base;

  // Shared flag set by omp single: do we need a dense→sparse materialisation
  // this round?
  bool need_dense2sparse{false};

  // -------------------------------------------------------------------------
  // Optimization 5: SIMD-vectorized WriteMin for the inner neighbor loop.
  // -------------------------------------------------------------------------
  inline void relax_neighbors_simd(int tid, NodeId u,
                                   EdgeId es_start, EdgeId es_end) {
    EdgeTy du = dist[u];
    EdgeId es = es_start;
#ifdef __AVX2__
    for (; es + 8 <= es_end; es += 8) {
      alignas(32) int nv[8], nw[8];
      for (int k = 0; k < 8; k++) {
        nv[k] = (int)G.edge[es+k].v;
        nw[k] = (int)G.edge[es+k].w;
      }
      __m256i vidx  = _mm256_load_si256((__m256i*)nv);
      __m256i vcur  = _mm256_i32gather_epi32((const int*)dist.begin(), vidx, 4);
      __m256i vdu   = _mm256_set1_epi32((int)du);
      __m256i vw    = _mm256_load_si256((__m256i*)nw);
      __m256i vnew  = _mm256_add_epi32(vdu, vw);
      __m256i vmask = _mm256_cmpgt_epi32(vcur, vnew);
      int     imask = _mm256_movemask_epi8(vmask);
      if (!imask) continue;
      alignas(32) int nd[8];
      _mm256_store_si256((__m256i*)nd, vnew);
      for (int k = 0; k < 8; k++)
        if ((imask >> (k*4)) & 0xF)
          if (write_min(&dist[nv[k]], (EdgeTy)nd[k],
                        [](EdgeTy a, EdgeTy b){ return a < b; }))
            add_to_local_buffer(tid, (NodeId)nv[k]);
    }
#endif
    for (; es < es_end; es++) {
      NodeId v = G.edge[es].v; EdgeTy w = G.edge[es].w;
      if (write_min(&dist[v], du + w, [](EdgeTy a, EdgeTy b){ return a < b; }))
        add_to_local_buffer(tid, v);
    }
  }

  // -- Bit-packed frontier helpers ------------------------------------------
  inline bool is_in_frontier(NodeId v) const {
    return (in_frontier[v >> 6].load(memory_order_relaxed) >> (v & 63)) & 1;
  }
  inline void clear_frontier_bit(NodeId v) {
    in_frontier[v >> 6].fetch_and(~(1ULL << (v & 63)), memory_order_relaxed);
  }
  // Set a bit in in_next_frontier; returns true if this thread was first.
  inline bool try_set_next(NodeId v) {
    uint64_t bit = 1ULL << (v & 63);
    uint64_t old = in_next_frontier[v >> 6].fetch_or(bit, memory_order_relaxed);
    return !(old & bit);
  }

  inline void flush_discovery_cache(int tid) {
    auto& cache = discovery_cache[tid];
    auto& out   = local_buffers[tid];

    for (size_t i = 0; i < cache.size(); i++) {
      NodeId v = cache[i];

      if (!is_in_frontier(v) && try_set_next(v))
        out.push_back(v);
    }

    cache.clear();
  }

  // Add v to this thread's local buffer if it is not already in the current
  // frontier and has not yet been added to the next frontier.
  inline void add_to_local_buffer(int tid, NodeId v) {
    auto& cache = discovery_cache[tid];

    cache.push_back(v);

    if (cache.size() >= LOCAL_DISCOVERY_CACHE)
      flush_discovery_cache(tid);
  }

  size_t estimate_size() {
    static uint32_t seed = 10086;
    size_t hits = 0;
    for (size_t i = 0; i < SSSP_SAMPLES; i++) {
      NodeId u = hash32(seed) % G.n;
      if (is_in_frontier(u)) hits++;
      seed++;
    }
    return hits * G.n / SSSP_SAMPLES;
  }

  // -------------------------------------------------------------------------
  // sparse_relax_body
  //
  // Barrier sequence (every thread hits every barrier unconditionally):
  //   B1  explicit — thread_offsets[t] all written
  //   B2  implicit (omp single end) — merge_base[] computed
  //   B3  explicit — frontier[] array fully written
  //
  // Returns: no trailing barrier; caller adds its own.
  // -------------------------------------------------------------------------
  void sparse_relax_body(int tid) {
    discovery_cache[tid].clear();
    int nt = omp_get_num_threads();

    EdgeTy th = round_threshold;
    local_buffers[tid].clear();

    // Dynamic scheduling avoids the prefix-work setup cost and handles the
    // degree skew in Kron-style graphs without extra per-round barriers.
    NodeId lq[LOCAL_QUEUE_SIZE];
    #pragma omp for schedule(dynamic, 64) nowait
    for (size_t i = 0; i < frontier_size; i++) {
      NodeId f = frontier[i];
      // Clear f's bit in the CURRENT frontier before we process it.
      // This must happen before we relax neighbours so that
      // add_to_local_buffer won't skip f if a neighbour pushes it back.
      clear_frontier_bit(f);

      if (dist[f] > th) {
        add_to_local_buffer(tid, f);
        continue;
      }
      size_t f_deg = G.offset[f+1] - G.offset[f];
      if (f_deg < LOCAL_QUEUE_SIZE) {
        // Small-degree fast path: local BFS expansion
        size_t lq_f = 0, lq_r = 0;
        lq[lq_r++] = f;
        while (lq_f < lq_r && lq_r < LOCAL_QUEUE_SIZE) {
          NodeId u = lq[lq_f++];
          size_t u_deg = G.offset[u+1] - G.offset[u];
          if (u_deg >= LOCAL_QUEUE_SIZE || dist[u] > th) {
            add_to_local_buffer(tid, u);
            continue;
          }
          // Pull step for symmetrized graphs (Optimization 5)
          if (G.symmetrized) {
            EdgeTy best = dist[u];
            #pragma omp simd reduction(min:best)
            for (EdgeId es = G.offset[u]; es < G.offset[u+1]; es++)
              best = min(best, dist[G.edge[es].v] + G.edge[es].w);
            write_min(&dist[u], best,
                      [](EdgeTy a, EdgeTy b){ return a < b; });
          }
          // Push step
          for (EdgeId es = G.offset[u]; es < G.offset[u+1]; es++) {
            NodeId v = G.edge[es].v; EdgeTy w = G.edge[es].w;
            if (write_min(&dist[v], dist[u] + w,
                          [](EdgeTy a, EdgeTy b){ return a < b; })) {
              if (lq_r < LOCAL_QUEUE_SIZE)
                lq[lq_r++] = v;
              else
                add_to_local_buffer(tid, v);
            }
          }
        }
        for (size_t j = lq_f; j < lq_r; j++)
          add_to_local_buffer(tid, lq[j]);
      } else {
        // High-degree vertex
        if (G.symmetrized) {
          EdgeTy df = dist[f], best = df;
          for (EdgeId es = G.offset[f]; es < G.offset[f+1]; es++) {
            NodeId v = G.edge[es].v;
            best = min(best, dist[v] + G.edge[es].w);
            if (write_min(&dist[v], df + G.edge[es].w,
                          [](EdgeTy a, EdgeTy b){ return a < b; }))
              add_to_local_buffer(tid, v);
          }
          if (write_min(&dist[f], best,
                        [](EdgeTy a, EdgeTy b){ return a < b; }))
            add_to_local_buffer(tid, f);
        } else {
          relax_neighbors_simd(tid, f, G.offset[f], G.offset[f+1]);
        }
      }
    }
    flush_discovery_cache(tid);
    // ── Step 5: Merge thread-local buffers → frontier[] ──────────────────
    thread_offsets[tid].value = local_buffers[tid].size();
    #pragma omp barrier  // B1

    #pragma omp single
    {                    // implicit barrier at end (B2)
      size_t acc = 0;
      for (int t = 0; t < nt; t++) {
        merge_base[t]          = acc;
        acc                   += thread_offsets[t].value;
      }
      merge_base[nt]    = acc;
      total_work_shared = acc;
    }

    size_t base = merge_base[tid];
    for (size_t i = 0; i < local_buffers[tid].size(); i++)
      frontier[base + i] = local_buffers[tid][i];
    #pragma omp barrier  // B3 — frontier[] fully written; no trailing barrier
  }

  // -------------------------------------------------------------------------
  // dense_relax_body — one pass over all vertices for this thread's slice.
  // No internal or trailing barriers; caller is responsible for both.
  // -------------------------------------------------------------------------
  void dense_relax_body(int tid) {
    EdgeTy th = round_threshold;

    #pragma omp for schedule(dynamic, 1024) nowait
    for (size_t uu = 0; uu < G.n; uu++) {
      NodeId u = (NodeId)uu;
      if (!is_in_frontier(u)) continue;
      clear_frontier_bit(u);
      if (dist[u] > th) {
        try_set_next(u);
      } else {
        // Pull step for symmetrized graphs (Optimization 5)
        if (G.symmetrized) {
          EdgeTy best = dist[u];
          #pragma omp simd reduction(min:best)
          for (EdgeId es = G.offset[u]; es < G.offset[u+1]; es++)
            best = min(best, dist[G.edge[es].v] + G.edge[es].w);
          if (write_min(&dist[u], best,
                        [](EdgeTy a, EdgeTy b){ return a < b; }))
            try_set_next(u);
        }
        // Push step
        for (EdgeId es = G.offset[u]; es < G.offset[u+1]; es++) {
          NodeId v = G.edge[es].v; EdgeTy w = G.edge[es].w;
          if (write_min(&dist[v], dist[u] + w,
                        [](EdgeTy a, EdgeTy b){ return a < b; }))
            try_set_next(v);
        }
      }
    }
    flush_discovery_cache(tid);
    // No barrier — caller is responsible.
  }

  // -------------------------------------------------------------------------
  // dense2sparse_body — materialise frontier[] from in_frontier bits.
  // Ends with an explicit barrier after writing frontier[]; caller adds its
  // own single/barrier if it needs frontier_size updated.
  // -------------------------------------------------------------------------
  void dense2sparse_body(int tid) {
    discovery_cache[tid].clear();
    int    nt = omp_get_num_threads();
    local_buffers[tid].clear();
    size_t lo = (size_t)tid       * G.n / nt;
    size_t hi = (size_t)(tid + 1) * G.n / nt;
    for (NodeId i = (NodeId)lo; i < (NodeId)hi; i++)
      if (is_in_frontier(i)) local_buffers[tid].push_back(i);

    thread_offsets[tid].value = local_buffers[tid].size();
    #pragma omp barrier  // all thread_offsets written

    #pragma omp single
    {                    // implicit barrier at end
      size_t acc = 0;
      for (int t = 0; t < nt; t++) {
        merge_base[t] = acc;
        acc          += thread_offsets[t].value;
      }
      merge_base[nt]    = acc;
      total_work_shared = acc;
    }

    size_t base = merge_base[tid];
    for (size_t i = 0; i < local_buffers[tid].size(); i++)
      frontier[base + i] = local_buffers[tid][i];
    #pragma omp barrier  // frontier[] fully written
  }

  // -------------------------------------------------------------------------
  // zero_next_frontier_body — clear every word of in_next_frontier.
  // Called after each swap so residual bits from the old frontier don't leak.
  // No trailing barrier; caller adds one if needed.
  // -------------------------------------------------------------------------
  void zero_next_frontier_body(int tid) {
    int    nt     = omp_get_num_threads();
    size_t nwords = (G.n + 63) / 64;
    size_t lo     = (size_t)tid       * nwords / nt;
    size_t hi     = (size_t)(tid + 1) * nwords / nt;
    for (size_t i = lo; i < hi; i++)
      in_next_frontier[i].store(0, memory_order_relaxed);
  }

  virtual void   init()          = 0;
  virtual EdgeTy get_threshold() = 0;

 public:
  SSSP() = delete;
  explicit SSSP(const Graph& _G) : G(_G), bag(G.n) {
    dist             = sequence<EdgeTy>::uninitialized(G.n);
    frontier         = sequence<NodeId>::uninitialized(G.n);
    size_t n_words   = (G.n + 63) / 64;
    in_frontier      = sequence<atomic<uint64_t>>::uninitialized(n_words);
    in_next_frontier = sequence<atomic<uint64_t>>::uninitialized(n_words);
    int nt = omp_get_max_threads();
    local_buffers.resize(nt);
    discovery_cache.resize(nt);

    thread_offsets.resize(nt);
    block_sum.resize(nt);

    for (int t = 0; t < nt; t++) {
      local_buffers[t].reserve(1 << 16);

      discovery_cache[t].reserve(LOCAL_DISCOVERY_CACHE);

      thread_offsets[t].value = 0;
      block_sum[t].value      = 0;
    }
    prefix_work.reserve(G.n + 1);
    merge_base.resize(nt + 1, 0);
    round_threshold   = 0;
    total_work_shared = 0;
  }

  // =========================================================================
  // sssp() — main driver.
  //
  // Per-round barrier discipline (ALL threads hit ALL barriers):
  //
  //  omp single: get_threshold                    [implicit barrier]
  //  if sparse:
  //    sparse_relax_body(tid)          — ends at B6 (explicit)
  //  else:
  //    while (estimate_size >= G.n/sd_scale):
  //      dense_relax_body(tid)
  //      omp barrier
  //      omp single: swap + zero_next  [implicit barrier]
  //      zero_next_frontier_body(tid)
  //      omp barrier
  //  omp barrier                       — all relax work done
  //  omp single: swap, update frontier_size & sparse, set need_dense2sparse
  //              [implicit barrier]
  //  zero_next_frontier_body(tid)      — zero old in_frontier (now in_next)
  //  omp barrier
  //  if need_dense2sparse:
  //    dense2sparse_body(tid)          — ends at explicit barrier
  //    omp single: frontier_size = total_work_shared  [implicit barrier]
  //  loop-back
  // =========================================================================
  sequence<EdgeTy> sssp(NodeId s) {
    if (!G.weighted) exit(EXIT_FAILURE);
    init();

    size_t n_words = (G.n + 63) / 64;
    parallel_for(0, n_words, [&](size_t i) {
      in_frontier[i].store(0, memory_order_relaxed);
      in_next_frontier[i].store(0, memory_order_relaxed);
    });
    parallel_for(0, G.n, [&](NodeId i) {
      dist[i] = numeric_limits<EdgeTy>::max() / 2;
    });

    frontier_size = 1;
    dist[s]       = 0;
    frontier[0]   = s;
    in_frontier[s >> 6].fetch_or(1ULL << (s & 63), memory_order_relaxed);
    sparse            = true;
    need_dense2sparse = false;
    int nt = omp_get_max_threads();

    #pragma omp parallel num_threads(nt)
    {
      int tid = omp_get_thread_num();

      while (frontier_size) {

        // ── Compute threshold for this round ───────────────────────────
        #pragma omp single
        {
          round_threshold = get_threshold();
        }
        // implicit barrier after omp single

        // ── Relax ───────────────────────────────────────────────────────
        if (sparse) {
          // sparse_relax_body ends at B6 (an explicit barrier).
          sparse_relax_body(tid);
          // No trailing barrier here — fall through to the global barrier.

        } else {
          // Dense mode: keep looping until the frontier shrinks enough.
          // We mirror the baseline's while (estimate_size() >= G.n/sd_scale)
          // structure.  The estimate and the loop-control decision are made
          // inside omp single so all threads agree.
          while (true) {
            dense_relax_body(tid);
            #pragma omp barrier  // all vertices processed for this pass

            // Swap arrays and decide whether to do another dense pass.
            #pragma omp single
            {
              swap(in_frontier, in_next_frontier);
              size_t est = estimate_size();
              // If we will loop again we must zero in_next_frontier
              // (= old in_frontier) so that try_set_next works cleanly.
              // We set total_work_shared to signal this to all threads.
              total_work_shared = (est >= (size_t)(G.n / sd_scale)) ? 1 : 0;
            }
            // implicit barrier after omp single

            if (total_work_shared) {
              // Another dense pass coming: zero in_next_frontier in parallel.
              zero_next_frontier_body(tid);
              #pragma omp barrier
            } else {
              break;
            }
          }
          // After the dense loop, in_frontier has the final frontier bits.
        }

        // ── All relax work is done; synchronise ─────────────────────────
        #pragma omp barrier

        // ── Swap arrays, update frontier_size & sparse, flag d2s need ───
        #pragma omp single
        {
          if (sparse) {
            // sparse_relax_body already placed the new frontier into
            // frontier[] and set total_work_shared.
            // We still need to swap so in_frontier reflects the new set.
            swap(in_frontier, in_next_frontier);
            frontier_size = total_work_shared;
          } else {
            // Dense relax already swapped inside the inner loop.
            // Materialise frontier[] and get the exact count.
            frontier_size = 0;  // will be set by dense2sparse below
          }
          bool prev_dense   = !sparse;
          sparse            = (frontier_size < (size_t)(G.n / sd_scale));
          // Need to materialise frontier[] if we were dense (frontier[] was
          // not maintained in dense mode) and either:
          //   (a) we are transitioning to sparse, or
          //   (b) we are staying sparse (frontier_size was 0 above, real
          //       count comes from dense2sparse).
          // In practice we always call dense2sparse after a dense round so
          // we have frontier[] and the exact frontier_size.
          need_dense2sparse = prev_dense && (frontier_size == 0 || sparse);
        }
        // implicit barrier after omp single

        // ── Zero in_next_frontier (residual bits from old in_frontier) ──
        // After the swap in_next_frontier holds the old frontier bits.
        // Clear them so try_set_next works correctly next round.
        zero_next_frontier_body(tid);
        #pragma omp barrier  // in_next_frontier clean

        // ── Dense→Sparse: materialise frontier[] ────────────────────────
        if (need_dense2sparse) {
          dense2sparse_body(tid);
          // dense2sparse_body ends with an explicit barrier.
          #pragma omp single
          { frontier_size = total_work_shared; }
          // implicit barrier after omp single
        }

      } // while (frontier_size)
    } // omp parallel

    return dist;
  }

  void set_sd_scale(int x) { sd_scale = x; }
};

// ---------------------------------------------------------------------------
// Rho_Stepping
// ---------------------------------------------------------------------------
class Rho_Stepping : public SSSP {
  size_t   rho;
  uint32_t seed;
 public:
  Rho_Stepping(const Graph& _G, size_t _rho = 1 << 20)
      : SSSP(_G), rho(_rho), seed(0) {}
  void init() override {}
  EdgeTy get_threshold() override {
    if (frontier_size <= rho) {
      if (sparse) {
        EdgeTy best = 0;
        for (size_t i = 0; i < frontier_size; i++)
          best = max(best, dist[frontier[i]]);
        return best;
      } else {
        return DIST_MAX;
      }
    }
    EdgeTy sample_dist[SSSP_SAMPLES + 1];
    for (size_t i = 0; i <= SSSP_SAMPLES; i++) {
      if (sparse) {
        NodeId v       = frontier[hash32(seed + i) % frontier_size];
        sample_dist[i] = dist[v];
      } else {
        NodeId v       = hash32(seed + i) % G.n;
        sample_dist[i] = is_in_frontier(v) ? dist[v] : DIST_MAX;
      }
    }
    seed += SSSP_SAMPLES + 1;
    size_t id = (size_t)(1.0 * rho / frontier_size * SSSP_SAMPLES);
    sort(sample_dist, sample_dist + SSSP_SAMPLES + 1);
    return sample_dist[id];
  }
};

// ---------------------------------------------------------------------------
// Delta_Stepping
// ---------------------------------------------------------------------------
class Delta_Stepping : public SSSP {
  EdgeTy delta, thres;
 public:
  Delta_Stepping(const Graph& _G, EdgeTy _delta = 1 << 15)
      : SSSP(_G), delta(_delta) {}
  void init() override { thres = 0; }
  EdgeTy get_threshold() override { thres += delta; return thres; }
};

// ---------------------------------------------------------------------------
// Bellman_Ford
// ---------------------------------------------------------------------------
class Bellman_Ford : public SSSP {
 public:
  Bellman_Ford(const Graph& _G) : SSSP(_G) {}
  void init() override {}
  EdgeTy get_threshold() override { return DIST_MAX; }
};
