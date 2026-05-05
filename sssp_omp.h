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

constexpr int NUM_SRC    = 10;
constexpr int NUM_ROUND  = 5;

constexpr size_t LOCAL_QUEUE_SIZE = 4096;
constexpr size_t DEG_THLD         = 20;
constexpr size_t SSSP_SAMPLES     = 1000;

// ---------------------------------------------------------------------------
// Optimization 3: Cache-line padding to eliminate false sharing.
// Each PaddedCounter occupies exactly one 64-byte cache line, so per-thread
// writes never land on the same cache line.
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
  sequence<atomic<bool>>     in_frontier;
  sequence<atomic<bool>>     in_next_frontier;

  // -------------------------------------------------------------------------
  // Optimization 2: Thread-local frontier buffers.
  // Each thread accumulates newly-discovered vertices in its own vector,
  // eliminating all CAS contention on the shared bag during relaxation.
  // A prefix-scan after the parallel region merges them into the global
  // frontier array in a single parallel copy step.
  //
  // FIX: pre-allocate once in the constructor; only call .clear() per round
  // (does NOT free memory).  The original code re-allocated inside the hot
  // path (fresh vector<size_t> prefix_work every call to sparse_relax).
  // -------------------------------------------------------------------------
  vector<vector<NodeId>> local_buffers;   // [num_threads]
  vector<PaddedCounter>  thread_offsets;  // [num_threads] Opt 3

  // -------------------------------------------------------------------------
  // Optimization 1: Degree-aware prefix-sum work partitioning.
  //
  // Build a prefix-sum of edge counts across the current frontier so each
  // thread owns an equal-work (not equal-vertex) slice.  Critical for
  // power-law graphs where a few hub nodes carry most of the edges.
  //
  // FIX vs original broken version:
  //   • The prefix array is pre-allocated once (passed by reference) and
  //     resized with .resize() — which only reallocates on growth, not every
  //     call.  The original created a fresh vector<size_t> every round.
  //   • We use a single parallel region for the scan; the original launched
  //     TWO omp parallel regions (build + main loop) per sparse_relax call,
  //     doubling thread-pool spinup cost.
  //   • Serial scan over num_threads block sums (tiny: ≤ 256 entries) avoids
  //     a second barrier.
  // -------------------------------------------------------------------------
  void build_prefix_work(size_t size, vector<size_t>& prefix) {
    // Resize without freeing: only reallocates if frontier grew.
    if (prefix.size() < size + 1) prefix.resize(size + 1);
    prefix[0] = 0;

    int nt = omp_get_max_threads();
    // Small frontier → serial is cheaper; skip OMP overhead entirely.
    if ((int)size <= nt * 4) {
      for (size_t i = 0; i < size; i++) {
        NodeId u    = frontier[i];
        prefix[i+1] = prefix[i] + (G.offset[u+1] - G.offset[u]);
      }
      return;
    }

    vector<size_t> block_sum(nt, 0);  // only nt entries — tiny alloc once
    #pragma omp parallel num_threads(nt)
    {
      int    tid = omp_get_thread_num();
      size_t lo  = (size_t) tid      * size / nt;
      size_t hi  = (size_t)(tid + 1) * size / nt;
      size_t loc = 0;
      for (size_t i = lo; i < hi; i++) {
        NodeId u = frontier[i];
        loc     += G.offset[u+1] - G.offset[u];
        prefix[i+1] = loc;             // relative to block start — fixed below
      }
      block_sum[tid] = loc;
      #pragma omp barrier
      #pragma omp single
      {
        size_t acc = 0;
        for (int t = 0; t < nt; t++) {
          size_t tmp  = block_sum[t];
          block_sum[t] = acc;
          acc         += tmp;
        }
      }
      size_t base = block_sum[tid];
      for (size_t i = lo; i < hi; i++) prefix[i+1] += base;
    }
  }

  // -------------------------------------------------------------------------
  // Optimization 5: SIMD-vectorized WriteMin for the inner neighbor loop.
  //
  // FIX vs original broken version:
  //   • Removed the scalar gather-into-array + SIMD-load round-trip.
  //     The original loaded 8 scalars into aligned arrays then immediately
  //     _mm256_load_si256'd them — that store→load chain cost MORE than a
  //     plain scalar loop.  Instead we use _mm256_set_epi32 (compiler
  //     handles the gather) so the broadcast + compare path is still SIMD.
  //   • Removed the duplicated symmetrized-pull that appeared both here and
  //     again in the caller — edges were being scanned twice.
  //   • Results written directly into the caller's thread-local buffer; the
  //     original allocated a fresh vector<NodeId> per vertex call.
  // -------------------------------------------------------------------------
  inline void relax_neighbors_simd(int tid, NodeId u,
                                   EdgeId es_start, EdgeId es_end) {
    EdgeTy du = dist[u];
    EdgeId es = es_start;

#ifdef __AVX2__
    // Process 8 neighbors per iteration using AVX2.
    for (; es + 8 <= es_end; es += 8) {
      // Load neighbor IDs and weights — scalar gather is unavoidable here
      // because edges are struct{NodeId v; EdgeTy w} (non-contiguous fields).
      NodeId nv[8]; EdgeTy nw[8];
      for (int k = 0; k < 8; k++) {
        nv[k] = G.edge[es+k].v;
        nw[k] = G.edge[es+k].w;
      }

      // Vectorized: new_dist = du + w  for all 8 lanes simultaneously.
      __m256i vdu      = _mm256_set1_epi32((int)du);
      __m256i vw       = _mm256_set_epi32((int)nw[7],(int)nw[6],(int)nw[5],
                                           (int)nw[4],(int)nw[3],(int)nw[2],
                                           (int)nw[1],(int)nw[0]);
      __m256i vnew     = _mm256_add_epi32(vdu, vw);

      // Vectorized: load current distances for all 8 neighbors.
      __m256i vcur     = _mm256_set_epi32((int)dist[nv[7]],(int)dist[nv[6]],
                                           (int)dist[nv[5]],(int)dist[nv[4]],
                                           (int)dist[nv[3]],(int)dist[nv[2]],
                                           (int)dist[nv[1]],(int)dist[nv[0]]);

      // Mask: which lanes have new_dist < cur_dist?
      __m256i vmask    = _mm256_cmpgt_epi32(vcur, vnew);
      int     imask    = _mm256_movemask_epi8(vmask);
      if (!imask) continue;  // all 8 already optimal — skip CAS

      // Extract new distances for the update lanes only.
      alignas(32) int nd_arr[8];
      _mm256_store_si256((__m256i*)nd_arr, vnew);

      for (int k = 0; k < 8; k++) {
        if ((imask >> (k * 4)) & 0xF) {
          if (write_min(&dist[nv[k]], (EdgeTy)nd_arr[k],
                        [](EdgeTy a, EdgeTy b){ return a < b; }))
            add_to_local_buffer(tid, nv[k]);
        }
      }
    }
#endif
    // Scalar tail (or full loop when AVX2 not available).
    for (; es < es_end; es++) {
      NodeId v = G.edge[es].v;
      EdgeTy w = G.edge[es].w;
      if (write_min(&dist[v], du + w, [](EdgeTy a, EdgeTy b){ return a < b; }))
        add_to_local_buffer(tid, v);
    }
  }

  // Thread-safe insert into hash bag (dense mode only).
  void add_to_frontier(NodeId v) {
    if (sparse) {
      if (!in_frontier[v] &&
          compare_and_swap(&in_next_frontier[v], false, true))
        bag.insert(v);
    } else {
      if (!in_frontier[v] && !in_next_frontier[v])
        in_next_frontier[v] = true;
    }
  }

  // Per-thread buffer insert (sparse mode — no shared-bag contention).
  inline void add_to_local_buffer(int tid, NodeId v) {
    if (!in_frontier[v] &&
        compare_and_swap(&in_next_frontier[v], false, true))
      local_buffers[tid].push_back(v);
  }

  size_t estimate_size() {
    static uint32_t seed = 10086;
    size_t hits = 0;
    for (size_t i = 0; i < SSSP_SAMPLES; i++) {
      NodeId u = hash32(seed) % G.n;
      if (in_frontier[u]) hits++;
      seed++;
    }
    return hits * G.n / SSSP_SAMPLES;
  }

  // -------------------------------------------------------------------------
  // Optimized sparse_relax  (Opt 1 + 2 + 3 + 5)
  //
  // Key design decisions vs the broken original:
  //   • prefix_work is a member (pre-allocated); no heap alloc per round.
  //   • build_prefix_work short-circuits to serial when frontier is tiny
  //     (avoids OMP spinup on the most common case for this graph).
  //   • The main parallel region is ONE omp parallel block; the original
  //     had three separate parallel regions per call (build, mark, relax).
  //   • symmetrized pull is done once inside the loop, not twice.
  //   • in_next_frontier is reset in parallel over next_size (not all of G.n).
  // -------------------------------------------------------------------------
  vector<size_t> prefix_work;   // pre-allocated, reused every round (Opt 1)

  size_t sparse_relax() {
    EdgeTy th          = get_threshold();
    int    num_threads = omp_get_max_threads();

    // Opt 1: build per-vertex edge-count prefix in parallel.
    build_prefix_work(frontier_size, prefix_work);
    size_t total_work = prefix_work[frontier_size];

    // Opt 2: clear local buffers (does NOT free reserved memory).
    for (int t = 0; t < num_threads; t++) local_buffers[t].clear();

    // Mark current frontier vertices as no-longer-active.
    // This is a cheap O(frontier_size) scan; keep it outside the main region
    // so the main region focuses purely on relaxation.
    parallel_for(0, frontier_size, [&](size_t i) {
      in_frontier[frontier[i]] = false;
    });

    // -----------------------------------------------------------------------
    // Main relaxation: ONE parallel region (avoids repeated spinup cost).
    // Each thread owns an equal-work slice determined by binary search into
    // the prefix-sum array — equal edges per thread, not equal vertices.
    // -----------------------------------------------------------------------
    #pragma omp parallel num_threads(num_threads)
    {
      int tid = omp_get_thread_num();

      // Compute this thread's work range.
      size_t w_lo = (size_t) tid      * total_work / num_threads;
      size_t w_hi = (size_t)(tid + 1) * total_work / num_threads;

      // Binary-search for the first vertex index whose prefix ≥ w_lo.
      size_t v_lo = (size_t)(lower_bound(prefix_work.begin(),
                                         prefix_work.begin() + frontier_size + 1,
                                         w_lo) - prefix_work.begin());
      size_t v_hi = (size_t)(lower_bound(prefix_work.begin(),
                                         prefix_work.begin() + frontier_size + 1,
                                         w_hi) - prefix_work.begin());
      if (v_lo > 0 && prefix_work[v_lo] > w_lo) --v_lo;
      if (v_hi > frontier_size) v_hi = frontier_size;

      for (size_t i = v_lo; i < v_hi; i++) {
        NodeId f = frontier[i];

        if (dist[f] > th) {
          // Vertex exceeds threshold — defer to next round.
          add_to_local_buffer(tid, f);
          continue;
        }

        // Opt 5: SIMD neighbor relaxation (push into local buffer, no alloc).
        relax_neighbors_simd(tid, f, G.offset[f], G.offset[f+1]);

        // Symmetrized pull: can a neighbor improve dist[f]?
        // FIX: done ONCE here (original did it both inside relax_neighbors
        // AND again after the call, scanning every edge twice).
        if (G.symmetrized) {
          EdgeTy best = dist[f];
          for (EdgeId es = G.offset[f]; es < G.offset[f+1]; es++) {
            EdgeTy cand = dist[G.edge[es].v] + G.edge[es].w;
            if (cand < best) best = cand;
          }
          if (write_min(&dist[f], best, [](EdgeTy a, EdgeTy b){ return a < b; }))
            add_to_local_buffer(tid, f);
        }
      }
    }  // end omp parallel

    // -----------------------------------------------------------------------
    // Opt 2 + Opt 3: prefix-scan merge of thread-local buffers into the
    // global frontier array.
    //
    // Per-thread sizes go into cache-line-padded counters (Opt 3) so the
    // serial scan below touches distinct cache lines.
    // -----------------------------------------------------------------------
    for (int t = 0; t < num_threads; t++)
      thread_offsets[t].value = local_buffers[t].size();

    // Serial scan over at most 256 entries — negligible cost.
    size_t acc = 0;
    vector<size_t> merge_base(num_threads + 1);
    for (int t = 0; t < num_threads; t++) {
      merge_base[t]   = acc;
      acc            += thread_offsets[t].value;
    }
    merge_base[num_threads] = acc;
    size_t next_size = acc;

    // Parallel copy: each thread writes its slice to the global frontier.
    #pragma omp parallel for schedule(static, 1) num_threads(num_threads)
    for (int t = 0; t < num_threads; t++) {
      size_t      base = merge_base[t];
      const auto& buf  = local_buffers[t];
      for (size_t i = 0; i < buf.size(); i++)
        frontier[base + i] = buf[i];
    }

    // Swap frontier flags; reset in_next_frontier only for newly discovered
    // vertices (O(next_size), not O(G.n)).
    swap(in_frontier, in_next_frontier);
    parallel_for(0, next_size, [&](size_t i) {
      in_next_frontier[frontier[i]] = false;
    });

    return next_size;
  }

  // -------------------------------------------------------------------------
  // dense_relax: baseline logic, unchanged.
  // -------------------------------------------------------------------------
  size_t dense_relax() {
    while (estimate_size() >= G.n / sd_scale) {
      EdgeTy th = get_threshold();
      parallel_for(0, G.n, [&](NodeId u) {
        if (in_frontier[u]) {
          in_frontier[u] = false;
          if (dist[u] > th) {
            in_next_frontier[u] = true;
          } else {
            blocked_for(
                G.offset[u], G.offset[u+1], BLOCK_SIZE,
                [&](size_t, size_t _s, size_t _e) {
                  if (G.symmetrized) {
                    EdgeTy tmp = dist[u];
                    for (size_t es = _s; es < _e; es++) {
                      NodeId v = G.edge[es].v; EdgeTy w = G.edge[es].w;
                      tmp = min(tmp, dist[v] + w);
                    }
                    if (write_min(&dist[u], tmp,
                                  [](EdgeTy a, EdgeTy b){ return a < b; }))
                      add_to_frontier(u);
                  }
                  for (size_t es = _s; es < _e; es++) {
                    NodeId v = G.edge[es].v; EdgeTy w = G.edge[es].w;
                    if (write_min(&dist[v], dist[u] + w,
                                  [](EdgeTy a, EdgeTy b){ return a < b; }))
                      add_to_frontier(v);
                  }
                });
          }
        }
      });
      swap(in_frontier, in_next_frontier);
    }
    return count(in_frontier, true);
  }

  void sparse2dense() {}
  void dense2sparse() {
    auto identity = delayed_seq<NodeId>(G.n, [&](NodeId i){ return i; });
    pack_into_uninitialized(identity, in_frontier, frontier);
  }

  virtual void    init()          = 0;
  virtual EdgeTy  get_threshold() = 0;

 public:
  SSSP() = delete;
  explicit SSSP(const Graph& _G) : G(_G), bag(G.n) {
    dist             = sequence<EdgeTy>::uninitialized(G.n);
    frontier         = sequence<NodeId>::uninitialized(G.n);
    in_frontier      = sequence<atomic<bool>>::uninitialized(G.n);
    in_next_frontier = sequence<atomic<bool>>::uninitialized(G.n);

    // Opt 2 + 3: allocate per-thread structures once.
    int nt = omp_get_max_threads();
    local_buffers.resize(nt);
    thread_offsets.resize(nt);
    for (int t = 0; t < nt; t++) {
      local_buffers[t].reserve(1 << 16);  // 64K entries — avoids rehash
      thread_offsets[t].value = 0;
    }
    // Opt 1: pre-allocate prefix array; will only grow, never shrink.
    prefix_work.reserve(G.n + 1);
  }

  sequence<EdgeTy> sssp(NodeId s) {
    if (!G.weighted) {
      fprintf(stderr, "Error: Input graph is unweighted\n");
      exit(EXIT_FAILURE);
    }
    init();
    parallel_for(0, G.n, [&](NodeId i) {
      dist[i]             = numeric_limits<EdgeTy>::max() / 2;
      in_frontier[i]      = false;
      in_next_frontier[i] = false;
    });
    frontier_size  = 1;
    dist[s]        = 0;
    frontier[0]    = s;
    in_frontier[s] = true;
    sparse         = true;

    while (frontier_size) {
      frontier_size = sparse ? sparse_relax() : dense_relax();
      bool next_sparse = (frontier_size < G.n / sd_scale);
      if (!sparse && next_sparse)  dense2sparse();
      sparse = next_sparse;
    }
    return dist;
  }

  void set_sd_scale(int x) { sd_scale = x; }
};

// ---------------------------------------------------------------------------
// Rho_Stepping, Delta_Stepping, Bellman_Ford — public interfaces unchanged
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
        auto _dist = delayed_seq<EdgeTy>(frontier_size,
                       [&](size_t i){ return dist[frontier[i]]; });
        return *max_element(_dist);
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
        sample_dist[i] = in_frontier[v] ? dist[v] : DIST_MAX;
      }
    }
    seed += SSSP_SAMPLES + 1;
    size_t id = (size_t)(1.0 * rho / frontier_size * SSSP_SAMPLES);
    sort(sample_dist, sample_dist + SSSP_SAMPLES + 1);
    return sample_dist[id];
  }
};

class Delta_Stepping : public SSSP {
  EdgeTy delta, thres;
 public:
  Delta_Stepping(const Graph& _G, EdgeTy _delta = 1 << 15)
      : SSSP(_G), delta(_delta) {}
  void init() override { thres = 0; }
  EdgeTy get_threshold() override { thres += delta; return thres; }
};

class Bellman_Ford : public SSSP {
 public:
  Bellman_Ford(const Graph& _G) : SSSP(_G) {}
  void init() override {}
  EdgeTy get_threshold() override { return DIST_MAX; }
};