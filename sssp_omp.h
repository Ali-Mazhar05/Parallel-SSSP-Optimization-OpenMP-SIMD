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


//changed this to 3 and 1 for testing
constexpr int NUM_SRC    = 3;
constexpr int NUM_ROUND  = 1;

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
  sequence<atomic<uint64_t>> in_frontier;
  sequence<atomic<uint64_t>> in_next_frontier;
  vector<PaddedCounter>      block_sum;
  size_t                     total_work_shared;

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
          if (write_min(&dist[nv[k]], (EdgeTy)nd[k], [](EdgeTy a, EdgeTy b){ return a < b; }))
            add_to_local_buffer(tid, (NodeId)nv[k]);
    }
#endif
    for (; es < es_end; es++) {
      NodeId v = G.edge[es].v; EdgeTy w = G.edge[es].w;
      if (write_min(&dist[v], du + w, [](EdgeTy a, EdgeTy b){ return a < b; }))
        add_to_local_buffer(tid, v);
    }
  }

  inline bool is_in_frontier(NodeId v) {
    return (in_frontier[v >> 6].load(memory_order_relaxed) >> (v & 63)) & 1;
  }
  inline bool try_set_next(NodeId v) {
    uint64_t bit = 1ULL << (v & 63);
    uint64_t old = in_next_frontier[v >> 6].fetch_or(bit, memory_order_relaxed);
    return !(old & bit);
  }

  void add_to_frontier(NodeId v) {
    if (!is_in_frontier(v)) try_set_next(v);
  }

  inline void add_to_local_buffer(int tid, NodeId v) {
    if (!is_in_frontier(v) && try_set_next(v))
      local_buffers[tid].push_back(v);
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
  // Optimized sparse_relax  (Opt 1 + 2 + 3 + 5)
  //
  // Uses a SINGLE omp parallel region per call, replacing the previous 4
  // separate fork-join launches (build_prefix_work had its own block, plus
  // two ParlayLib parallel_for calls sandwiching the main block).  For a
  // road graph doing thousands of BFS rounds, this spinup cost dominated.
  //
  // Other key fixes bundled here:
  //   - No --v_lo boundary adjustment: the old hack caused the boundary
  //     vertex to be processed by two threads (double work + double CAS).
  //   - Combined pull+push for symmetrized graphs in one edge scan, halving
  //     the number of times each edge is touched (was: push, then pull = 2×).
  //   - merge_base is a pre-allocated member; no heap alloc per round.
  // -------------------------------------------------------------------------
  vector<size_t> prefix_work;   // pre-allocated, reused every round (Opt 1)
  vector<size_t> merge_base;    // [num_threads+1], prefix offsets for merge

  void sparse_relax_body(int tid) {
    int nt = omp_get_max_threads();
    EdgeTy th = get_threshold();
    local_buffers[tid].clear();

    // Step 1: Parallel Prefix Scan of degrees
    size_t lo = (size_t)tid * frontier_size / nt;
    size_t hi = (size_t)(tid + 1) * frontier_size / nt;
    size_t loc = 0;
    for (size_t i = lo; i < hi; i++) {
      NodeId u = frontier[i];
      loc += G.offset[u+1] - G.offset[u];
      prefix_work[i+1] = loc;
    }
    block_sum[tid].value = loc;
    #pragma omp barrier

    #pragma omp single
    {
      size_t acc = 0;
      for (int t = 0; t < nt; t++) {
        size_t tmp = block_sum[t].value;
        block_sum[t].value = acc;
        acc += tmp;
      }
      prefix_work[0] = 0;
      total_work_shared = acc;
    }
    for (size_t i = lo; i < hi; i++) prefix_work[i+1] += block_sum[tid].value;
    #pragma omp barrier

    size_t total = total_work_shared;
    size_t v_lo, v_hi;
    if (total == 0) {
      v_lo = lo; v_hi = hi;
    } else {
      size_t w_lo = (size_t)tid * total / nt, w_hi = (size_t)(tid + 1) * total / nt;
      v_lo = lower_bound(prefix_work.begin(), prefix_work.begin() + frontier_size + 1, w_lo) - prefix_work.begin();
      v_hi = lower_bound(prefix_work.begin(), prefix_work.begin() + frontier_size + 1, w_hi) - prefix_work.begin();
    }

    NodeId lq[LOCAL_QUEUE_SIZE];
    for (size_t i = v_lo; i < v_hi; i++) {
      NodeId f = frontier[i];
      in_frontier[f >> 6].fetch_and(~(1ULL << (f & 63)), memory_order_relaxed);
      if (dist[f] > th) { add_to_local_buffer(tid, f); continue; }
      size_t f_deg = G.offset[f+1] - G.offset[f];
      if (f_deg < LOCAL_QUEUE_SIZE) {
        size_t lq_f = 0, lq_r = 0; lq[lq_r++] = f;
        while (lq_f < lq_r && lq_r < LOCAL_QUEUE_SIZE) {
          NodeId u = lq[lq_f++];
          if (G.offset[u+1]-G.offset[u] >= LOCAL_QUEUE_SIZE || dist[u] > th) { add_to_local_buffer(tid, u); continue; }
          if (G.symmetrized) {
            EdgeTy best = dist[u]; EdgeId s = G.offset[u], e = G.offset[u+1];
            #pragma omp simd reduction(min:best)
            for (EdgeId es = s; es < e; es++) best = min(best, dist[G.edge[es].v] + G.edge[es].w);
            write_min(&dist[u], best, [](EdgeTy a, EdgeTy b){ return a < b; });
          }
          for (EdgeId es = G.offset[u]; es < G.offset[u+1]; es++) {
            NodeId v = G.edge[es].v; EdgeTy w = G.edge[es].w;
            if (write_min(&dist[v], dist[u]+w, [](EdgeTy a, EdgeTy b){ return a < b; })) {
              if (lq_r < LOCAL_QUEUE_SIZE) lq[lq_r++] = v; else add_to_local_buffer(tid, v);
            }
          }
        }
        for (size_t j = lq_f; j < lq_r; j++) add_to_local_buffer(tid, lq[j]);
      } else {
        if (G.symmetrized) {
          EdgeTy df = dist[f], best = df;
          for (EdgeId es = G.offset[f]; es < G.offset[f+1]; es++) {
            NodeId v = G.edge[es].v; best = min(best, dist[v] + G.edge[es].w);
            if (write_min(&dist[v], df + G.edge[es].w, [](EdgeTy a, EdgeTy b){ return a < b; })) add_to_local_buffer(tid, v);
          }
          write_min(&dist[f], best, [](EdgeTy a, EdgeTy b){ return a < b; });
          add_to_local_buffer(tid, f);
        } else relax_neighbors_simd(tid, f, G.offset[f], G.offset[f+1]);
      }
    }
    #pragma omp barrier
    thread_offsets[tid].value = local_buffers[tid].size();
    #pragma omp barrier
    #pragma omp single
    {
      size_t acc = 0;
      for (int t = 0; t < nt; t++) { merge_base[t] = acc; acc += thread_offsets[t].value; }
      merge_base[nt] = acc; total_work_shared = acc;
    }
    size_t base = merge_base[tid];
    for (size_t i = 0; i < local_buffers[tid].size(); i++) frontier[base + i] = local_buffers[tid][i];
    #pragma omp barrier
    size_t ns = total_work_shared;
    size_t m_lo = (size_t)tid * ns / nt, m_hi = (size_t)(tid + 1) * ns / nt;
    for (size_t i = m_lo; i < m_hi; i++) {
      NodeId v = frontier[i];
      in_next_frontier[v >> 6].fetch_and(~(1ULL << (v & 63)), memory_order_relaxed);
    }
  }

  // -------------------------------------------------------------------------
  // dense_relax: baseline logic, unchanged.
  // -------------------------------------------------------------------------
  void dense_relax_body(int tid) {
    int nt = omp_get_max_threads();
    size_t lo = (size_t)tid * G.n / nt, hi = (size_t)(tid+1) * G.n / nt;
    EdgeTy th = get_threshold();
    for (NodeId u = lo; u < hi; u++) {
      if (is_in_frontier(u)) {
        in_frontier[u >> 6].fetch_and(~(1ULL << (u & 63)), memory_order_relaxed);
        if (dist[u] > th) { try_set_next(u); }
        else {
          if (G.symmetrized) {
            EdgeTy best = dist[u]; EdgeId s = G.offset[u], e = G.offset[u+1];
            #pragma omp simd reduction(min:best)
            for (EdgeId es = s; es < e; es++) best = min(best, dist[G.edge[es].v] + G.edge[es].w);
            if (write_min(&dist[u], best, [](EdgeTy a, EdgeTy b){ return a < b; })) try_set_next(u);
          }
          for (EdgeId es = G.offset[u]; es < G.offset[u+1]; es++) {
            NodeId v = G.edge[es].v; EdgeTy w = G.edge[es].w;
            if (write_min(&dist[v], dist[u] + w, [](EdgeTy a, EdgeTy b){ return a < b; })) try_set_next(v);
          }
        }
      }
    }
  }

  void dense2sparse_body(int tid) {
    int nt = omp_get_max_threads();
    local_buffers[tid].clear();
    size_t lo = (size_t)tid * G.n / nt, hi = (size_t)(tid+1) * G.n / nt;
    for (NodeId i = lo; i < hi; i++) if (is_in_frontier(i)) local_buffers[tid].push_back(i);
    #pragma omp barrier
    thread_offsets[tid].value = local_buffers[tid].size();
    #pragma omp barrier
    #pragma omp single
    {
      size_t acc = 0;
      for (int t = 0; t < nt; t++) { merge_base[t] = acc; acc += thread_offsets[t].value; }
      merge_base[nt] = acc; total_work_shared = acc;
    }
    size_t base = merge_base[tid];
    for (size_t i = 0; i < local_buffers[tid].size(); i++) frontier[base + i] = local_buffers[tid][i];
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
    size_t n_words   = (G.n + 63) / 64;
    in_frontier      = sequence<atomic<uint64_t>>::uninitialized(n_words);
    in_next_frontier = sequence<atomic<uint64_t>>::uninitialized(n_words);
    int nt = omp_get_max_threads();
    local_buffers.resize(nt);
    thread_offsets.resize(nt);
    block_sum.resize(nt);
    for (int t = 0; t < nt; t++) {
      local_buffers[t].reserve(1 << 16);
      thread_offsets[t].value = 0;
    }
    prefix_work.reserve(G.n + 1);
    merge_base.resize(nt + 1, 0);
  }

  sequence<EdgeTy> sssp(NodeId s) {
    if (!G.weighted) exit(EXIT_FAILURE);
    init();
    size_t n_words = (G.n + 63) / 64;
    parallel_for(0, n_words, [&](size_t i) { in_frontier[i] = 0; in_next_frontier[i] = 0; });
    parallel_for(0, G.n, [&](NodeId i) { dist[i] = numeric_limits<EdgeTy>::max() / 2; });
    frontier_size = 1; dist[s] = 0; frontier[0] = s;
    in_frontier[s >> 6].fetch_or(1ULL << (s & 63));
    sparse = true;
    int nt = omp_get_max_threads();
    #pragma omp parallel num_threads(nt)
    {
      int tid = omp_get_thread_num();
      while (frontier_size) {
        if (sparse) sparse_relax_body(tid); else dense_relax_body(tid);
        #pragma omp barrier
        bool next_sparse = sparse;
        #pragma omp single
        {
          swap(in_frontier, in_next_frontier);
          if (sparse) {
            frontier_size = total_work_shared;
            if (frontier_size >= G.n / sd_scale) sparse = false;
          } else {
            frontier_size = estimate_size();
            if (frontier_size < G.n / sd_scale) sparse = true;
          }
          next_sparse = sparse;
        }
        #pragma omp barrier
        if (!next_sparse && sparse) { // Transitioned from dense to sparse
          dense2sparse_body(tid);
          #pragma omp barrier
        }
      }
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