# Parallel-SSSP Optimization Project

This project focuses on high-performance optimizations for the Single Source Shortest Path (SSSP) algorithm using OpenMP and SIMD (AVX2) intrinsics. The implementation builds upon the $\rho$-stepping and $\Delta$-stepping algorithms.

## Key Optimizations

### 1. Persistent Thread Pool (Zero Fork-Join Overhead)
The original implementation launched and tore down the OpenMP thread pool in every iteration of the SSSP main loop. For graphs with thousands of rounds (like the Germany road graph), this overhead dominated the execution time. 
- **Change**: Wrapped the entire SSSP main loop in a single persistent `#pragma omp parallel` region.
- **Result**: Threads stay alive throughout the algorithm, synchronizing via barriers instead of costly `pthread_join`/`pthread_create` pairs.

### 2. Parallel Two-Pass Prefix Scan
Sparse mode relaxation requires calculating offsets for merging thread-local buffers into the global frontier.
- **Change**: Replaced the serial $O(FrontierSize)$ prefix scan with a parallel two-pass scan.
- **Result**: Reduced the serial bottleneck in every round from millions of additions to just $O(NumThreads)$ (typically 8-32), significantly improving scalability on high-core-count machines.

### 3. Bit-Packed Frontier Management
Replaced the `atomic<bool>` frontier arrays (which consume 1 byte per vertex) with bit-packed `atomic<uint64_t>` words.
- **Optimization**: Each 64-byte cache line now covers 512 vertices (64 words * 8 bits) instead of just 64.
- **Benefit**: Virtually eliminates false sharing when threads update adjacent vertices in the frontier. It also reduces memory bandwidth requirements during the dense phase scans.

### 4. SIMD Hardware Gather (AVX2)
SSSP is traditionally memory-bound due to the indirect access pattern `dist[neighbor]`.
- **Change**: Implemented `_mm256_i32gather_epi32` to perform 8 distance lookups in parallel.
- **Benefit**: Hardware gather allows the CPU's load-store units to overlap multiple cache misses and take advantage of memory-level parallelism (MLP) better than scalar loads.

### 5. SIMD-Vectorized Pull Loops
In the dense phase and local BFS expansion for symmetrized graphs:
- **Change**: Added `#pragma omp simd reduction(min:best)` to the pull loops.
- **Result**: The compiler generates `_mm256_min_epi32` instructions to find the best incoming distance across 8 neighbors simultaneously.

## Build and Run

### Prerequisites
* g++ >= 7 with support for OpenMP and AVX2.

### Compilation
```bash
make omp
```

### Usage
```bash
./bin/sssp_omp -i <graph_file> -a delta-stepping -p <delta_value> -v
```

## Performance Analysis
Detailed performance comparisons between the baseline and optimized versions can be found in the accompanying [Analysis Report](analysis_report.md).
 
