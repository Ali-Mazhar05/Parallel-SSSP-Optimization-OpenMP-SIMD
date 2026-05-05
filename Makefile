# ==============================================================================
# Makefile — Parallel SSSP: Baseline (parlay scheduler) vs OpenMP-optimized
#
# WHY NO -lcilkrts:
#   g++ >= 11 dropped CilkPlus support. The parlay library has its own built-in
#   work-stealing scheduler that needs only -pthread — no Cilk dependency at all.
#   The baseline binary uses that scheduler. The OMP binary uses -fopenmp + AVX2.
#
# Targets:
#   make all          Build both binaries           → bin/sssp_baseline  bin/sssp_omp
#   make baseline     Build only the baseline       → bin/sssp_baseline
#   make omp          Build only the OpenMP version → bin/sssp_omp
#   make symmetrize   Build the symmetrize utility  → bin/symmetrize
#   make compare      Run side-by-side benchmark    (requires GRAPH=<path>)
#   make run-baseline Run baseline only             (requires GRAPH=<path>)
#   make run-omp      Run OpenMP version only       (requires GRAPH=<path>)
#   make clean        Remove all build artifacts
#   make help         Print this message
#
# Benchmark variables (pass on command line):
#   GRAPH        Path to the graph file          (required for run-* / compare)
#   ALGO         Algorithm name                  (default: rho-stepping)
#   PARAM        Algorithm parameter             (default: empty = built-in default)
#   THREADS      Number of OpenMP threads        (default: all logical CPUs)
#   WEIGHTED     Set to 1 for weighted graphs    (default: 0)
#   SYMMETRIZED  Set to 1 for symmetric graphs   (default: 0)
#
# Examples:
#   make all
#   make compare GRAPH=data/road.adj ALGO=delta-stepping PARAM=32768
#   make compare GRAPH=data/twitter.adj WEIGHTED=1 SYMMETRIZED=1 THREADS=16
#   make run-omp GRAPH=data/test.adj THREADS=8
# ==============================================================================

# ── Detect logical CPU count for default thread setting ───────────────────────
NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# ── User-configurable benchmark variables ─────────────────────────────────────
GRAPH       ?=
ALGO        ?= rho-stepping
PARAM       ?=
THREADS     ?= $(NPROC)
WEIGHTED    ?= 0
SYMMETRIZED ?= 0

# ── Compiler ──────────────────────────────────────────────────────────────────
CXX  := g++
STD  := -std=c++17
OPT  := -O3
WARN := -Wall -Wextra -Wno-unused-parameter

# ParlayLib include path — submodule is expected at parlaylib/ inside the project.
# If yours lives elsewhere, change this one line.
PARLAY_INC := -I parlaylib/include

# ─────────────────────────────────────────────────────────────────────────────
# BASELINE flags
#   Parlay's own work-stealing scheduler — no Cilk, no OpenMP needed.
#   Just -pthread.  This is the correct way to build parlay on g++ >= 11.
# ─────────────────────────────────────────────────────────────────────────────
BASE_CXXFLAGS := $(STD) $(OPT) $(WARN) $(PARLAY_INC) -pthread
BASE_LDFLAGS  := -pthread

# ─────────────────────────────────────────────────────────────────────────────
# OPENMP flags
#   -DPARLAY_OPENMP  tells parlay to delegate its parallel_for to OpenMP,
#                    so both the baseline parlay calls AND the new omp pragmas
#                    all use the same OpenMP thread pool — no double-scheduling.
#   -fopenmp         enables #pragma omp directives.
#   -mavx2           enables AVX2 SIMD intrinsics (Optimization 5).
# ─────────────────────────────────────────────────────────────────────────────
OMP_CXXFLAGS := $(STD) $(OPT) $(WARN) $(PARLAY_INC) \
                -fopenmp -DPARLAY_OPENMP -mavx2
OMP_LDFLAGS  := -fopenmp

# ── Directory layout ──────────────────────────────────────────────────────────
BINDIR := bin
RESDIR := results

# ── Source / header lists ─────────────────────────────────────────────────────
BASE_SRC := sssp_baseline.cc
BASE_HDR := sssp_baseline.h graph.h dijkstra.h hashbag.h utils.h

OMP_SRC  := sssp_omp.cc
OMP_HDR  := sssp_omp.h  graph.h dijkstra.h hashbag.h utils.h

SYM_SRC  := symmetrize.cc

# ── Argument builder helpers ──────────────────────────────────────────────────
_WFLAG := $(if $(filter 1,$(WEIGHTED)),-w,)
_SFLAG := $(if $(filter 1,$(SYMMETRIZED)),-s,)
_PFLAG := $(if $(PARAM),-p $(PARAM),)
_ARGS  := -i $(GRAPH) -a $(ALGO) $(_PFLAG) $(_WFLAG) $(_SFLAG) -v

# ── Phony targets ─────────────────────────────────────────────────────────────
.PHONY: all baseline omp symmetrize compare run-baseline run-omp \
        clean help dirs check-graph

# ── Default target ────────────────────────────────────────────────────────────
all: dirs baseline omp
	@echo ""
	@echo "  ✓  Both binaries ready:"
	@echo "       $(BINDIR)/sssp_baseline   (parlay work-stealing scheduler)"
	@echo "       $(BINDIR)/sssp_omp        (OpenMP + AVX2, 5 optimizations)"
	@echo ""
	@echo "  Next:  make compare GRAPH=<path/to/graph>"

# ── Create output directories ─────────────────────────────────────────────────
dirs:
	@mkdir -p $(BINDIR) $(RESDIR)

# ── Baseline binary ───────────────────────────────────────────────────────────
baseline: dirs $(BINDIR)/sssp_baseline

$(BINDIR)/sssp_baseline: $(BASE_SRC) $(BASE_HDR)
	@echo "[BUILD] Baseline  →  $(BINDIR)/sssp_baseline"
	$(CXX) $(BASE_CXXFLAGS) -o $@ $(BASE_SRC) $(BASE_LDFLAGS)
	@echo "[OK]    $(BINDIR)/sssp_baseline"

# ── OpenMP binary ─────────────────────────────────────────────────────────────
omp: dirs $(BINDIR)/sssp_omp

$(BINDIR)/sssp_omp: $(OMP_SRC) $(OMP_HDR)
	@echo "[BUILD] OpenMP    →  $(BINDIR)/sssp_omp"
	$(CXX) $(OMP_CXXFLAGS) -o $@ $(OMP_SRC) $(OMP_LDFLAGS)
	@echo "[OK]    $(BINDIR)/sssp_omp"

# ── Symmetrize utility ────────────────────────────────────────────────────────
symmetrize: dirs $(BINDIR)/symmetrize

$(BINDIR)/symmetrize: $(SYM_SRC) graph.h utils.h
	@echo "[BUILD] symmetrize → $(BINDIR)/symmetrize"
	$(CXX) $(OMP_CXXFLAGS) -o $@ $(SYM_SRC) $(OMP_LDFLAGS)
	@echo "[OK]    $(BINDIR)/symmetrize"

# ── Guard: ensure GRAPH is set and the file exists ────────────────────────────
check-graph:
	@if [ -z "$(GRAPH)" ]; then \
	  echo ""; \
	  echo "  ERROR: GRAPH is not set."; \
	  echo "  Usage: make <target> GRAPH=path/to/graph.adj [ALGO=rho-stepping] [THREADS=8]"; \
	  echo ""; \
	  exit 1; \
	fi
	@if [ ! -f "$(GRAPH)" ]; then \
	  echo "  ERROR: Graph file '$(GRAPH)' not found."; \
	  exit 1; \
	fi

# ── Run baseline only ─────────────────────────────────────────────────────────
run-baseline: check-graph $(BINDIR)/sssp_baseline
	@echo ""
	@echo "════════════════════════════════════════════"
	@echo " BASELINE  |  Graph: $(GRAPH)"
	@echo " Algorithm : $(ALGO)  Param: $(if $(PARAM),$(PARAM),default)"
	@echo "════════════════════════════════════════════"
	@> $(RESDIR)/baseline.tsv
	./$(BINDIR)/sssp_baseline $(_ARGS)

# ── Run OpenMP only ───────────────────────────────────────────────────────────
run-omp: check-graph $(BINDIR)/sssp_omp
	@echo ""
	@echo "════════════════════════════════════════════"
	@echo " OPENMP  (threads=$(THREADS))  |  Graph: $(GRAPH)"
	@echo " Algorithm : $(ALGO)  Param: $(if $(PARAM),$(PARAM),default)"
	@echo "════════════════════════════════════════════"
	@> $(RESDIR)/omp.tsv
	OMP_NUM_THREADS=$(THREADS) ./$(BINDIR)/sssp_omp $(_ARGS)

# ── Side-by-side comparison (calls compare.sh) ───────────────────────────────
compare: check-graph $(BINDIR)/sssp_baseline $(BINDIR)/sssp_omp
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo " Side-by-Side: Baseline vs OpenMP"
	@echo " Graph      : $(GRAPH)"
	@echo " Algorithm  : $(ALGO)"
	@echo " Param      : $(if $(PARAM),$(PARAM),default)"
	@echo " OMP Threads: $(THREADS)"
	@echo "════════════════════════════════════════════════════════════════"
	@bash compare.sh \
			-i "$(GRAPH)" \
			-a "$(ALGO)" \
			$(if $(PARAM),-p $(PARAM),) \
			$(if $(filter 1,$(WEIGHTED)),-w,) \
			$(if $(filter 1,$(SYMMETRIZED)),-s,)

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	@echo "[CLEAN] Removing $(BINDIR)/ and $(RESDIR)/"
	rm -rf $(BINDIR) $(RESDIR)
	@echo "[OK]    Clean complete."

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Parallel SSSP Build System"
	@echo "  ────────────────────────────────────────────────────────────────"
	@echo "  make all              Build both binaries"
	@echo "  make baseline         Build only bin/sssp_baseline  (parlay)"
	@echo "  make omp              Build only bin/sssp_omp        (OpenMP+AVX2)"
	@echo "  make symmetrize       Build bin/symmetrize utility"
	@echo "  make run-baseline     Run baseline           (requires GRAPH=)"
	@echo "  make run-omp          Run OpenMP version     (requires GRAPH=)"
	@echo "  make compare          Full side-by-side run  (requires GRAPH=)"
	@echo "  make clean            Remove bin/ and results/"
	@echo ""
	@echo "  Variables:"
	@echo "    GRAPH=path/to/file.adj    Graph file (required for run/compare)"
	@echo "    ALGO=rho-stepping         rho-stepping | delta-stepping | bellman-ford"
	@echo "    PARAM=2000000             Algorithm parameter (rho or delta value)"
	@echo "    THREADS=16                OpenMP threads (default: auto-detected)"
	@echo "    WEIGHTED=1                Graph has pre-existing weights"
	@echo "    SYMMETRIZED=1             Graph is already symmetric"
	@echo ""
	@echo "  Examples:"
	@echo "    make all"
	@echo "    make compare GRAPH=data/road.adj ALGO=delta-stepping PARAM=32768"
	@echo "    make compare GRAPH=data/twitter.adj WEIGHTED=1 SYMMETRIZED=1 THREADS=32"
	@echo "    make run-omp GRAPH=data/test.adj THREADS=8"
	@echo ""
