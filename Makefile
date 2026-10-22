.DEFAULT_GOAL := test_cpu

CPPSTD ?= c++17
CXXFLAGS ?= -O2
OPENMP_FLAGS ?= -fopenmp -pthread
BUILD_DIR ?= build
SCFD_ROOT ?= ../scfd
NMFD_OPERATIONS_ROOT ?= ../nmfd-operations
NMFD_NEWTON_ROOT ?= ../nmfd-newton
NMFD_LINSOLVERS_ROOT ?= ../nmfd-linsolvers
JSON_INCLUDE ?= ../json
CUDA_ROOT ?= /usr/local/cuda
NVCC ?= $(CUDA_ROOT)/bin/nvcc
CUDA_ARCH ?= 75
NVCCFLAGS ?= -std=$(CPPSTD) -O2 --extended-lambda -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)
TEST_OMP_NUM_THREADS ?= 2
CLANG_FORMAT ?= clang-format-21

INCLUDES := -Iinclude -I$(NMFD_OPERATIONS_ROOT)/include -I$(SCFD_ROOT)/include
HEADERS := $(wildcard include/nmfd/time_steppers/*/*.h tests/*.h)
PUBLIC_HEADERS := $(wildcard include/nmfd/time_steppers/*/*.h)
EXPLICIT_TESTS := $(filter-out lorenz_simulation,$(patsubst tests/test_explicit_%.cpp,%,$(wildcard tests/test_explicit_*.cpp)))
TABLE_BINS := $(addprefix $(BUILD_DIR)/test_rk_tables_,$(addsuffix .bin,explicit implicit))
CPU_BINS := $(foreach backend,cpu cpu_omp,$(addprefix $(BUILD_DIR)/test_explicit_,$(addsuffix _$(backend).bin,$(EXPLICIT_TESTS))))
CUDA_BINS := $(addprefix $(BUILD_DIR)/test_explicit_,$(addsuffix _cuda.bin,$(EXPLICIT_TESTS)))
IMPLICIT_TESTS := $(patsubst tests/test_implicit_%.cpp,%,$(wildcard tests/test_implicit_*.cpp))
IMPLICIT_CPU_BINS := $(foreach backend,cpu cpu_omp,$(addprefix $(BUILD_DIR)/test_implicit_,$(addsuffix _$(backend).bin,$(IMPLICIT_TESTS))))
IMPLICIT_CUDA_BINS := $(addprefix $(BUILD_DIR)/test_implicit_,$(addsuffix _cuda.bin,$(IMPLICIT_TESTS)))
IMPLICIT_INCLUDES := $(INCLUDES) -I$(NMFD_NEWTON_ROOT)/include -I$(NMFD_LINSOLVERS_ROOT)/include
IMPLICIT_HEADERS := $(wildcard $(NMFD_NEWTON_ROOT)/include/nmfd/solvers/*.h $(NMFD_LINSOLVERS_ROOT)/include/nmfd/solvers/*.h)
SIMULATION_BINS := $(addprefix $(BUILD_DIR)/test_explicit_lorenz_simulation_,$(addsuffix .bin,cpu cpu_omp cuda))
BIN_TARGETS := $(notdir $(TABLE_BINS) $(CPU_BINS) $(CUDA_BINS) $(SIMULATION_BINS) $(IMPLICIT_CPU_BINS) $(IMPLICIT_CUDA_BINS))

.PHONY: check_dependencies test_tables test_cpu test_cuda check_headers_cpu check_headers_cuda format format-check $(BIN_TARGETS)
.PHONY: check_implicit_dependencies test_implicit_cpu test_implicit_cuda
$(BIN_TARGETS): %: $(BUILD_DIR)/%

check_dependencies:
	@test -f $(NMFD_OPERATIONS_ROOT)/include/nmfd/detail/vector_wrap.h || { echo 'Set NMFD_OPERATIONS_ROOT to the sibling NMFD-operations repository'; exit 1; }
	@test -f $(SCFD_ROOT)/include/scfd/backend/serial_cpu.h || { echo 'Set SCFD_ROOT to the sibling SCFD repository'; exit 1; }

check_implicit_dependencies: check_dependencies
	@test -f $(NMFD_NEWTON_ROOT)/include/nmfd/solvers/nonlinear_solver.h || { echo 'Set NMFD_NEWTON_ROOT for the implicit tests'; exit 1; }
	@test -f $(NMFD_LINSOLVERS_ROOT)/include/nmfd/solvers/gmres.h || { echo 'Set NMFD_LINSOLVERS_ROOT for the implicit tests'; exit 1; }

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/test_rk_tables_%.bin: tests/test_butcher_tables_%.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) -std=$(CPPSTD) $(CXXFLAGS) -Iinclude $< -o $@

$(BUILD_DIR)/test_explicit_%_cpu.bin: tests/test_explicit_%.cpp $(HEADERS) | check_dependencies $(BUILD_DIR)
	$(CXX) -std=$(CPPSTD) $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) $< $(OPENMP_FLAGS) -o $@

$(BUILD_DIR)/test_explicit_%_cpu_omp.bin: tests/test_explicit_%.cpp $(HEADERS) | check_dependencies $(BUILD_DIR)
	$(CXX) -std=$(CPPSTD) $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) -DTEST_VECTOR_BACKEND_OMP $< $(OPENMP_FLAGS) -o $@

$(BUILD_DIR)/test_implicit_%_cpu.bin: tests/test_implicit_%.cpp $(HEADERS) $(IMPLICIT_HEADERS) | check_implicit_dependencies $(BUILD_DIR)
	$(CXX) -std=$(CPPSTD) $(CXXFLAGS) $(CPPFLAGS) $(IMPLICIT_INCLUDES) $< $(OPENMP_FLAGS) -o $@

$(BUILD_DIR)/test_implicit_%_cpu_omp.bin: tests/test_implicit_%.cpp $(HEADERS) $(IMPLICIT_HEADERS) | check_implicit_dependencies $(BUILD_DIR)
	$(CXX) -std=$(CPPSTD) $(CXXFLAGS) $(CPPFLAGS) $(IMPLICIT_INCLUDES) -DTEST_VECTOR_BACKEND_OMP $< $(OPENMP_FLAGS) -o $@

$(SIMULATION_BINS): CPPFLAGS += -I$(JSON_INCLUDE)

$(BUILD_DIR)/cuda_reductions.o: $(NMFD_OPERATIONS_ROOT)/include/nmfd/operations/blas1/high_precision/cuda/gpu_reduction_ogita_kernels.cu | check_dependencies $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/test_explicit_%_cuda.bin: tests/test_explicit_%.cpp $(HEADERS) $(BUILD_DIR)/cuda_reductions.o | check_dependencies $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $(INCLUDES) -DTEST_VECTOR_BACKEND_CUDA -x cu -c $< -o $(@:.bin=.o)
	$(NVCC) $(NVCCFLAGS) $(@:.bin=.o) $(BUILD_DIR)/cuda_reductions.o -lcublas -lcurand -o $@

$(BUILD_DIR)/test_implicit_%_cuda.bin: tests/test_implicit_%.cpp $(HEADERS) $(IMPLICIT_HEADERS) $(BUILD_DIR)/cuda_reductions.o | check_implicit_dependencies $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $(IMPLICIT_INCLUDES) -DTEST_VECTOR_BACKEND_CUDA -x cu -c $< -o $(@:.bin=.o)
	$(NVCC) $(NVCCFLAGS) $(@:.bin=.o) $(BUILD_DIR)/cuda_reductions.o -lcublas -lcurand -o $@

test_tables: $(TABLE_BINS)
	@set -e; for test in $(TABLE_BINS); do "$$test"; done

test_cpu: test_tables $(CPU_BINS)
	@set -e; for test in $(CPU_BINS); do OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) "$$test"; done

test_cuda: $(CUDA_BINS)
	@set -e; for test in $(CUDA_BINS); do "$$test"; done

test_implicit_cpu: $(IMPLICIT_CPU_BINS)
	@set -e; for test in $(IMPLICIT_CPU_BINS); do OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) "$$test"; done

test_implicit_cuda: $(IMPLICIT_CUDA_BINS)
	@set -e; for test in $(IMPLICIT_CUDA_BINS); do "$$test"; done

check_headers_cpu: check_dependencies
	@set -e; for header in $(PUBLIC_HEADERS); do \
		echo "Checking $$header"; \
		printf '#include <%s>\nint main() {}\n' "$${header#include/}" | \
		$(CXX) -std=$(CPPSTD) $(CXXFLAGS) $(INCLUDES) $(OPENMP_FLAGS) -x c++ -fsyntax-only -; \
	done

check_headers_cuda: check_dependencies
	@set -e; for header in $(PUBLIC_HEADERS); do \
		echo "Checking $$header (CUDA)"; \
		printf '#include <%s>\nint main() {}\n' "$${header#include/}" | \
		$(NVCC) $(NVCCFLAGS) $(INCLUDES) -x cu -c -o /dev/null -; \
	done

format:
	find include tests -type f \( -name '*.h' -o -name '*.cpp' \) -exec $(CLANG_FORMAT) --style=file -i {} +

format-check:
	find include tests -type f \( -name '*.h' -o -name '*.cpp' \) -exec $(CLANG_FORMAT) --style=file --dry-run --Werror {} +
