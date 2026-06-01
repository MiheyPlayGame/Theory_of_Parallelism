# Task 6: heat equation, OpenACC + boost::program_options
# host: nvc++/pgc++ preferred; g++ -fopenacc allowed with warning
# gpu / multicore: require NVIDIA HPC SDK (nvc++ or pgc++)

ACC      ?= host
BUILD_DIR = build
SRC       = heat_solver.cpp

# Auto-detect HPC SDK under /opt/nvidia/hpc_sdk or ~/.local/nvidia/hpc_sdk if not on PATH
NVHPC_BIN := $(shell \
  if command -v nvc++ >/dev/null 2>&1; then command -v nvc++; \
  elif command -v pgc++ >/dev/null 2>&1; then command -v pgc++; \
  else \
    for root in /opt/nvidia/hpc_sdk "$$HOME/.local/nvidia/hpc_sdk" "$(abspath .deps/nvhpc)"; do \
      latest=$$(ls -d $$root/Linux_x86_64/*/compilers/bin 2>/dev/null | sort -V | tail -1); \
      if [ -n "$$latest" ] && [ -x "$$latest/nvc++" ]; then echo "$$latest/nvc++"; break; \
      elif [ -n "$$latest" ] && [ -x "$$latest/pgc++" ]; then echo "$$latest/pgc++"; break; \
      fi; \
    done; \
  fi)
NVHPC_CXX := $(NVHPC_BIN)

ifeq ($(ACC),host)
  ifneq ($(NVHPC_CXX),)
    CXX := $(NVHPC_CXX)
  else
    CXX := $(shell command -v g++ 2>/dev/null || echo g++)
  endif
else
  CXX := $(NVHPC_CXX)
endif

BOOST_ROOT ?= $(abspath .deps/boost/usr)
BOOST_LIB  ?= $(abspath .deps/boost2/usr/lib/x86_64-linux-gnu)

INCLUDES = -I"$(BOOST_ROOT)/include"
BOOST_A = .deps/boost2/usr/lib/x86_64-linux-gnu/libboost_program_options.a
ifneq ($(wildcard $(BOOST_A)),)
  LIBS = $(BOOST_A) -pthread
else
  LIBS = -L"$(BOOST_LIB)" -lboost_program_options -pthread
endif

COMMON_FLAGS = -O3 -std=c++17 $(INCLUDES)

IS_NVHPC := $(or $(findstring nvc++,$(CXX)),$(findstring pgc++,$(CXX)))

ifeq ($(IS_NVHPC),)
  ifneq ($(filter gpu multicore,$(ACC)),)
    $(error ACC=$(ACC) requires nvc++ or pgc++ (NVIDIA HPC SDK). Install: see scripts/install_nvhpc.sh)
  endif
  ACC_FLAGS = -fopenacc
  $(warning host build uses g++ OpenACC; install NVIDIA HPC SDK for -acc=host|gpu|multicore)
else
  ACC_FLAGS = -acc=$(ACC) -Minfo=accel
  # WSL: libcuda lives in /usr/lib/wsl/lib
  ifneq ($(wildcard /usr/lib/wsl/lib/libcuda.so),)
    export LD_LIBRARY_PATH := /usr/lib/wsl/lib$(if $(LD_LIBRARY_PATH),:$(LD_LIBRARY_PATH),)
  endif
endif

.PHONY: all clean dirs host gpu multicore bench verify report-deps check-nvhpc

all: dirs check-acc $(BUILD_DIR)/heat_base_$(ACC) $(BUILD_DIR)/heat_opt_$(ACC)

check-acc:
ifneq ($(filter gpu multicore,$(ACC)),)
	@if [ -z "$(IS_NVHPC)" ]; then \
	  echo "ERROR: ACC=$(ACC) needs nvc++/pgc++. Run: ./scripts/install_nvhpc.sh"; \
	  exit 1; \
	fi
endif

check-nvhpc:
	@echo "NVHPC_CXX=$(NVHPC_CXX)"
	@echo "CXX=$(CXX)"
	@$(CXX) --version 2>/dev/null | head -1 || echo "Compiler not found"

dirs:
	@mkdir -p $(BUILD_DIR) results

NVHPC_ENV := $(if $(wildcard /usr/lib/wsl/lib/libcuda.so),LD_LIBRARY_PATH=/usr/lib/wsl/lib$(if $(LD_LIBRARY_PATH),:$(LD_LIBRARY_PATH),),)

$(BUILD_DIR)/heat_base_$(ACC): $(SRC) Makefile
	@echo "Building $@ with $(CXX) $(ACC_FLAGS)"
	$(NVHPC_ENV) $(CXX) $(COMMON_FLAGS) $(ACC_FLAGS) -DHEAT_BASELINE=1 -o $@ $(SRC) $(LIBS)

$(BUILD_DIR)/heat_opt_$(ACC): $(SRC) Makefile
	@echo "Building $@ with $(CXX) $(ACC_FLAGS)"
	$(NVHPC_ENV) $(CXX) $(COMMON_FLAGS) $(ACC_FLAGS) -o $@ $(SRC) $(LIBS)

host:
	@$(MAKE) ACC=host all

gpu:
	@$(MAKE) ACC=gpu all

multicore:
	@$(MAKE) ACC=multicore all

bench: dirs host
	./scripts/benchmark.sh host

verify: dirs host
	@echo "=== 10x10 grid ===" | tee results/verify_10x10.txt
	$(BUILD_DIR)/heat_opt_host --size 10 --eps 1e-6 --max-iter 1000000 | tee -a results/verify_10x10.txt
	@echo "=== 13x13 grid ===" | tee results/verify_13x13.txt
	$(BUILD_DIR)/heat_opt_host --size 13 --eps 1e-6 --max-iter 1000000 | tee -a results/verify_13x13.txt

clean:
	rm -rf $(BUILD_DIR) results/*.dat results/*.png results/*.csv

report-deps:
	@echo "Compiler: $(CXX)"
	@echo "ACC=$(ACC)"
	@test -f $(BOOST_ROOT)/include/boost/program_options.hpp || \
	  (echo "Run: apt install libboost-program-options-dev OR extract .deps (see README)"; exit 1)
