# Task 6: heat equation, OpenACC + boost::program_options
# Prefer NVIDIA HPC SDK (nvc++/pgc++); fallback: g++ -fopenacc

CXX      ?= $(shell command -v nvc++ 2>/dev/null || command -v pgc++ 2>/dev/null || echo g++)
ACC      ?= host
BUILD_DIR = build
SRC       = heat_solver.cpp

BOOST_ROOT ?= $(abspath .deps/boost/usr)
BOOST_LIB  ?= $(abspath .deps/boost2/usr/lib/x86_64-linux-gnu)

INCLUDES = -I"$(BOOST_ROOT)/include"
# Prefer static Boost from .deps (path without spaces for wildcard)
BOOST_A = .deps/boost2/usr/lib/x86_64-linux-gnu/libboost_program_options.a
ifneq ($(wildcard $(BOOST_A)),)
  LIBS = $(BOOST_A) -pthread
else
  LIBS = -L"$(BOOST_LIB)" -lboost_program_options -pthread
endif

COMMON_FLAGS = -O3 -std=c++17 $(INCLUDES)
ACC_FLAGS    = -acc=$(ACC) -Minfo=all

# Detect compiler family
ifeq ($(findstring nvc++,$(CXX)),nvc++)
  ACC_FLAGS = -acc=$(ACC) -Minfo=all
else ifeq ($(findstring pgc++,$(CXX)),pgc++)
  ACC_FLAGS = -acc=$(ACC) -Minfo=all
else
  ACC_FLAGS = -fopenacc
  $(warning Using g++ OpenACC; for lab use nvc++/pgc++ with -acc=host|gpu|multicore)
endif

.PHONY: all clean dirs host gpu multicore bench verify report-deps

all: dirs $(BUILD_DIR)/heat_base_$(ACC) $(BUILD_DIR)/heat_opt_$(ACC)

dirs:
	@mkdir -p $(BUILD_DIR) results

$(BUILD_DIR)/heat_base_$(ACC): $(SRC) Makefile
	$(CXX) $(COMMON_FLAGS) $(ACC_FLAGS) -DHEAT_BASELINE=1 -o $@ $(SRC) $(LIBS)

$(BUILD_DIR)/heat_opt_$(ACC): $(SRC) Makefile
	$(CXX) $(COMMON_FLAGS) $(ACC_FLAGS) -o $@ $(SRC) $(LIBS)

host: ACC=host
host: all

gpu: ACC=gpu
gpu: all

multicore: ACC=multicore
multicore: all

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
