CXX ?= g++
CXXFLAGS ?= -std=c++17 -O3 -DNDEBUG -Wall -Wextra -pedantic
CPPFLAGS ?= -DRS_OVER_SRJ_USE_UPSTREAM_ALIAS=1 -I. -Iinclude -Isrc
LDFLAGS ?=

BUILD_DIR := build
BIN_BENCH := $(BUILD_DIR)/kdtree_benchmark
BIN_SMOKE := $(BUILD_DIR)/isrjs_smoke

.PHONY: all clean

all: $(BIN_BENCH) $(BIN_SMOKE)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_BENCH): $(BUILD_DIR) src/benchmark.cpp src/dataset_io.hpp src/experiment_kernel.hpp src/output_utils.hpp src/stat_utils.hpp include/core/types.h include/core/rng.h include/sampling/alias_table.h include/baselines/baseline_api.h include/join/join_enumerator.h vendor/KDTree/utils/kdtree.hpp vendor/KDTree/utils/weighted_sampling.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ src/benchmark.cpp $(LDFLAGS)

$(BIN_SMOKE): $(BUILD_DIR) src/isrjs_smoke.cpp include/core/types.h include/core/rng.h include/sampling/alias_table.h include/baselines/baseline_api.h include/join/join_enumerator.h vendor/KDTree/isrjs_kds/isrjs_kds.hpp vendor/KDTree/utils/kdtree.hpp vendor/KDTree/utils/weighted_sampling.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ src/isrjs_smoke.cpp $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)
