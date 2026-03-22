#pragma once

#include "core/types.h"

#include <cmath>
#include <limits>
#include <random>

namespace sjs {

class Rng {
 public:
  explicit Rng(u64 seed = 0) : seed_(seed), eng_(seed) {}

  void Seed(u64 seed) {
    seed_ = seed;
    eng_.seed(seed);
  }

  u64 seed() const noexcept { return seed_; }

  u32 UniformU32(u32 upper_exclusive) {
    if (upper_exclusive == 0) return 0;
    std::uniform_int_distribution<u32> dist(0, upper_exclusive - 1);
    return dist(eng_);
  }

  u64 UniformU64(u64 upper_exclusive) {
    if (upper_exclusive == 0) return 0;
    std::uniform_int_distribution<u64> dist(0, upper_exclusive - 1);
    return dist(eng_);
  }

  double NextDouble() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double x = dist(eng_);
    // Guard the upper edge in case an implementation returns 1.0 exactly.
    if (x >= 1.0) {
      x = std::nextafter(1.0, 0.0);
    }
    return x;
  }

 private:
  u64 seed_ = 0;
  std::mt19937_64 eng_;
};

}  // namespace sjs
