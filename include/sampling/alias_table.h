#pragma once

#include "core/rng.h"
#include "core/types.h"

#include <string>
#include <vector>

namespace sjs {
namespace sampling {

class AliasTable {
 public:
  AliasTable() = default;

  void Clear() {
    prob_.clear();
    alias_.clear();
  }

  usize Size() const noexcept { return prob_.size(); }
  bool Empty() const noexcept { return prob_.empty(); }

  bool BuildFromU64(Span<const u64> weights, std::string* err = nullptr) {
    Clear();
    if (weights.empty()) {
      if (err) *err = "AliasTable::BuildFromU64: empty weight array";
      return false;
    }

    long double sum = 0.0L;
    for (usize i = 0; i < weights.size(); ++i) {
      sum += static_cast<long double>(weights[i]);
    }
    if (!(sum > 0.0L)) {
      if (err) *err = "AliasTable::BuildFromU64: all weights are zero";
      return false;
    }

    const usize n = weights.size();
    prob_.assign(n, 0.0);
    alias_.assign(n, 0U);

    std::vector<double> scaled(n, 0.0);
    for (usize i = 0; i < n; ++i) {
      scaled[i] = static_cast<double>(static_cast<long double>(weights[i]) *
                                      static_cast<long double>(n) / sum);
    }

    std::vector<u32> small;
    std::vector<u32> large;
    small.reserve(n);
    large.reserve(n);

    for (usize i = 0; i < n; ++i) {
      if (scaled[i] < 1.0) {
        small.push_back(static_cast<u32>(i));
      } else {
        large.push_back(static_cast<u32>(i));
      }
    }

    while (!small.empty() && !large.empty()) {
      const u32 s = small.back();
      small.pop_back();
      const u32 l = large.back();
      large.pop_back();

      prob_[s] = scaled[s];
      alias_[s] = l;
      scaled[l] = (scaled[l] + scaled[s]) - 1.0;
      if (scaled[l] < 1.0) {
        small.push_back(l);
      } else {
        large.push_back(l);
      }
    }

    for (u32 i : large) {
      prob_[i] = 1.0;
      alias_[i] = i;
    }
    for (u32 i : small) {
      prob_[i] = 1.0;
      alias_[i] = i;
    }
    return true;
  }

  usize Sample(Rng* rng) const noexcept {
    const u32 n = static_cast<u32>(prob_.size());
    if (n == 0 || rng == nullptr) return 0;
    const u32 i = rng->UniformU32(n);
    const double u = rng->NextDouble();
    return (u < prob_[i]) ? static_cast<usize>(i) : static_cast<usize>(alias_[i]);
  }

 private:
  std::vector<double> prob_;
  std::vector<u32> alias_;
};

}  // namespace sampling
}  // namespace sjs
