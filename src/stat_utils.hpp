#pragma once

#include "core/types.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace kexp {

struct SummaryStats {
  sjs::usize count = 0;
  double weight_sum = 0.0;
  double min = std::numeric_limits<double>::quiet_NaN();
  double max = std::numeric_limits<double>::quiet_NaN();
  double mean = std::numeric_limits<double>::quiet_NaN();
  double median = std::numeric_limits<double>::quiet_NaN();
  double p90 = std::numeric_limits<double>::quiet_NaN();
  double p99 = std::numeric_limits<double>::quiet_NaN();
};

inline double QuantileFromSorted(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) return std::numeric_limits<double>::quiet_NaN();
  if (q <= 0.0) return sorted.front();
  if (q >= 1.0) return sorted.back();
  const double pos = q * static_cast<double>(sorted.size() - 1);
  const sjs::usize i0 = static_cast<sjs::usize>(std::floor(pos));
  const sjs::usize i1 = static_cast<sjs::usize>(std::ceil(pos));
  const double frac = pos - static_cast<double>(i0);
  return sorted[i0] * (1.0 - frac) + sorted[i1] * frac;
}

inline double WeightedQuantile(std::vector<std::pair<double, double>> values, double q) {
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(values.begin(), values.end(), [](const auto& a, const auto& b) {
    if (a.first < b.first) return true;
    if (b.first < a.first) return false;
    return a.second < b.second;
  });
  long double total = 0.0L;
  for (const auto& [v, w] : values) total += static_cast<long double>(w);
  if (!(total > 0.0L)) return std::numeric_limits<double>::quiet_NaN();
  long double target = static_cast<long double>(q) * total;
  long double accum = 0.0L;
  for (const auto& [v, w] : values) {
    accum += static_cast<long double>(w);
    if (accum >= target) return v;
  }
  return values.back().first;
}

inline SummaryStats Summarize(const std::vector<double>& values) {
  SummaryStats out;
  std::vector<double> valid;
  valid.reserve(values.size());
  long double sum = 0.0L;
  for (double v : values) {
    if (std::isnan(v)) continue;
    valid.push_back(v);
    sum += static_cast<long double>(v);
  }
  out.count = valid.size();
  out.weight_sum = static_cast<double>(valid.size());
  if (valid.empty()) return out;

  std::sort(valid.begin(), valid.end());
  out.min = valid.front();
  out.max = valid.back();
  out.mean = static_cast<double>(sum / static_cast<long double>(valid.size()));
  out.median = QuantileFromSorted(valid, 0.50);
  out.p90 = QuantileFromSorted(valid, 0.90);
  out.p99 = QuantileFromSorted(valid, 0.99);
  return out;
}

inline SummaryStats SummarizeWeighted(const std::vector<double>& values, const std::vector<double>& weights) {
  SummaryStats out;
  std::vector<std::pair<double, double>> valid;
  const sjs::usize n = std::min(values.size(), weights.size());
  valid.reserve(n);
  long double wsum = 0.0L;
  long double wvsum = 0.0L;
  for (sjs::usize i = 0; i < n; ++i) {
    const double v = values[i];
    const double w = weights[i];
    if (std::isnan(v) || std::isnan(w) || !(w > 0.0)) continue;
    valid.emplace_back(v, w);
    wsum += static_cast<long double>(w);
    wvsum += static_cast<long double>(v) * static_cast<long double>(w);
  }
  out.count = valid.size();
  out.weight_sum = static_cast<double>(wsum);
  if (valid.empty()) return out;
  std::vector<double> raw;
  raw.reserve(valid.size());
  for (const auto& [v, w] : valid) raw.push_back(v);
  std::sort(raw.begin(), raw.end());
  out.min = raw.front();
  out.max = raw.back();
  out.mean = static_cast<double>(wvsum / wsum);
  out.median = WeightedQuantile(valid, 0.50);
  out.p90 = WeightedQuantile(valid, 0.90);
  out.p99 = WeightedQuantile(valid, 0.99);
  return out;
}

inline double SafeRatio(double num, double den) {
  if (!(den > 0.0)) return std::numeric_limits<double>::quiet_NaN();
  return num / den;
}

}  // namespace kexp
