#pragma once

#include "baselines/baseline_api.h"
#include "core/rng.h"
#include "core/types.h"
#include "src/stat_utils.hpp"
#include "vendor/KDTree/utils/kdtree.hpp"
#include "vendor/KDTree/utils/weighted_sampling.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kexp {

using Clock = std::chrono::steady_clock;
using MsDouble = std::chrono::duration<double, std::milli>;

inline double ElapsedMs(const Clock::time_point& t0, const Clock::time_point& t1) {
  return std::chrono::duration_cast<MsDouble>(t1 - t0).count();
}

inline std::string ToFixed(double x, int digits = 6) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(digits) << x;
  return oss.str();
}

struct SummaryBundle {
  SummaryStats equal;
  SummaryStats weighted;
};

struct BBoxDepthStats {
  sjs::u32 depth = 0;
  sjs::u64 node_count = 0;
  sjs::u64 edge_count = 0;
  SummaryStats edge_lengths;
  SummaryStats volumes;
};

struct BuildMetrics {
  double t_lift_ms = 0.0;
  double t_tree_ms = 0.0;
  double t_build_ms = 0.0;
  sjs::u32 tree_height = 0;    // number of levels = max_depth + 1
  sjs::u32 max_depth = 0;      // root depth = 0
  double avg_node_depth = 0.0;
  sjs::u64 node_count = 0;
  std::vector<BBoxDepthStats> bbox_by_depth;
};

struct CountQueryStats {
  sjs::u64 w = 0;
  sjs::u64 visited = 0;
  sjs::u64 pruned = 0;
  sjs::u64 contained = 0;
  sjs::u64 partial = 0;
  sjs::u64 saved_prune = 0;
  sjs::u64 saved_contain = 0;
  sjs::u32 max_depth = 0;
};

struct SearchQueryStats {
  sjs::u64 visited = 0;
  sjs::u64 pruned = 0;
  sjs::u64 b_mid = 0;
  sjs::u64 b_leaf = 0;
  sjs::u64 b_subtree = 0;
  sjs::u64 m_point = 0;
  sjs::u64 m_subtree = 0;
  sjs::u64 m_total = 0;
  double rho = std::numeric_limits<double>::quiet_NaN();
  double sigma = std::numeric_limits<double>::quiet_NaN();
  double avg_subtree_mass = std::numeric_limits<double>::quiet_NaN();
};

struct CountPhaseMetrics {
  double t_count_query_ms = 0.0;
  double t_left_dist_ms = 0.0;
  double t_count_ms = 0.0;
  sjs::u64 exact_join_size = 0;
  double alpha_actual = 0.0;
  std::vector<CountQueryStats> per_left;

  SummaryStats w_summary;
  SummaryStats visited_summary;
  SummaryStats pruned_summary;
  SummaryStats contained_summary;
  SummaryStats partial_summary;
  SummaryStats saved_prune_summary;
  SummaryStats saved_contain_summary;
  SummaryStats max_depth_summary;
};

struct SearchPhaseMetrics {
  double t_search_scan_ms = 0.0;  // analysis-only full scan across all left boxes
  std::vector<SearchQueryStats> per_left;

  SummaryStats visited_equal;
  SummaryStats pruned_equal;
  SummaryStats b_mid_equal;
  SummaryStats b_leaf_equal;
  SummaryStats b_subtree_equal;
  SummaryStats m_total_equal;
  SummaryStats m_point_equal;
  SummaryStats m_subtree_equal;
  SummaryStats rho_equal;
  SummaryStats sigma_equal;
  SummaryStats avg_subtree_mass_equal;

  SummaryStats visited_weighted;
  SummaryStats pruned_weighted;
  SummaryStats b_mid_weighted;
  SummaryStats b_leaf_weighted;
  SummaryStats b_subtree_weighted;
  SummaryStats m_total_weighted;
  SummaryStats m_point_weighted;
  SummaryStats m_subtree_weighted;
  SummaryStats rho_weighted;
  SummaryStats sigma_weighted;
  SummaryStats avg_subtree_mass_weighted;
};

struct GroupStats {
  sjs::u64 size = 0;
  sjs::u64 k_point = 0;
  sjs::u64 k_subtree = 0;
};

struct SamplePhaseMetrics {
  double t_plan_left_ms = 0.0;
  double t_group_ms = 0.0;
  double t_decompose_ms = 0.0;
  double t_local_dist_ms = 0.0;
  double t_emit_ms = 0.0;
  double t_sample_ms = 0.0;
  double samples_per_sec = 0.0;
  double ns_per_sample = 0.0;

  sjs::u64 t = 0;
  sjs::u64 chunk_size = 0;
  sjs::u64 num_chunks = 0;
  sjs::u64 U_t = 0;
  sjs::u64 G_t = 0;
  double avg_repeat_global = 0.0;
  double avg_group_size_exec = 0.0;
  std::vector<GroupStats> groups;
  std::map<sjs::u64, sjs::u64> group_size_hist;
  std::vector<sjs::PairId> samples;
  std::uint64_t checksum = 1469598103934665603ULL;

  SummaryStats group_size_equal;
  SummaryStats group_size_sample_weighted;
  SummaryStats k_point_equal;
  SummaryStats k_point_sample_weighted;
  SummaryStats k_subtree_equal;
  SummaryStats k_subtree_sample_weighted;
};

struct CorrectnessMetrics {
  bool count_sum_matches = true;
  bool decompose_mass_matches = true;
  bool sample_valid = true;
  bool empty_output_consistent = true;
  bool materialized_join_matches = true;

  sjs::u64 materialized_join_size = 0;
  double pair_tv_to_uniform = std::numeric_limits<double>::quiet_NaN();
  double pair_linf_to_uniform = std::numeric_limits<double>::quiet_NaN();
  double left_tv_to_theory = std::numeric_limits<double>::quiet_NaN();
  sjs::u64 distinct_pairs_sampled = 0;
};

struct RunMetadata {
  std::string run_id;
  std::string network;
  std::string preset = "TH";
  std::string family;
  int dim = 0;
  sjs::u64 N = 0;
  double alpha_target = 0.0;
  double alpha_expected_est = 0.0;
  double pair_intersection_prob_est = 0.0;
  double coverage = 0.0;
  sjs::u64 data_seed = 0;
  sjs::u64 sample_seed = 0;
  sjs::u64 sample_t = 0;
  int data_rep = 0;
  bool materialize_join = false;
  int left_metrics_sample_rows = 1024;
  int group_metrics_sample_rows = 2048;
};

inline std::uint64_t Fnv1aMix(std::uint64_t h, std::uint64_t x) {
  constexpr std::uint64_t kPrime = 1099511628211ULL;
  h ^= x;
  h *= kPrime;
  return h;
}

template <class T>
inline T NextUp(T x) {
  if (!std::isfinite(x)) return x;
  return std::nextafter(x, std::numeric_limits<T>::infinity());
}

template <class T>
inline T NextDown(T x) {
  if (!std::isfinite(x)) return x;
  return std::nextafter(x, -std::numeric_limits<T>::infinity());
}

template <int D, class T>
inline bool PointInRange(const KDPoint<D, T>& p,
                         const std::array<T, D>& qlo,
                         const std::array<T, D>& qhi) noexcept {
  for (int d = 0; d < D; ++d) {
    const sjs::usize j = static_cast<sjs::usize>(d);
    const T v = p.coord[j];
    if (v < qlo[j] || v > qhi[j]) return false;
  }
  return true;
}

template <int D, class T>
inline bool Disjoint(const typename KDTree<D, T>::Node& nd,
                     const std::array<T, D>& qlo,
                     const std::array<T, D>& qhi) noexcept {
  for (int d = 0; d < D; ++d) {
    const sjs::usize j = static_cast<sjs::usize>(d);
    if (nd.hi[j] < qlo[j] || nd.lo[j] > qhi[j]) return true;
  }
  return false;
}

template <int D, class T>
inline bool Contained(const typename KDTree<D, T>::Node& nd,
                      const std::array<T, D>& qlo,
                      const std::array<T, D>& qhi) noexcept {
  for (int d = 0; d < D; ++d) {
    const sjs::usize j = static_cast<sjs::usize>(d);
    if (nd.lo[j] < qlo[j]) return false;
    if (nd.hi[j] > qhi[j]) return false;
  }
  return true;
}

template <int Dim, class T>
inline KDPoint<2 * Dim, T> EmbedRightBoxPoint(const sjs::Box<Dim, T>& b, std::uint32_t idx) {
  KDPoint<2 * Dim, T> p;
  for (int axis = 0; axis < Dim; ++axis) {
    p.coord[static_cast<sjs::usize>(axis)] = b.lo.v[static_cast<sjs::usize>(axis)];
    p.coord[static_cast<sjs::usize>(Dim + axis)] = b.hi.v[static_cast<sjs::usize>(axis)];
  }
  p.idx = idx;
  return p;
}

template <int Dim, class T>
inline void BuildClosedLiftedQuery(const sjs::Box<Dim, T>& r,
                                   std::array<T, 2 * Dim>* qlo,
                                   std::array<T, 2 * Dim>* qhi) {
  const T inf = std::numeric_limits<T>::infinity();
  for (int axis = 0; axis < Dim; ++axis) {
    (*qlo)[static_cast<sjs::usize>(axis)] = -inf;
    (*qhi)[static_cast<sjs::usize>(axis)] = NextDown<T>(r.hi.v[static_cast<sjs::usize>(axis)]);
    (*qlo)[static_cast<sjs::usize>(Dim + axis)] = NextUp<T>(r.lo.v[static_cast<sjs::usize>(axis)]);
    (*qhi)[static_cast<sjs::usize>(Dim + axis)] = inf;
  }
}

template <int D, class T>
struct SearchWorkItem {
  std::uint32_t node = KDTree<D, T>::kNull;
  bool fully_contained = false;
};

template <int D, class T>
sjs::u64 CountRecInstrumented(const KDTree<D, T>& tree,
                              std::uint32_t node,
                              const std::array<T, D>& qlo,
                              const std::array<T, D>& qhi,
                              std::uint32_t depth,
                              CountQueryStats* stats) {
  if (node == KDTree<D, T>::kNull) return 0;
  const auto& nd = tree.GetNode(node);
  stats->visited += 1;
  if (depth > stats->max_depth) stats->max_depth = depth;

  if (Disjoint<D, T>(nd, qlo, qhi)) {
    stats->pruned += 1;
    if (nd.Size() > 0) stats->saved_prune += static_cast<sjs::u64>(nd.Size() - 1);
    return 0;
  }
  if (Contained<D, T>(nd, qlo, qhi)) {
    stats->contained += 1;
    if (nd.Size() > 0) stats->saved_contain += static_cast<sjs::u64>(nd.Size() - 1);
    return static_cast<sjs::u64>(nd.Size());
  }

  sjs::u64 cnt = 0;
  if (PointInRange<D, T>(tree.Points()[nd.mid], qlo, qhi)) {
    cnt += 1;
  }
  if (nd.left != KDTree<D, T>::kNull) cnt += CountRecInstrumented<D, T>(tree, nd.left, qlo, qhi, depth + 1, stats);
  if (nd.right != KDTree<D, T>::kNull) cnt += CountRecInstrumented<D, T>(tree, nd.right, qlo, qhi, depth + 1, stats);
  return cnt;
}

template <int D, class T>
void SearchRecInstrumented(const KDTree<D, T>& tree,
                           std::uint32_t node,
                           const std::array<T, D>& qlo,
                           const std::array<T, D>& qhi,
                           std::vector<SearchWorkItem<D, T>>* items,
                           SearchQueryStats* stats) {
  if (node == KDTree<D, T>::kNull) return;
  const auto& nd = tree.GetNode(node);
  stats->visited += 1;

  if (Disjoint<D, T>(nd, qlo, qhi)) {
    stats->pruned += 1;
    return;
  }

  if (nd.IsLeaf()) {
    items->push_back(SearchWorkItem<D, T>{node, false});
    stats->b_leaf += 1;
    stats->m_point += 1;
    stats->m_total += 1;
    return;
  }

  if (PointInRange<D, T>(tree.Points()[nd.mid], qlo, qhi)) {
    items->push_back(SearchWorkItem<D, T>{node, false});
    stats->b_mid += 1;
    stats->m_point += 1;
    stats->m_total += 1;
  }

  if (nd.left != KDTree<D, T>::kNull) {
    const auto& c = tree.GetNode(nd.left);
    if (Contained<D, T>(c, qlo, qhi)) {
      items->push_back(SearchWorkItem<D, T>{nd.left, true});
      stats->b_subtree += 1;
      stats->m_subtree += static_cast<sjs::u64>(c.Size());
      stats->m_total += 1;
    } else {
      SearchRecInstrumented<D, T>(tree, nd.left, qlo, qhi, items, stats);
    }
  }
  if (nd.right != KDTree<D, T>::kNull) {
    const auto& c = tree.GetNode(nd.right);
    if (Contained<D, T>(c, qlo, qhi)) {
      items->push_back(SearchWorkItem<D, T>{nd.right, true});
      stats->b_subtree += 1;
      stats->m_subtree += static_cast<sjs::u64>(c.Size());
      stats->m_total += 1;
    } else {
      SearchRecInstrumented<D, T>(tree, nd.right, qlo, qhi, items, stats);
    }
  }
}

template <int Dim>
class ExperimentKernel {
 public:
  using T = float;
  static constexpr int D = 2 * Dim;
  using DatasetT = sjs::Dataset<Dim, T>;
  using BoxT = sjs::Box<Dim, T>;
  using TreeT = KDTree<D, T>;

  explicit ExperimentKernel(const DatasetT& ds) : ds_(ds) {}

  bool Build(std::string* err = nullptr) {
    (void)err;
    const auto t0 = Clock::now();
    const auto t_lift0 = Clock::now();
    std::vector<KDPoint<D, T>> pts;
    pts.reserve(ds_.S.Size());
    for (sjs::usize i = 0; i < ds_.S.boxes.size(); ++i) {
      pts.push_back(EmbedRightBoxPoint<Dim, T>(ds_.S.boxes[i], static_cast<std::uint32_t>(i)));
    }
    const auto t_lift1 = Clock::now();
    build_.t_lift_ms = ElapsedMs(t_lift0, t_lift1);

    const auto t_tree0 = Clock::now();
    tree_.Build(std::move(pts));
    const auto t_tree1 = Clock::now();
    build_.t_tree_ms = ElapsedMs(t_tree0, t_tree1);
    build_.t_build_ms = ElapsedMs(t0, t_tree1);
    ComputeBuildMetrics();
    built_ = true;
    return true;
  }

  bool RunCount(std::string* err = nullptr) {
    if (!built_) {
      if (err) *err = "RunCount: call Build() first";
      return false;
    }
    const auto t0 = Clock::now();
    const auto t_query0 = Clock::now();

    count_.per_left.assign(ds_.R.Size(), CountQueryStats{});
    w_r_.assign(ds_.R.Size(), 0ULL);
    sjs::u64 W = 0;

    for (sjs::usize i = 0; i < ds_.R.boxes.size(); ++i) {
      const auto& r = ds_.R.boxes[i];
      std::array<T, D> qlo{};
      std::array<T, D> qhi{};
      BuildClosedLiftedQuery<Dim, T>(r, &qlo, &qhi);
      auto& st = count_.per_left[i];
      st.w = CountRecInstrumented<D, T>(tree_, tree_.Root(), qlo, qhi, 0, &st);
      st.partial = st.visited - st.pruned - st.contained;
      w_r_[i] = st.w;
      if (st.w > 0) {
        if (W > std::numeric_limits<sjs::u64>::max() - st.w) {
          if (err) *err = "RunCount: join size overflowed u64";
          return false;
        }
        W += st.w;
      }
    }

    const auto t_query1 = Clock::now();
    count_.t_count_query_ms = ElapsedMs(t_query0, t_query1);
    count_.exact_join_size = W;
    count_.alpha_actual = ds_.R.Size() + ds_.S.Size() > 0
                              ? static_cast<double>(W) /
                                    static_cast<double>(ds_.R.Size() + ds_.S.Size())
                              : 0.0;

    const auto t_alias0 = Clock::now();
    alias_r_.Clear();
    if (W > 0) {
      if (!alias_r_.Build(w_r_, err)) {
        if (err && err->empty()) *err = "RunCount: failed to build left alias table";
        return false;
      }
    }
    const auto t_alias1 = Clock::now();
    count_.t_left_dist_ms = ElapsedMs(t_alias0, t_alias1);
    count_.t_count_ms = ElapsedMs(t0, t_alias1);
    W_ = W;
    count_ready_ = true;
    SummarizeCountMetrics();
    return true;
  }

  bool RunFullSearchScan(std::string* err = nullptr) {
    if (!count_ready_) {
      if (err) *err = "RunFullSearchScan: call RunCount() first";
      return false;
    }
    const auto t0 = Clock::now();
    search_.per_left.assign(ds_.R.Size(), SearchQueryStats{});
    std::vector<SearchWorkItem<D, T>> items;
    items.reserve(256);

    correctness_.decompose_mass_matches = true;
    for (sjs::usize i = 0; i < ds_.R.boxes.size(); ++i) {
      const auto& r = ds_.R.boxes[i];
      std::array<T, D> qlo{};
      std::array<T, D> qhi{};
      BuildClosedLiftedQuery<Dim, T>(r, &qlo, &qhi);
      auto& st = search_.per_left[i];
      items.clear();
      if (!tree_.Empty()) {
        SearchRecInstrumented<D, T>(tree_, tree_.Root(), qlo, qhi, &items, &st);
      }
      if (st.m_total > 0) {
        st.rho = SafeRatio(static_cast<double>(w_r_[i]), static_cast<double>(st.m_total));
      }
      if (w_r_[i] > 0) {
        st.sigma = SafeRatio(static_cast<double>(st.m_subtree), static_cast<double>(w_r_[i]));
      }
      if (st.b_subtree > 0) {
        st.avg_subtree_mass = SafeRatio(static_cast<double>(st.m_subtree), static_cast<double>(st.b_subtree));
      }
      const sjs::u64 mass = st.m_point + st.m_subtree;
      if (mass != w_r_[i]) {
        correctness_.decompose_mass_matches = false;
      }
    }
    const auto t1 = Clock::now();
    search_.t_search_scan_ms = ElapsedMs(t0, t1);
    SummarizeSearchMetrics();
    return true;
  }

  bool RunSample(const RunMetadata& meta, std::string* err = nullptr) {
    if (!count_ready_) {
      if (err) *err = "RunSample: call RunCount() first";
      return false;
    }
    sample_ = SamplePhaseMetrics{};
    sample_.t = meta.sample_t;
    sample_.chunk_size = std::max<sjs::u64>(1ULL, 2'000'000ULL);
    sample_.samples.clear();
    sample_.samples.reserve(static_cast<sjs::usize>(meta.sample_t));

    if (meta.sample_t == 0) {
      correctness_.empty_output_consistent = true;
      return true;
    }
    if (W_ == 0) {
      correctness_.empty_output_consistent = sample_.samples.empty();
      return true;
    }
    if (meta.sample_t > static_cast<sjs::u64>(std::numeric_limits<sjs::u32>::max())) {
      if (err) *err = "RunSample: t too large for u32 chunk slot indexing";
      return false;
    }

    sjs::Rng rng(meta.sample_seed);
    std::vector<sjs::u8> seen_left(ds_.R.Size(), 0);
    const auto t0 = Clock::now();

    sjs::u64 emitted_total = 0;
    while (emitted_total < meta.sample_t) {
      const sjs::u64 chunk_t = std::min(sample_.chunk_size, meta.sample_t - emitted_total);
      sample_.num_chunks += 1;

      struct SlotAssign {
        sjs::u32 ridx = 0;
        sjs::u32 slot = 0;
      };
      std::vector<SlotAssign> asg;
      asg.reserve(static_cast<sjs::usize>(chunk_t));

      const auto t_plan0 = Clock::now();
      for (sjs::u32 slot = 0; slot < static_cast<sjs::u32>(chunk_t); ++slot) {
        const sjs::u32 ridx = static_cast<sjs::u32>(alias_r_.Sample(&rng));
        asg.push_back(SlotAssign{ridx, slot});
        if (!seen_left[static_cast<sjs::usize>(ridx)]) {
          seen_left[static_cast<sjs::usize>(ridx)] = 1;
          sample_.U_t += 1;
        }
      }
      const auto t_plan1 = Clock::now();
      sample_.t_plan_left_ms += ElapsedMs(t_plan0, t_plan1);

      const auto t_group0 = Clock::now();
      std::sort(asg.begin(), asg.end(), [](const SlotAssign& a, const SlotAssign& b) {
        if (a.ridx < b.ridx) return true;
        if (b.ridx < a.ridx) return false;
        return a.slot < b.slot;
      });
      const auto t_group1 = Clock::now();
      sample_.t_group_ms += ElapsedMs(t_group0, t_group1);

      std::vector<SearchWorkItem<D, T>> items;
      items.reserve(256);
      std::vector<sjs::u64> weights;
      weights.reserve(256);

      sjs::usize ptr = 0;
      std::vector<sjs::PairId> chunk_pairs(static_cast<sjs::usize>(chunk_t));
      while (ptr < asg.size()) {
        const sjs::u32 ridx = asg[ptr].ridx;
        sjs::usize end = ptr;
        while (end < asg.size() && asg[end].ridx == ridx) ++end;
        const sjs::u64 k = static_cast<sjs::u64>(end - ptr);
        sample_.G_t += 1;

        const auto& r = ds_.R.boxes[static_cast<sjs::usize>(ridx)];
        std::array<T, D> qlo{};
        std::array<T, D> qhi{};
        BuildClosedLiftedQuery<Dim, T>(r, &qlo, &qhi);

        const auto t_dec0 = Clock::now();
        items.clear();
        SearchQueryStats st{};
        SearchRecInstrumented<D, T>(tree_, tree_.Root(), qlo, qhi, &items, &st);
        const auto t_dec1 = Clock::now();
        sample_.t_decompose_ms += ElapsedMs(t_dec0, t_dec1);
        if (items.empty()) {
          if (err) *err = "RunSample: empty decomposition for sampled left box";
          return false;
        }

        const auto t_local0 = Clock::now();
        weights.clear();
        weights.reserve(items.size());
        for (const auto& it : items) {
          if (it.fully_contained) {
            weights.push_back(static_cast<sjs::u64>(tree_.GetNode(it.node).Size()));
          } else {
            weights.push_back(1ULL);
          }
        }
        ::alias alias_q;
        if (!alias_q.Build(weights, err)) {
          if (err && err->empty()) *err = "RunSample: failed to build per-query alias";
          return false;
        }
        const auto t_local1 = Clock::now();
        sample_.t_local_dist_ms += ElapsedMs(t_local0, t_local1);

        GroupStats gst{};
        gst.size = k;
        const auto t_emit0 = Clock::now();
        for (sjs::u64 j = 0; j < k; ++j) {
          const sjs::usize ii = alias_q.Sample(&rng);
          if (ii >= items.size()) {
            if (err) *err = "RunSample: alias sampled out of range";
            return false;
          }
          const auto& it = items[ii];
          std::uint32_t s_idx = 0;
          if (it.fully_contained) {
            const auto& nd = tree_.GetNode(it.node);
            const sjs::u64 off = rng.UniformU64(static_cast<sjs::u64>(nd.Size()));
            const std::uint32_t pos = nd.l + static_cast<std::uint32_t>(off);
            s_idx = tree_.Points()[static_cast<sjs::usize>(pos)].idx;
            gst.k_subtree += 1;
          } else {
            s_idx = tree_.NodePointPayload(it.node);
            gst.k_point += 1;
          }
          const sjs::u32 slot = asg[ptr + static_cast<sjs::usize>(j)].slot;
          chunk_pairs[static_cast<sjs::usize>(slot)] =
              sjs::PairId{ds_.R.GetId(static_cast<sjs::usize>(ridx)), ds_.S.GetId(static_cast<sjs::usize>(s_idx))};
        }
        const auto t_emit1 = Clock::now();
        sample_.t_emit_ms += ElapsedMs(t_emit0, t_emit1);

        sample_.groups.push_back(gst);
        sample_.group_size_hist[gst.size] += 1;
        ptr = end;
      }

      for (const auto& p : chunk_pairs) {
        sample_.checksum = Fnv1aMix(sample_.checksum, (static_cast<std::uint64_t>(p.r) << 32) ^ p.s);
      }
      sample_.samples.insert(sample_.samples.end(), chunk_pairs.begin(), chunk_pairs.end());
      emitted_total += chunk_t;
    }

    const auto t1 = Clock::now();
    sample_.t_sample_ms = ElapsedMs(t0, t1);
    if (sample_.t_sample_ms > 0.0 && meta.sample_t > 0) {
      sample_.samples_per_sec = static_cast<double>(meta.sample_t) / (sample_.t_sample_ms / 1000.0);
      sample_.ns_per_sample = (1.0e6 * sample_.t_sample_ms) / static_cast<double>(meta.sample_t);
    }
    sample_.avg_repeat_global = sample_.U_t > 0 ? static_cast<double>(meta.sample_t) / static_cast<double>(sample_.U_t) : 0.0;
    sample_.avg_group_size_exec = sample_.G_t > 0 ? static_cast<double>(meta.sample_t) / static_cast<double>(sample_.G_t) : 0.0;
    SummarizeSampleMetrics();
    correctness_.empty_output_consistent = !(W_ == 0 && !sample_.samples.empty());
    ValidateSamplePairs();
    return true;
  }

  void RunCorrectnessMaterialization() {
    correctness_.count_sum_matches = (std::accumulate(w_r_.begin(), w_r_.end(), 0ULL) == W_);
    if (!correctness_input_materialize_) return;

    std::vector<std::uint64_t> join_keys;
    join_keys.reserve(static_cast<sjs::usize>(W_));
    for (sjs::usize i = 0; i < ds_.R.boxes.size(); ++i) {
      const auto& r = ds_.R.boxes[i];
      for (sjs::usize j = 0; j < ds_.S.boxes.size(); ++j) {
        if (r.Intersects(ds_.S.boxes[j])) {
          join_keys.push_back((static_cast<std::uint64_t>(i) << 32) | static_cast<std::uint64_t>(j));
        }
      }
    }
    std::sort(join_keys.begin(), join_keys.end());
    correctness_.materialized_join_size = static_cast<sjs::u64>(join_keys.size());
    correctness_.materialized_join_matches = (correctness_.materialized_join_size == W_);
    if (sample_.samples.empty() || join_keys.empty()) return;

    std::vector<sjs::u64> pair_counts(join_keys.size(), 0ULL);
    std::vector<sjs::u64> left_counts(ds_.R.Size(), 0ULL);
    sjs::u64 distinct_pairs = 0;
    for (const auto& p : sample_.samples) {
      const std::uint64_t key = (static_cast<std::uint64_t>(p.r) << 32) | static_cast<std::uint64_t>(p.s);
      auto it = std::lower_bound(join_keys.begin(), join_keys.end(), key);
      if (it != join_keys.end() && *it == key) {
        const sjs::usize pos = static_cast<sjs::usize>(it - join_keys.begin());
        if (pair_counts[pos] == 0) distinct_pairs += 1;
        pair_counts[pos] += 1;
      } else {
        correctness_.materialized_join_matches = false;
      }
      if (p.r < left_counts.size()) left_counts[p.r] += 1;
    }
    correctness_.distinct_pairs_sampled = distinct_pairs;

    const double t = static_cast<double>(sample_.samples.size());
    const double W = static_cast<double>(join_keys.size());
    if (t > 0.0 && W > 0.0) {
      const double uniform_p = 1.0 / W;
      double tv = 0.0;
      double linf = 0.0;
      for (sjs::u64 c : pair_counts) {
        const double p_hat = static_cast<double>(c) / t;
        const double diff = std::abs(p_hat - uniform_p);
        tv += diff;
        if (diff > linf) linf = diff;
      }
      correctness_.pair_tv_to_uniform = 0.5 * tv;
      correctness_.pair_linf_to_uniform = linf;

      double left_tv = 0.0;
      for (sjs::usize i = 0; i < left_counts.size(); ++i) {
        const double empirical = static_cast<double>(left_counts[i]) / t;
        const double theory = W_ > 0 ? static_cast<double>(w_r_[i]) / static_cast<double>(W_) : 0.0;
        left_tv += std::abs(empirical - theory);
      }
      correctness_.left_tv_to_theory = 0.5 * left_tv;
    }
  }

  void SetCorrectnessMaterialize(bool v) { correctness_input_materialize_ = v; }

  const BuildMetrics& build() const noexcept { return build_; }
  const CountPhaseMetrics& count() const noexcept { return count_; }
  const SearchPhaseMetrics& search() const noexcept { return search_; }
  const SamplePhaseMetrics& sample() const noexcept { return sample_; }
  const CorrectnessMetrics& correctness() const noexcept { return correctness_; }
  const std::vector<sjs::u64>& left_weights() const noexcept { return w_r_; }
  sjs::u64 exact_join_size() const noexcept { return W_; }

 private:
  void ComputeBuildMetrics() {
    build_.node_count = tree_.Nodes().size();
    if (tree_.Nodes().empty()) {
      build_.tree_height = 0;
      build_.max_depth = 0;
      build_.avg_node_depth = 0.0;
      return;
    }

    std::vector<sjs::u32> depths(tree_.Nodes().size(), 0U);
    std::vector<std::vector<double>> edge_by_depth;
    std::vector<std::vector<double>> vol_by_depth;
    edge_by_depth.resize(1);
    vol_by_depth.resize(1);

    std::function<void(std::uint32_t, sjs::u32)> dfs = [&](std::uint32_t node, sjs::u32 depth) {
      if (node == TreeT::kNull) return;
      depths[node] = depth;
      if (depth >= edge_by_depth.size()) {
        edge_by_depth.resize(static_cast<sjs::usize>(depth + 1));
        vol_by_depth.resize(static_cast<sjs::usize>(depth + 1));
      }
      const auto& nd = tree_.GetNode(node);
      double vol = 1.0;
      for (int axis = 0; axis < D; ++axis) {
        const sjs::usize j = static_cast<sjs::usize>(axis);
        const double len = static_cast<double>(nd.hi[j]) - static_cast<double>(nd.lo[j]);
        edge_by_depth[depth].push_back(len);
        vol *= std::max(0.0, len);
      }
      vol_by_depth[depth].push_back(vol);
      if (nd.left != TreeT::kNull) dfs(nd.left, depth + 1);
      if (nd.right != TreeT::kNull) dfs(nd.right, depth + 1);
    };
    dfs(tree_.Root(), 0);

    build_.max_depth = *std::max_element(depths.begin(), depths.end());
    build_.tree_height = build_.max_depth + 1;
    build_.avg_node_depth = std::accumulate(depths.begin(), depths.end(), 0.0) /
                            static_cast<double>(depths.size());
    build_.bbox_by_depth.clear();
    for (sjs::u32 depth = 0; depth < edge_by_depth.size(); ++depth) {
      BBoxDepthStats row;
      row.depth = depth;
      row.node_count = static_cast<sjs::u64>(vol_by_depth[depth].size());
      row.edge_count = static_cast<sjs::u64>(edge_by_depth[depth].size());
      row.edge_lengths = Summarize(edge_by_depth[depth]);
      row.volumes = Summarize(vol_by_depth[depth]);
      build_.bbox_by_depth.push_back(row);
    }
  }

  void SummarizeCountMetrics() {
    std::vector<double> w;
    std::vector<double> visited;
    std::vector<double> pruned;
    std::vector<double> contained;
    std::vector<double> partial;
    std::vector<double> saved_prune;
    std::vector<double> saved_contain;
    std::vector<double> max_depth;
    const sjs::usize n = count_.per_left.size();
    w.reserve(n);
    visited.reserve(n);
    pruned.reserve(n);
    contained.reserve(n);
    partial.reserve(n);
    saved_prune.reserve(n);
    saved_contain.reserve(n);
    max_depth.reserve(n);
    for (const auto& st : count_.per_left) {
      w.push_back(static_cast<double>(st.w));
      visited.push_back(static_cast<double>(st.visited));
      pruned.push_back(static_cast<double>(st.pruned));
      contained.push_back(static_cast<double>(st.contained));
      partial.push_back(static_cast<double>(st.partial));
      saved_prune.push_back(static_cast<double>(st.saved_prune));
      saved_contain.push_back(static_cast<double>(st.saved_contain));
      max_depth.push_back(static_cast<double>(st.max_depth));
    }
    count_.w_summary = Summarize(w);
    count_.visited_summary = Summarize(visited);
    count_.pruned_summary = Summarize(pruned);
    count_.contained_summary = Summarize(contained);
    count_.partial_summary = Summarize(partial);
    count_.saved_prune_summary = Summarize(saved_prune);
    count_.saved_contain_summary = Summarize(saved_contain);
    count_.max_depth_summary = Summarize(max_depth);
    correctness_.count_sum_matches = (std::accumulate(w_r_.begin(), w_r_.end(), 0ULL) == W_);
  }

  void SummarizeSearchMetrics() {
    std::vector<double> weights;
    weights.reserve(w_r_.size());
    for (sjs::u64 w : w_r_) weights.push_back(static_cast<double>(w));

    auto gather = [&](auto getter) {
      std::vector<double> out;
      out.reserve(search_.per_left.size());
      for (const auto& st : search_.per_left) out.push_back(getter(st));
      return out;
    };

    auto fill = [&](SummaryStats* eq, SummaryStats* wt, const std::vector<double>& values) {
      *eq = Summarize(values);
      *wt = SummarizeWeighted(values, weights);
    };

    fill(&search_.visited_equal, &search_.visited_weighted, gather([](const auto& st) { return static_cast<double>(st.visited); }));
    fill(&search_.pruned_equal, &search_.pruned_weighted, gather([](const auto& st) { return static_cast<double>(st.pruned); }));
    fill(&search_.b_mid_equal, &search_.b_mid_weighted, gather([](const auto& st) { return static_cast<double>(st.b_mid); }));
    fill(&search_.b_leaf_equal, &search_.b_leaf_weighted, gather([](const auto& st) { return static_cast<double>(st.b_leaf); }));
    fill(&search_.b_subtree_equal, &search_.b_subtree_weighted, gather([](const auto& st) { return static_cast<double>(st.b_subtree); }));
    fill(&search_.m_total_equal, &search_.m_total_weighted, gather([](const auto& st) { return static_cast<double>(st.m_total); }));
    fill(&search_.m_point_equal, &search_.m_point_weighted, gather([](const auto& st) { return static_cast<double>(st.m_point); }));
    fill(&search_.m_subtree_equal, &search_.m_subtree_weighted, gather([](const auto& st) { return static_cast<double>(st.m_subtree); }));
    fill(&search_.rho_equal, &search_.rho_weighted, gather([](const auto& st) { return st.rho; }));
    fill(&search_.sigma_equal, &search_.sigma_weighted, gather([](const auto& st) { return st.sigma; }));
    fill(&search_.avg_subtree_mass_equal, &search_.avg_subtree_mass_weighted,
         gather([](const auto& st) { return st.avg_subtree_mass; }));
  }

  void SummarizeSampleMetrics() {
    auto gather = [&](auto getter) {
      std::vector<double> out;
      out.reserve(sample_.groups.size());
      for (const auto& g : sample_.groups) out.push_back(getter(g));
      return out;
    };
    std::vector<double> group_weights = gather([](const auto& g) { return static_cast<double>(g.size); });
    const std::vector<double> sizes = gather([](const auto& g) { return static_cast<double>(g.size); });
    const std::vector<double> kpoint = gather([](const auto& g) { return static_cast<double>(g.k_point); });
    const std::vector<double> ksub = gather([](const auto& g) { return static_cast<double>(g.k_subtree); });

    sample_.group_size_equal = Summarize(sizes);
    sample_.group_size_sample_weighted = SummarizeWeighted(sizes, group_weights);
    sample_.k_point_equal = Summarize(kpoint);
    sample_.k_point_sample_weighted = SummarizeWeighted(kpoint, group_weights);
    sample_.k_subtree_equal = Summarize(ksub);
    sample_.k_subtree_sample_weighted = SummarizeWeighted(ksub, group_weights);
  }

  void ValidateSamplePairs() {
    correctness_.sample_valid = true;
    for (const auto& p : sample_.samples) {
      if (p.r >= ds_.R.Size() || p.s >= ds_.S.Size()) {
        correctness_.sample_valid = false;
        return;
      }
      if (!ds_.R.boxes[p.r].Intersects(ds_.S.boxes[p.s])) {
        correctness_.sample_valid = false;
        return;
      }
    }
  }

  const DatasetT& ds_;
  TreeT tree_;
  BuildMetrics build_{};
  CountPhaseMetrics count_{};
  SearchPhaseMetrics search_{};
  SamplePhaseMetrics sample_{};
  CorrectnessMetrics correctness_{};

  bool built_ = false;
  bool count_ready_ = false;
  bool correctness_input_materialize_ = false;
  ::alias alias_r_;
  std::vector<sjs::u64> w_r_;
  sjs::u64 W_ = 0;
};

}  // namespace kexp
