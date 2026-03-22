#pragma once

#include "src/experiment_kernel.hpp"

#include <fstream>
#include <string>

namespace kexp {


inline std::string JsonNum(double x) {
  if (!std::isfinite(x)) return "null";
  std::ostringstream oss;
  oss << x;
  return oss.str();
}

inline std::string JsonEscape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += c; break;
    }
  }
  return out;
}

inline void WriteSummaryStats(std::ostream& os, const SummaryStats& st, int indent = 0) {
  const std::string pad(indent, ' ');
  os << pad << "{\n";
  os << pad << "  \"count\": " << st.count << ",\n";
  os << pad << "  \"weight_sum\": " << JsonNum(st.weight_sum) << ",\n";
  os << pad << "  \"min\": " << JsonNum(st.min) << ",\n";
  os << pad << "  \"max\": " << JsonNum(st.max) << ",\n";
  os << pad << "  \"mean\": " << JsonNum(st.mean) << ",\n";
  os << pad << "  \"median\": " << JsonNum(st.median) << ",\n";
  os << pad << "  \"p90\": " << JsonNum(st.p90) << ",\n";
  os << pad << "  \"p99\": " << JsonNum(st.p99) << "\n";
  os << pad << "}";
}

inline bool WriteBuildDepthCsv(const std::string& path, const BuildMetrics& build, std::string* err = nullptr) {
  std::ofstream fout(path);
  if (!fout) {
    if (err) *err = "WriteBuildDepthCsv: failed to open " + path;
    return false;
  }
  fout << "depth,node_count,edge_count,edge_p50,edge_p90,edge_p99,vol_p50,vol_p90,vol_p99\n";
  for (const auto& row : build.bbox_by_depth) {
    fout << row.depth << ',' << row.node_count << ',' << row.edge_count << ','
         << row.edge_lengths.median << ',' << row.edge_lengths.p90 << ',' << row.edge_lengths.p99 << ','
         << row.volumes.median << ',' << row.volumes.p90 << ',' << row.volumes.p99 << '\n';
  }
  return true;
}

inline bool WriteGroupHistCsv(const std::string& path, const SamplePhaseMetrics& sample, std::string* err = nullptr) {
  std::ofstream fout(path);
  if (!fout) {
    if (err) *err = "WriteGroupHistCsv: failed to open " + path;
    return false;
  }
  fout << "group_size,count\n";
  for (const auto& [size, count] : sample.group_size_hist) {
    fout << size << ',' << count << '\n';
  }
  return true;
}

inline bool WriteLeftMetricsSampleCsv(const std::string& path,
                                      const CountPhaseMetrics& count,
                                      const SearchPhaseMetrics& search,
                                      int max_rows,
                                      std::string* err = nullptr) {
  std::ofstream fout(path);
  if (!fout) {
    if (err) *err = "WriteLeftMetricsSampleCsv: failed to open " + path;
    return false;
  }
  fout << "left_idx,w,V_count,P_count,C_count,H_count,S_prune,S_contain,D_count_max,V_search,P_search,B_mid,B_leaf,B_subtree,m,M_point,M_subtree,rho,sigma,avg_subtree_mass\n";
  const sjs::usize n = std::min<sjs::usize>(count.per_left.size(), static_cast<sjs::usize>(std::max(0, max_rows)));
  for (sjs::usize i = 0; i < n; ++i) {
    const auto& c = count.per_left[i];
    const auto& s = search.per_left[i];
    fout << i << ',' << c.w << ',' << c.visited << ',' << c.pruned << ',' << c.contained << ',' << c.partial << ','
         << c.saved_prune << ',' << c.saved_contain << ',' << c.max_depth << ','
         << s.visited << ',' << s.pruned << ',' << s.b_mid << ',' << s.b_leaf << ',' << s.b_subtree << ','
         << s.m_total << ',' << s.m_point << ',' << s.m_subtree << ','
         << s.rho << ',' << s.sigma << ',' << s.avg_subtree_mass << '\n';
  }
  return true;
}

inline bool WriteGroupMetricsSampleCsv(const std::string& path,
                                       const SamplePhaseMetrics& sample,
                                       int max_rows,
                                       std::string* err = nullptr) {
  std::ofstream fout(path);
  if (!fout) {
    if (err) *err = "WriteGroupMetricsSampleCsv: failed to open " + path;
    return false;
  }
  fout << "group_idx,group_size,K_point,K_subtree\n";
  const sjs::usize n = std::min<sjs::usize>(sample.groups.size(), static_cast<sjs::usize>(std::max(0, max_rows)));
  for (sjs::usize i = 0; i < n; ++i) {
    const auto& g = sample.groups[i];
    fout << i << ',' << g.size << ',' << g.k_point << ',' << g.k_subtree << '\n';
  }
  return true;
}

inline bool WriteMetricsJson(const std::string& path,
                             const RunMetadata& meta,
                             const BuildMetrics& build,
                             const CountPhaseMetrics& count,
                             const SearchPhaseMetrics& search,
                             const SamplePhaseMetrics& sample,
                             const CorrectnessMetrics& checks,
                             std::string* err = nullptr) {
  std::ofstream os(path);
  if (!os) {
    if (err) *err = "WriteMetricsJson: failed to open " + path;
    return false;
  }

  os << "{\n";
  os << "  \"run_id\": \"" << JsonEscape(meta.run_id) << "\",\n";
  os << "  \"network\": \"" << JsonEscape(meta.network) << "\",\n";
  os << "  \"preset\": \"" << JsonEscape(meta.preset) << "\",\n";
  os << "  \"family\": \"" << JsonEscape(meta.family) << "\",\n";
  os << "  \"dim\": " << meta.dim << ",\n";
  os << "  \"N\": " << meta.N << ",\n";
  os << "  \"alpha_target\": " << JsonNum(meta.alpha_target) << ",\n";
  os << "  \"alpha_expected_est\": " << JsonNum(meta.alpha_expected_est) << ",\n";
  os << "  \"pair_intersection_prob_est\": " << JsonNum(meta.pair_intersection_prob_est) << ",\n";
  os << "  \"coverage\": " << JsonNum(meta.coverage) << ",\n";
  os << "  \"data_seed\": " << meta.data_seed << ",\n";
  os << "  \"sample_seed\": " << meta.sample_seed << ",\n";
  os << "  \"data_rep\": " << meta.data_rep << ",\n";
  os << "  \"materialize_join\": " << (meta.materialize_join ? "true" : "false") << ",\n";

  os << "  \"timing_ms\": {\n";
  os << "    \"T_lift\": " << JsonNum(build.t_lift_ms) << ",\n";
  os << "    \"T_tree\": " << JsonNum(build.t_tree_ms) << ",\n";
  os << "    \"T_build\": " << JsonNum(build.t_build_ms) << ",\n";
  os << "    \"T_count_query\": " << JsonNum(count.t_count_query_ms) << ",\n";
  os << "    \"T_left_dist\": " << JsonNum(count.t_left_dist_ms) << ",\n";
  os << "    \"T_count\": " << JsonNum(count.t_count_ms) << ",\n";
  os << "    \"T_search_scan\": " << JsonNum(search.t_search_scan_ms) << ",\n";
  os << "    \"T_plan_left\": " << JsonNum(sample.t_plan_left_ms) << ",\n";
  os << "    \"T_group\": " << JsonNum(sample.t_group_ms) << ",\n";
  os << "    \"T_decompose\": " << JsonNum(sample.t_decompose_ms) << ",\n";
  os << "    \"T_local_dist\": " << JsonNum(sample.t_local_dist_ms) << ",\n";
  os << "    \"T_emit\": " << JsonNum(sample.t_emit_ms) << ",\n";
  os << "    \"T_sample\": " << JsonNum(sample.t_sample_ms) << "\n";
  os << "  },\n";

  os << "  \"build\": {\n";
  os << "    \"tree_height\": " << build.tree_height << ",\n";
  os << "    \"max_depth\": " << build.max_depth << ",\n";
  os << "    \"avg_node_depth\": " << JsonNum(build.avg_node_depth) << ",\n";
  os << "    \"node_count\": " << build.node_count << "\n";
  os << "  },\n";

  os << "  \"count\": {\n";
  os << "    \"exact_join_size\": " << count.exact_join_size << ",\n";
  os << "    \"alpha_actual\": " << JsonNum(count.alpha_actual) << ",\n";
  os << "    \"w\": "; WriteSummaryStats(os, count.w_summary, 4); os << ",\n";
  os << "    \"V_count\": "; WriteSummaryStats(os, count.visited_summary, 4); os << ",\n";
  os << "    \"P_count\": "; WriteSummaryStats(os, count.pruned_summary, 4); os << ",\n";
  os << "    \"C_count\": "; WriteSummaryStats(os, count.contained_summary, 4); os << ",\n";
  os << "    \"H_count\": "; WriteSummaryStats(os, count.partial_summary, 4); os << ",\n";
  os << "    \"S_prune\": "; WriteSummaryStats(os, count.saved_prune_summary, 4); os << ",\n";
  os << "    \"S_contain\": "; WriteSummaryStats(os, count.saved_contain_summary, 4); os << ",\n";
  os << "    \"D_count_max\": "; WriteSummaryStats(os, count.max_depth_summary, 4); os << "\n";
  os << "  },\n";

  os << "  \"search_equal\": {\n";
  os << "    \"V_search\": "; WriteSummaryStats(os, search.visited_equal, 4); os << ",\n";
  os << "    \"P_search\": "; WriteSummaryStats(os, search.pruned_equal, 4); os << ",\n";
  os << "    \"B_mid\": "; WriteSummaryStats(os, search.b_mid_equal, 4); os << ",\n";
  os << "    \"B_leaf\": "; WriteSummaryStats(os, search.b_leaf_equal, 4); os << ",\n";
  os << "    \"B_subtree\": "; WriteSummaryStats(os, search.b_subtree_equal, 4); os << ",\n";
  os << "    \"m\": "; WriteSummaryStats(os, search.m_total_equal, 4); os << ",\n";
  os << "    \"M_point\": "; WriteSummaryStats(os, search.m_point_equal, 4); os << ",\n";
  os << "    \"M_subtree\": "; WriteSummaryStats(os, search.m_subtree_equal, 4); os << ",\n";
  os << "    \"rho\": "; WriteSummaryStats(os, search.rho_equal, 4); os << ",\n";
  os << "    \"sigma\": "; WriteSummaryStats(os, search.sigma_equal, 4); os << ",\n";
  os << "    \"avg_subtree_mass\": "; WriteSummaryStats(os, search.avg_subtree_mass_equal, 4); os << "\n";
  os << "  },\n";

  os << "  \"search_weighted\": {\n";
  os << "    \"V_search\": "; WriteSummaryStats(os, search.visited_weighted, 4); os << ",\n";
  os << "    \"P_search\": "; WriteSummaryStats(os, search.pruned_weighted, 4); os << ",\n";
  os << "    \"B_mid\": "; WriteSummaryStats(os, search.b_mid_weighted, 4); os << ",\n";
  os << "    \"B_leaf\": "; WriteSummaryStats(os, search.b_leaf_weighted, 4); os << ",\n";
  os << "    \"B_subtree\": "; WriteSummaryStats(os, search.b_subtree_weighted, 4); os << ",\n";
  os << "    \"m\": "; WriteSummaryStats(os, search.m_total_weighted, 4); os << ",\n";
  os << "    \"M_point\": "; WriteSummaryStats(os, search.m_point_weighted, 4); os << ",\n";
  os << "    \"M_subtree\": "; WriteSummaryStats(os, search.m_subtree_weighted, 4); os << ",\n";
  os << "    \"rho\": "; WriteSummaryStats(os, search.rho_weighted, 4); os << ",\n";
  os << "    \"sigma\": "; WriteSummaryStats(os, search.sigma_weighted, 4); os << ",\n";
  os << "    \"avg_subtree_mass\": "; WriteSummaryStats(os, search.avg_subtree_mass_weighted, 4); os << "\n";
  os << "  },\n";

  os << "  \"sample\": {\n";
  os << "    \"t\": " << sample.t << ",\n";
  os << "    \"chunk_size\": " << sample.chunk_size << ",\n";
  os << "    \"num_chunks\": " << sample.num_chunks << ",\n";
  os << "    \"U_t\": " << sample.U_t << ",\n";
  os << "    \"G_t\": " << sample.G_t << ",\n";
  os << "    \"avg_repeat_global\": " << JsonNum(sample.avg_repeat_global) << ",\n";
  os << "    \"avg_group_size_exec\": " << JsonNum(sample.avg_group_size_exec) << ",\n";
  os << "    \"samples_per_sec\": " << JsonNum(sample.samples_per_sec) << ",\n";
  os << "    \"ns_per_sample\": " << JsonNum(sample.ns_per_sample) << ",\n";
  os << "    \"checksum\": " << sample.checksum << ",\n";
  os << "    \"group_size_equal\": "; WriteSummaryStats(os, sample.group_size_equal, 4); os << ",\n";
  os << "    \"group_size_sample_weighted\": "; WriteSummaryStats(os, sample.group_size_sample_weighted, 4); os << ",\n";
  os << "    \"K_point_equal\": "; WriteSummaryStats(os, sample.k_point_equal, 4); os << ",\n";
  os << "    \"K_point_sample_weighted\": "; WriteSummaryStats(os, sample.k_point_sample_weighted, 4); os << ",\n";
  os << "    \"K_subtree_equal\": "; WriteSummaryStats(os, sample.k_subtree_equal, 4); os << ",\n";
  os << "    \"K_subtree_sample_weighted\": "; WriteSummaryStats(os, sample.k_subtree_sample_weighted, 4); os << "\n";
  os << "  },\n";

  os << "  \"checks\": {\n";
  os << "    \"count_sum_matches\": " << (checks.count_sum_matches ? "true" : "false") << ",\n";
  os << "    \"decompose_mass_matches\": " << (checks.decompose_mass_matches ? "true" : "false") << ",\n";
  os << "    \"sample_valid\": " << (checks.sample_valid ? "true" : "false") << ",\n";
  os << "    \"empty_output_consistent\": " << (checks.empty_output_consistent ? "true" : "false") << ",\n";
  os << "    \"materialized_join_matches\": " << (checks.materialized_join_matches ? "true" : "false") << ",\n";
  os << "    \"materialized_join_size\": " << checks.materialized_join_size << ",\n";
  os << "    \"pair_tv_to_uniform\": " << JsonNum(checks.pair_tv_to_uniform) << ",\n";
  os << "    \"pair_linf_to_uniform\": " << JsonNum(checks.pair_linf_to_uniform) << ",\n";
  os << "    \"left_tv_to_theory\": " << JsonNum(checks.left_tv_to_theory) << ",\n";
  os << "    \"distinct_pairs_sampled\": " << checks.distinct_pairs_sampled << "\n";
  os << "  }\n";

  os << "}\n";
  return true;
}

}  // namespace kexp
