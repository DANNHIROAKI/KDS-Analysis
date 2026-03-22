#include "src/dataset_io.hpp"
#include "src/experiment_kernel.hpp"
#include "src/output_utils.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace fs = std::filesystem;

namespace {

std::unordered_map<std::string, std::string> ParseArgs(int argc, char** argv) {
  std::unordered_map<std::string, std::string> out;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key.rfind("--", 0) != 0) {
      throw std::runtime_error("unexpected positional argument: " + key);
    }
    if (i + 1 >= argc) {
      throw std::runtime_error("missing value for argument: " + key);
    }
    out[key.substr(2)] = argv[++i];
  }
  return out;
}

template <class T>
T GetOr(const std::unordered_map<std::string, std::string>& args, const std::string& key, T default_value);

template <>
std::string GetOr(const std::unordered_map<std::string, std::string>& args, const std::string& key, std::string default_value) {
  auto it = args.find(key);
  return it == args.end() ? default_value : it->second;
}

template <>
int GetOr(const std::unordered_map<std::string, std::string>& args, const std::string& key, int default_value) {
  auto it = args.find(key);
  return it == args.end() ? default_value : std::stoi(it->second);
}

template <>
unsigned long long GetOr(const std::unordered_map<std::string, std::string>& args,
                         const std::string& key,
                         unsigned long long default_value) {
  auto it = args.find(key);
  return it == args.end() ? default_value : static_cast<unsigned long long>(std::stoull(it->second));
}

template <>
double GetOr(const std::unordered_map<std::string, std::string>& args, const std::string& key, double default_value) {
  auto it = args.find(key);
  return it == args.end() ? default_value : std::stod(it->second);
}

template <int Dim>
int Run(const kexp::RunMetadata& meta, const std::string& dataset_path, const fs::path& out_dir) {
  std::string err;
  sjs::Dataset<Dim, float> ds;
  if (!kexp::LoadBinaryDataset<Dim>(dataset_path, &ds, &err)) {
    std::cerr << err << std::endl;
    return 1;
  }

  kexp::ExperimentKernel<Dim> kernel(ds);
  kernel.SetCorrectnessMaterialize(meta.materialize_join);
  if (!kernel.Build(&err)) {
    std::cerr << err << std::endl;
    return 1;
  }
  if (!kernel.RunCount(&err)) {
    std::cerr << err << std::endl;
    return 1;
  }
  if (!kernel.RunFullSearchScan(&err)) {
    std::cerr << err << std::endl;
    return 1;
  }
  if (!kernel.RunSample(meta, &err)) {
    std::cerr << err << std::endl;
    return 1;
  }
  kernel.RunCorrectnessMaterialization();

  fs::create_directories(out_dir);
  if (!kexp::WriteBuildDepthCsv((out_dir / "bbox_by_depth.csv").string(), kernel.build(), &err)) {
    std::cerr << err << std::endl;
    return 1;
  }
  if (!kexp::WriteGroupHistCsv((out_dir / "group_size_hist.csv").string(), kernel.sample(), &err)) {
    std::cerr << err << std::endl;
    return 1;
  }
  if (!kexp::WriteLeftMetricsSampleCsv((out_dir / "left_metrics_sample.csv").string(),
                                      kernel.count(), kernel.search(),
                                      meta.left_metrics_sample_rows, &err)) {
    std::cerr << err << std::endl;
    return 1;
  }
  if (!kexp::WriteGroupMetricsSampleCsv((out_dir / "group_metrics_sample.csv").string(),
                                       kernel.sample(), meta.group_metrics_sample_rows, &err)) {
    std::cerr << err << std::endl;
    return 1;
  }
  if (!kexp::WriteMetricsJson((out_dir / "metrics.json").string(),
                             meta,
                             kernel.build(),
                             kernel.count(),
                             kernel.search(),
                             kernel.sample(),
                             kernel.correctness(),
                             &err)) {
    std::cerr << err << std::endl;
    return 1;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = ParseArgs(argc, argv);
    const std::string dataset_path = GetOr<std::string>(args, "dataset", "");
    const std::string out_dir = GetOr<std::string>(args, "out-dir", "");
    if (dataset_path.empty() || out_dir.empty()) {
      std::cerr << "usage: benchmark --dataset PATH --out-dir DIR --dim {2,3,4,5} [metadata ...]" << std::endl;
      return 2;
    }

    kexp::RunMetadata meta;
    meta.run_id = GetOr<std::string>(args, "run-id", "run");
    meta.network = GetOr<std::string>(args, "network", "unknown");
    meta.preset = GetOr<std::string>(args, "preset", "TH");
    meta.family = GetOr<std::string>(args, "family", "F0");
    meta.dim = GetOr<int>(args, "dim", 0);
    meta.N = static_cast<sjs::u64>(GetOr<unsigned long long>(args, "N", 0ULL));
    meta.alpha_target = GetOr<double>(args, "alpha-target", 0.0);
    meta.alpha_expected_est = GetOr<double>(args, "alpha-expected", 0.0);
    meta.pair_intersection_prob_est = GetOr<double>(args, "pair-prob", 0.0);
    meta.coverage = GetOr<double>(args, "coverage", 0.0);
    meta.data_seed = static_cast<sjs::u64>(GetOr<unsigned long long>(args, "data-seed", 0ULL));
    meta.sample_seed = static_cast<sjs::u64>(GetOr<unsigned long long>(args, "sample-seed", 0ULL));
    meta.sample_t = static_cast<sjs::u64>(GetOr<unsigned long long>(args, "t", 0ULL));
    meta.data_rep = GetOr<int>(args, "data-rep", 0);
    meta.materialize_join = GetOr<int>(args, "materialize-join", 0) != 0;
    meta.left_metrics_sample_rows = GetOr<int>(args, "left-sample-rows", 1024);
    meta.group_metrics_sample_rows = GetOr<int>(args, "group-sample-rows", 2048);

    const fs::path out_path(out_dir);
    switch (meta.dim) {
      case 2: return Run<2>(meta, dataset_path, out_path);
      case 3: return Run<3>(meta, dataset_path, out_path);
      case 4: return Run<4>(meta, dataset_path, out_path);
      case 5: return Run<5>(meta, dataset_path, out_path);
      default:
        std::cerr << "unsupported dim: " << meta.dim << std::endl;
        return 2;
    }
  } catch (const std::exception& ex) {
    std::cerr << "benchmark failed: " << ex.what() << std::endl;
    return 1;
  }
}
