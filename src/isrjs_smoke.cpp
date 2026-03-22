#include "vendor/KDTree/isrjs_kds/isrjs_kds.hpp"

#include <iostream>

int main() {
  using Baseline = sjs::baselines::rs_over_srj::RSOverSRJKDTreeSamplingBaseline<2, float>;
  sjs::Dataset<2, float> ds;
  ds.R.boxes.resize(3);
  ds.R.ids = {0, 1, 2};
  ds.S.boxes.resize(3);
  ds.S.ids = {0, 1, 2};

  ds.R.boxes[0].lo.v = {0.0f, 0.0f}; ds.R.boxes[0].hi.v = {0.6f, 0.6f};
  ds.R.boxes[1].lo.v = {0.5f, 0.5f}; ds.R.boxes[1].hi.v = {0.9f, 0.9f};
  ds.R.boxes[2].lo.v = {0.1f, 0.7f}; ds.R.boxes[2].hi.v = {0.3f, 0.95f};

  ds.S.boxes[0].lo.v = {0.2f, 0.2f}; ds.S.boxes[0].hi.v = {0.7f, 0.7f};
  ds.S.boxes[1].lo.v = {0.8f, 0.8f}; ds.S.boxes[1].hi.v = {0.95f, 0.95f};
  ds.S.boxes[2].lo.v = {0.0f, 0.75f}; ds.S.boxes[2].hi.v = {0.2f, 0.99f};

  Baseline baseline;
  sjs::Config cfg;
  cfg.run.t = 256;
  sjs::PhaseRecorder phases;
  std::string err;
  if (!baseline.Build(ds, cfg, &phases, &err)) {
    std::cerr << err << std::endl;
    return 1;
  }
  sjs::CountResult cnt;
  if (!baseline.Count(cfg, nullptr, &cnt, &phases, &err)) {
    std::cerr << err << std::endl;
    return 1;
  }
  sjs::Rng rng(123456789ULL);
  sjs::SampleSet out;
  if (!baseline.Sample(cfg, &rng, &out, &phases, &err)) {
    std::cerr << err << std::endl;
    return 1;
  }
  std::uint64_t checksum = 1469598103934665603ULL;
  for (const auto& p : out.pairs) {
    checksum ^= (static_cast<std::uint64_t>(p.r) << 32) ^ p.s;
    checksum *= 1099511628211ULL;
  }
  std::cout << "count=" << cnt.value << " samples=" << out.pairs.size() << " checksum=" << checksum << std::endl;
  return 0;
}
