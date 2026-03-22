#pragma once

#include "core/rng.h"
#include "core/types.h"

#include <array>
#include <chrono>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sjs {

struct PairId {
  Id r = 0;
  Id s = 0;
};

struct CountResult {
  bool exact = true;
  u64 value = 0;
};

inline CountResult MakeExactCount(u64 value) {
  CountResult out;
  out.exact = true;
  out.value = value;
  return out;
}

struct SampleSet {
  std::vector<PairId> pairs;
  std::vector<double> weights;
  bool with_replacement = true;
  bool weighted = false;

  void Clear() {
    pairs.clear();
    weights.clear();
    with_replacement = true;
    weighted = false;
  }
};

enum class Method {
  RSOverSRJ,
};

enum class Variant {
  Sampling,
};

class PhaseRecorder {
 public:
  class ScopedPhase {
   public:
    ScopedPhase(PhaseRecorder* recorder, std::string name)
        : recorder_(recorder), name_(std::move(name)), t0_(Clock::now()) {}
    ScopedPhase(const ScopedPhase&) = delete;
    ScopedPhase& operator=(const ScopedPhase&) = delete;
    ScopedPhase(ScopedPhase&& other) noexcept
        : recorder_(other.recorder_), name_(std::move(other.name_)), t0_(other.t0_), active_(other.active_) {
      other.active_ = false;
    }
    ~ScopedPhase() {
      if (active_ && recorder_ != nullptr) {
        const auto t1 = Clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0_).count();
        recorder_->Add(name_, ms);
      }
    }

   private:
    using Clock = std::chrono::steady_clock;
    PhaseRecorder* recorder_ = nullptr;
    std::string name_;
    Clock::time_point t0_{};
    bool active_ = true;
  };

  ScopedPhase Scoped(std::string name) { return ScopedPhase(this, std::move(name)); }

  void Add(const std::string& name, double ms) { totals_ms_[name] += ms; }

  double GetMs(const std::string& name) const {
    auto it = totals_ms_.find(name);
    return it == totals_ms_.end() ? 0.0 : it->second;
  }

 private:
  std::unordered_map<std::string, double> totals_ms_;
};

struct Config {
  struct RunConfig {
    u64 t = 0;
  } run;
};

template <int Dim, class T>
struct Vec {
  std::array<T, Dim> v{};
};

template <int Dim, class T>
struct Box {
  Vec<Dim, T> lo{};
  Vec<Dim, T> hi{};

  bool Intersects(const Box& other) const noexcept {
    for (int axis = 0; axis < Dim; ++axis) {
      const usize i = static_cast<usize>(axis);
      if (!(lo.v[i] < other.hi.v[i] && other.lo.v[i] < hi.v[i])) {
        return false;
      }
    }
    return true;
  }
};

template <int Dim, class T>
struct Relation {
  std::vector<Box<Dim, T>> boxes;
  std::vector<Id> ids;

  usize Size() const noexcept { return boxes.size(); }

  Id GetId(usize i) const noexcept {
    return ids.empty() ? static_cast<Id>(i) : ids[i];
  }
};

template <int Dim, class T>
struct Dataset {
  Relation<Dim, T> R;
  Relation<Dim, T> S;
};

class IJoinEnumerator;

template <int Dim, class T>
class IBaseline {
 public:
  using DatasetT = Dataset<Dim, T>;
  virtual ~IBaseline() = default;

  virtual Method method() const noexcept = 0;
  virtual Variant variant() const noexcept = 0;
  virtual std::string_view Name() const noexcept = 0;

  virtual void Reset() = 0;
  virtual bool Build(const DatasetT& ds, const Config& cfg, PhaseRecorder* phases, std::string* err) = 0;
  virtual bool Count(const Config& cfg, Rng* rng, CountResult* out, PhaseRecorder* phases, std::string* err) = 0;
  virtual bool Sample(const Config& cfg, Rng* rng, SampleSet* out, PhaseRecorder* phases, std::string* err) = 0;
  virtual std::unique_ptr<IJoinEnumerator> Enumerate(const Config& cfg, PhaseRecorder* phases, std::string* err) = 0;
};

}  // namespace sjs
