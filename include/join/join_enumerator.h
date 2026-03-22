#pragma once

#include "baselines/baseline_api.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

namespace sjs {
namespace join {

struct JoinStats {
  u64 pairs_emitted = 0;
};

enum class SideTieBreak {
  RBeforeS,
  SBeforeR,
};

struct PlaneSweepOptions {
  int axis = 0;
  SideTieBreak side_order = SideTieBreak::RBeforeS;
};

template <int Dim, class T>
class PlaneSweepJoinStream {
 public:
  PlaneSweepJoinStream(const Relation<Dim, T>& R,
                      const Relation<Dim, T>& S,
                      const PlaneSweepOptions& opt)
      : r_(R), s_(S), opt_(opt) {
    BuildPairs();
  }

  void Reset() { pos_ = 0; }

  bool Next(PairId* out) {
    if (out == nullptr || pos_ >= pairs_.size()) return false;
    *out = pairs_[pos_++];
    return true;
  }

  const JoinStats& Stats() const noexcept { return stats_; }

 private:
  struct ActiveByHi {
    T hi = T{};
    u32 sid = 0;
    bool operator>(const ActiveByHi& other) const noexcept {
      if (hi > other.hi) return true;
      if (hi < other.hi) return false;
      return sid > other.sid;
    }
  };

  void BuildPairs() {
    const int axis = std::clamp(opt_.axis, 0, Dim - 1);
    std::vector<u32> r_order(r_.Size());
    std::vector<u32> s_order(s_.Size());
    for (usize i = 0; i < r_order.size(); ++i) r_order[i] = static_cast<u32>(i);
    for (usize i = 0; i < s_order.size(); ++i) s_order[i] = static_cast<u32>(i);

    auto key_lo_r = [&](u32 idx) { return r_.boxes[idx].lo.v[static_cast<usize>(axis)]; };
    auto key_lo_s = [&](u32 idx) { return s_.boxes[idx].lo.v[static_cast<usize>(axis)]; };

    auto by_lo_r = [&](u32 a, u32 b) {
      if (key_lo_r(a) < key_lo_r(b)) return true;
      if (key_lo_r(b) < key_lo_r(a)) return false;
      return a < b;
    };
    auto by_lo_s = [&](u32 a, u32 b) {
      if (key_lo_s(a) < key_lo_s(b)) return true;
      if (key_lo_s(b) < key_lo_s(a)) return false;
      return a < b;
    };

    std::sort(r_order.begin(), r_order.end(), by_lo_r);
    std::sort(s_order.begin(), s_order.end(), by_lo_s);

    std::priority_queue<ActiveByHi, std::vector<ActiveByHi>, std::greater<ActiveByHi>> pq;
    std::vector<u32> active;
    std::vector<u8> alive(s_.Size(), 0);
    active.reserve(s_.Size());

    usize sp = 0;
    for (u32 ridx : r_order) {
      const auto& rbox = r_.boxes[ridx];
      const T rlo = rbox.lo.v[static_cast<usize>(axis)];
      const T rhi = rbox.hi.v[static_cast<usize>(axis)];

      while (sp < s_order.size() && s_.boxes[s_order[sp]].lo.v[static_cast<usize>(axis)] < rhi) {
        const u32 sidx = s_order[sp++];
        const auto& sbox = s_.boxes[sidx];
        if (sbox.hi.v[static_cast<usize>(axis)] > rlo) {
          alive[sidx] = 1;
          active.push_back(sidx);
          pq.push(ActiveByHi{sbox.hi.v[static_cast<usize>(axis)], sidx});
        }
      }

      while (!pq.empty() && !(rlo < pq.top().hi)) {
        alive[pq.top().sid] = 0;
        pq.pop();
      }

      for (u32 sidx : active) {
        if (!alive[sidx]) continue;
        if (rbox.Intersects(s_.boxes[sidx])) {
          pairs_.push_back(PairId{r_.GetId(static_cast<usize>(ridx)), s_.GetId(static_cast<usize>(sidx))});
        }
      }

      if (active.size() > 1024 && pq.size() * 3 < active.size()) {
        std::vector<u32> compact;
        compact.reserve(pq.size());
        for (u32 sidx : active) {
          if (alive[sidx]) compact.push_back(sidx);
        }
        active.swap(compact);
      }
    }

    stats_.pairs_emitted = static_cast<u64>(pairs_.size());
  }

  const Relation<Dim, T>& r_;
  const Relation<Dim, T>& s_;
  PlaneSweepOptions opt_{};
  std::vector<PairId> pairs_;
  usize pos_ = 0;
  JoinStats stats_{};
};

}  // namespace join

class IJoinEnumerator {
 public:
  virtual ~IJoinEnumerator() = default;
  virtual void Reset() = 0;
  virtual bool Next(PairId* out) = 0;
  virtual const join::JoinStats& Stats() const noexcept = 0;
};

}  // namespace sjs
