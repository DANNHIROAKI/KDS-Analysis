#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace sjs {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using usize = std::size_t;
using Scalar = float;
using Id = std::uint32_t;

#ifndef SJS_DASSERT
#define SJS_DASSERT(x) assert(x)
#endif

template <class T>
class Span {
 public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using pointer = T*;
  using reference = T&;

  Span() noexcept = default;
  Span(pointer data, usize size) noexcept : data_(data), size_(size) {}

  template <class U,
            class = std::enable_if_t<std::is_same_v<std::remove_const_t<U>, value_type> &&
                                     std::is_convertible_v<U (*)[], T (*)[]>>>
  Span(std::vector<U>& v) noexcept : data_(v.data()), size_(v.size()) {}

  template <class U,
            class = std::enable_if_t<std::is_same_v<std::remove_const_t<U>, value_type> &&
                                     std::is_convertible_v<const U (*)[], T (*)[]>>>
  Span(const std::vector<U>& v) noexcept : data_(v.data()), size_(v.size()) {}

  pointer data() const noexcept { return data_; }
  usize size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }

  reference operator[](usize i) const noexcept { return data_[i]; }
  pointer begin() const noexcept { return data_; }
  pointer end() const noexcept { return data_ + size_; }

 private:
  pointer data_ = nullptr;
  usize size_ = 0;
};

}  // namespace sjs
