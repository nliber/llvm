//==----------- range.hpp --- SYCL iteration range -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/array.hpp>

#include <stdexcept>
#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
template <int dimensions> class id;
template <int dimensions = 1> class range : public detail::array<dimensions> {
  static_assert(dimensions >= 1, "range must be at least 1-dimensional.");
  using base = detail::array<dimensions>;
  template <typename N, typename T>
  using IntegralType = detail::enable_if_t<std::is_integral<N>::value, T>;

public:
  /* The following constructor is only available in the range class
  specialization where: dimensions==1 */
  template <int N = dimensions>
  range(typename std::enable_if<(N == 1), size_t>::type dim0) : base(dim0) {}

  /* The following constructor is only available in the range class
  specialization where: dimensions>=2 */
  template <
      typename Dim1, typename Dim2, typename... Dims,
      typename = std::enable_if_t<dimensions == 2 + sizeof...(Dims) &&
                                  std::is_convertible_v<Dim1, size_t> &&
                                  std::is_convertible_v<Dim2, size_t> &&
                                  (std::is_convertible_v<Dims, size_t> && ...)>>
  range(Dim1 &&dim1, Dim2 &&dim2, Dims &&... dims)
      : base(static_cast<size_t>(std::forward<Dim1>(dim1)),
             static_cast<size_t>(std::forward<Dim2>(dim2)),
             static_cast<size_t>(std::forward<Dims>(dims))...) {}

  explicit operator id<dimensions>() const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result[i] = this->get(i);
    }
    return result;
  }

  size_t size() const {
    size_t size = 1;
    for (int i = 0; i < dimensions; ++i) {
      size *= this->get(i);
    }
    return size;
  }

  range(const range<dimensions> &rhs) = default;
  range(range<dimensions> &&rhs) = default;
  range<dimensions> &operator=(const range<dimensions> &rhs) = default;
  range<dimensions> &operator=(range<dimensions> &&rhs) = default;
  range() = delete;

// OP is: +, -, *, /, %, <<, >>, &, |, ^, &&, ||, <, >, <=, >=
#define __SYCL_GEN_OPT(op)                                                     \
  range<dimensions> operator op(const range<dimensions> &rhs) const {          \
    range<dimensions> result(*this);                                           \
    for (int i = 0; i < dimensions; ++i) {                                     \
      result.common_array[i] = this->common_array[i] op rhs.common_array[i];   \
    }                                                                          \
    return result;                                                             \
  }                                                                            \
  template <typename T>                                                        \
  IntegralType<T, range<dimensions>> operator op(const T &rhs) const {         \
    range<dimensions> result(*this);                                           \
    for (int i = 0; i < dimensions; ++i) {                                     \
      result.common_array[i] = this->common_array[i] op rhs;                   \
    }                                                                          \
    return result;                                                             \
  }                                                                            \
  template <typename T>                                                        \
  friend IntegralType<T, range<dimensions>> operator op(                       \
      const T &lhs, const range<dimensions> &rhs) {                            \
    range<dimensions> result(rhs);                                             \
    for (int i = 0; i < dimensions; ++i) {                                     \
      result.common_array[i] = lhs op rhs.common_array[i];                     \
    }                                                                          \
    return result;                                                             \
  }

  __SYCL_GEN_OPT(+)
  __SYCL_GEN_OPT(-)
  __SYCL_GEN_OPT(*)
  __SYCL_GEN_OPT(/)
  __SYCL_GEN_OPT(%)
  __SYCL_GEN_OPT(<<)
  __SYCL_GEN_OPT(>>)
  __SYCL_GEN_OPT(&)
  __SYCL_GEN_OPT(|)
  __SYCL_GEN_OPT(^)
  __SYCL_GEN_OPT(&&)
  __SYCL_GEN_OPT(||)
  __SYCL_GEN_OPT(<)
  __SYCL_GEN_OPT(>)
  __SYCL_GEN_OPT(<=)
  __SYCL_GEN_OPT(>=)

#undef __SYCL_GEN_OPT

// OP is: +=, -=, *=, /=, %=, <<=, >>=, &=, |=, ^=
#define __SYCL_GEN_OPT(op)                                                     \
  range<dimensions> &operator op(const range<dimensions> &rhs) {               \
    for (int i = 0; i < dimensions; ++i) {                                     \
      this->common_array[i] op rhs[i];                                         \
    }                                                                          \
    return *this;                                                              \
  }                                                                            \
  range<dimensions> &operator op(const size_t &rhs) {                          \
    for (int i = 0; i < dimensions; ++i) {                                     \
      this->common_array[i] op rhs;                                            \
    }                                                                          \
    return *this;                                                              \
  }

  __SYCL_GEN_OPT(+=)
  __SYCL_GEN_OPT(-=)
  __SYCL_GEN_OPT(*=)
  __SYCL_GEN_OPT(/=)
  __SYCL_GEN_OPT(%=)
  __SYCL_GEN_OPT(<<=)
  __SYCL_GEN_OPT(>>=)
  __SYCL_GEN_OPT(&=)
  __SYCL_GEN_OPT(|=)
  __SYCL_GEN_OPT(^=)

#undef __SYCL_GEN_OPT
};

#ifdef __cpp_deduction_guides
range(size_t)->range<1>;

template <
    typename Dim1, typename Dim2, typename... Dims,
    typename = std::enable_if_t<std::is_convertible_v<Dim1, size_t> &&
                                std::is_convertible_v<Dim2, size_t> &&
                                (std::is_convertible_v<Dims, size_t> && ...)>>
range(Dim1 &&, Dim2 &&, Dims &&...) -> range<2 + sizeof...(Dims)>;
#endif

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
