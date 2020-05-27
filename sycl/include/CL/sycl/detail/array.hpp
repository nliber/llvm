//==-------- array.hpp --- SYCL common iteration object --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/exception.hpp>
#include <functional>
#include <stdexcept>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
template <int dimensions> class id;
template <int dimensions> class range;
namespace detail {

template <int dimensions = 1> class array {
  static_assert(dimensions >= 1, "Array cannot be 0-dimensional.");

public:
  array() : common_array{} {}

  /* The following constructor is only available in the array struct
   * specialization where: dimensions==1 */
  template <int N = dimensions>
  array(typename std::enable_if<(N == 1), size_t>::type dim0 = 0)
      : common_array{dim0} {}

  /* The following constructor is only available in the array struct
   * specialization where: dimensions>=2 */
  template <
      typename Dim1, typename Dim2, typename... Dims,
      typename = enable_if_t<dimensions == 2 + sizeof...(Dims) &&
                             std::is_convertible_v<Dim1, size_t> &&
                             std::is_convertible_v<Dim2, size_t> &&
                             (std::is_convertible_v<Dims, size_t> && ...)>>
  array(Dim1 &&dim1, Dim2 &&dim2, Dims &&... dims)
      : common_array{static_cast<size_t>(std::forward<Dim1>(dim1)),
                     static_cast<size_t>(std::forward<Dim2>(dim2)),
                     static_cast<size_t>(std::forward<Dims>(dims))...} {}

  // Conversion operators to derived classes
  operator cl::sycl::id<dimensions>() const {
    cl::sycl::id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result[i] = common_array[i];
    }
    return result;
  }

  operator cl::sycl::range<dimensions>() const {
    cl::sycl::range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result[i] = common_array[i];
    }
    return result;
  }

  size_t get(int dimension) const {
    check_dimension(dimension);
    return common_array[dimension];
  }

  size_t &operator[](int dimension) {
    check_dimension(dimension);
    return common_array[dimension];
  }

  size_t operator[](int dimension) const {
    check_dimension(dimension);
    return common_array[dimension];
  }

  array(const array<dimensions> &rhs) = default;
  array(array<dimensions> &&rhs) = default;
  array<dimensions> &operator=(const array<dimensions> &rhs) = default;
  array<dimensions> &operator=(array<dimensions> &&rhs) = default;

  // Returns true iff all elements in 'this' are equal to
  // the corresponding elements in 'rhs'.
  bool operator==(const array<dimensions> &rhs) const {
    for (int i = 0; i < dimensions; ++i) {
      if (this->common_array[i] != rhs.common_array[i]) {
        return false;
      }
    }
    return true;
  }

  // Returns true iff there is at least one element in 'this'
  // which is not equal to the corresponding element in 'rhs'.
  bool operator!=(const array<dimensions> &rhs) const {
    for (int i = 0; i < dimensions; ++i) {
      if (this->common_array[i] != rhs.common_array[i]) {
        return true;
      }
    }
    return false;
  }

protected:
  size_t common_array[dimensions];
  ALWAYS_INLINE void check_dimension(int dimension) const {
#ifndef __SYCL_DEVICE_ONLY__
    if (dimension >= dimensions || dimension < 0) {
      throw cl::sycl::invalid_parameter_error("Index out of range",
                                              PI_INVALID_VALUE);
    }
#endif
    (void)dimension;
  }
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
