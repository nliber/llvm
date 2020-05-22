//==---------- item_base.hpp --- SYCL iteration ItemBase ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/id.hpp>
#include <CL/sycl/range.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
template <int dimensions> class id;
template <int dimensions> class range;

namespace detail {
template <int Dims, bool WithOffset> struct ItemBase;

template <int Dims> struct ItemBase<Dims, true> {

  bool operator==(const ItemBase &Rhs) const {
    return (Rhs.MIndex == MIndex) && (Rhs.MExtent == MExtent) &&
           (Rhs.MOffset == MOffset);
  }

  bool operator!=(const ItemBase &Rhs) const { return !((*this) == Rhs); }

  size_t get_linear_id() const {

    size_t linearId = 0;

    for (int dim = 0; dim != Dims; ++dim) {
      linearId *= MExtent[dim];
      linearId += MIndex[dim] - MOffset[dim];
    }

    return linearId;
  }

  range<Dims> MExtent;
  id<Dims> MIndex;
  id<Dims> MOffset;
};

template <int Dims> struct ItemBase<Dims, false> {

  bool operator==(const ItemBase &Rhs) const {
    return (Rhs.MIndex == MIndex) && (Rhs.MExtent == MExtent);
  }

  bool operator!=(const ItemBase &Rhs) const { return !((*this) == Rhs); }

  operator ItemBase<Dims, true>() const {
    return ItemBase<Dims, true>(MExtent, MIndex, id<Dims>{});
  }

  size_t get_linear_id() const {

    size_t linearId = 0;

    for (int dim = 0; dim != Dims; ++dim) {
      linearId *= MExtent[dim];
      linearId += MIndex[dim];
    }

    return linearId;
  }

  range<Dims> MExtent;
  id<Dims> MIndex;
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
