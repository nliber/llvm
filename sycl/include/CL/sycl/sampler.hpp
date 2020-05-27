//==----------------- sampler.hpp - SYCL standard header file --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_types.hpp>
#include <CL/sycl/access/access.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/detail/sampler_impl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
enum class addressing_mode : unsigned int {
  mirrored_repeat = CL_ADDRESS_MIRRORED_REPEAT,
  repeat = CL_ADDRESS_REPEAT,
  clamp_to_edge = CL_ADDRESS_CLAMP_TO_EDGE,
  clamp = CL_ADDRESS_CLAMP,
  none = CL_ADDRESS_NONE
};

enum class filtering_mode : unsigned int {
  nearest = CL_FILTER_NEAREST,
  linear = CL_FILTER_LINEAR
};

enum class coordinate_normalization_mode : unsigned int {
  normalized = 1,
  unnormalized = 0
};

namespace detail {
template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder>
class image_accessor;
}

class __SYCL_EXPORT sampler {
public:
  sampler(coordinate_normalization_mode normalizationMode,
          addressing_mode addressingMode, filtering_mode filteringMode);

  sampler(cl_sampler clSampler, const context &syclContext);

  sampler(const sampler &rhs) = default;

  sampler(sampler &&rhs) = default;

  sampler &operator=(const sampler &rhs) = default;

  sampler &operator=(sampler &&rhs) = default;

  bool operator==(const sampler &rhs) const;

  bool operator!=(const sampler &rhs) const;

  addressing_mode get_addressing_mode() const;

  filtering_mode get_filtering_mode() const;

  coordinate_normalization_mode get_coordinate_normalization_mode() const;

private:
#ifdef __SYCL_DEVICE_ONLY__
  detail::sampler_impl impl;
  void __init(__ocl_sampler_t Sampler) { impl.m_Sampler = Sampler; }
  char padding[sizeof(std::shared_ptr<detail::sampler_impl>) - sizeof(impl)];

public:
  sampler() = default;

private:
#else
  std::shared_ptr<detail::sampler_impl> impl;
  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);
#endif
  template <typename DataT, int Dimensions, cl::sycl::access::mode AccessMode,
            cl::sycl::access::target AccessTarget, access::placeholder IsPlaceholder>
  friend class detail::image_accessor;
};
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

namespace std {
template <> struct hash<cl::sycl::sampler> {
  size_t operator()(const cl::sycl::sampler &s) const {
#ifdef __SYCL_DEVICE_ONLY__
    (void)s;
    return 0;
#else
    return hash<std::shared_ptr<cl::sycl::detail::sampler_impl>>()(
        cl::sycl::detail::getSyclObjImpl(s));
#endif
  }
};
} // namespace std
