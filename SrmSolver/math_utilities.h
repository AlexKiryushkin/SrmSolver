#pragma once

#include <thrust/extrema.h>

#include "cuda_float_types.h"

namespace kae {

namespace detail {

struct AbsMinComparator
{
  template <class ElemT>
  __host__ __device__ bool operator()(ElemT lhsValue, ElemT rhsValue) const
  {
    return std::fabs(lhsValue) < std::fabs(rhsValue);
  }
};

struct AbsMaxComparator
{
  template <class ElemT>
  __host__ __device__ bool operator()(ElemT lhsValue, ElemT rhsValue) const
  {
    return std::fabs(lhsValue) > std::fabs(rhsValue);
  }
};

} // namespace detail

template <class ElemT>
__host__ __device__ ElemT absmin(ElemT lhsValue, ElemT rhsValue)
{
  return thrust::min(lhsValue, rhsValue, detail::AbsMinComparator{});
}

template <class T, class... Rest>
__host__ __device__ T absmin(T value, Rest... restValues)
{
  return absmin(value, absmin(restValues...));
}


template <class ElemT>
__host__ __device__ ElemT absmax(ElemT lhsValue, ElemT rhsValue)
{
  return thrust::max(lhsValue, rhsValue, detail::AbsMinComparator{});
}

template <class T, class... Rest>
__host__ __device__ T absmax(T value, Rest... restValues)
{
  return absmax(value, absmax(restValues...));
}

template <class ElemT>
__host__ __device__ ElemT sqr(ElemT value)
{
  return value * value;
}

struct ElemwiseMax
{
  __host__ __device__ float2 operator()(float2 lhsValue, float2 rhsValue) const
  {
    return { thrust::max(lhsValue.x, rhsValue.x), thrust::max(lhsValue.y, rhsValue.y) };
  }

  __host__ __device__ double2 operator()(double2 lhsValue, double2 rhsValue) const
  {
    return { thrust::max(lhsValue.x, rhsValue.x), thrust::max(lhsValue.y, rhsValue.y) };
  }
};

struct ElemwiseAbsMax
{
  __host__ __device__ float4 operator()(float4 lhsValue, float4 rhsValue) const
  {
    return { absmax(lhsValue.x, rhsValue.x),
             absmax(lhsValue.y, rhsValue.y),
             absmax(lhsValue.z, rhsValue.z),
             absmax(lhsValue.w, rhsValue.w) };
  }

  __host__ __device__ double4 operator()(double4 lhsValue, double4 rhsValue) const
  {
    return { absmax(lhsValue.x, rhsValue.x),
             absmax(lhsValue.y, rhsValue.y),
             absmax(lhsValue.z, rhsValue.z),
             absmax(lhsValue.w, rhsValue.w) };
  }
};

} // namespace kae
