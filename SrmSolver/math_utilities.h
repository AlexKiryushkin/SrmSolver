#pragma once

#include <thrust/extrema.h>

namespace kae {

namespace detail {

struct AbsMinComparator
{
  __host__ __device__ bool operator()(float lhsValue, float rhsValue) const
  {
    return std::fabs(lhsValue) < std::fabs(rhsValue);
  }
};

struct AbsMaxComparator
{
  __host__ __device__ bool operator()(float lhsValue, float rhsValue) const
  {
    return std::fabs(lhsValue) > std::fabs(rhsValue);
  }
};

} // namespace detail

inline __host__ __device__ float absmin(float lhsValue, float rhsValue)
{
  return thrust::min(lhsValue, rhsValue, detail::AbsMinComparator{});
}

template <class T, class... Rest>
__host__ __device__ float absmin(T value, Rest... restValues)
{
  return absmin(value, absmin(restValues...));
}

inline __host__ __device__ float absmax(float lhsValue, float rhsValue)
{
  return thrust::max(lhsValue, rhsValue, detail::AbsMinComparator{});
}

template <class T, class... Rest>
__host__ __device__ float absmax(T value, Rest... restValues)
{
  return absmax(value, absmax(restValues...));
}

inline __host__ __device__ float sqr(float value)
{
  return value * value;
}

struct ElemwiseMax
{
  __host__ __device__ float2 operator()(float2 lhsValue, float2 rhsValue) const
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
};

} // namespace kae
