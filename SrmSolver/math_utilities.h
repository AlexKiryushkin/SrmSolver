#pragma once

#include "cuda_includes.h"

namespace kae {

namespace detail {

struct AbsMinComparator
{
  template <class ElemT>
  HOST_DEVICE bool operator()(ElemT lhsValue, ElemT rhsValue) const
  {
    return std::fabs(lhsValue) < std::fabs(rhsValue);
  }
};

struct AbsMaxComparator
{
  template <class ElemT>
  HOST_DEVICE bool operator()(ElemT lhsValue, ElemT rhsValue) const
  {
    return std::fabs(lhsValue) > std::fabs(rhsValue);
  }
};

} // namespace detail

template <class ElemT>
HOST_DEVICE ElemT absmin(ElemT lhsValue, ElemT rhsValue)
{
  return thrust::min(lhsValue, rhsValue, detail::AbsMinComparator{});
}

template <class T, class... Rest>
HOST_DEVICE T absmin(T value, Rest... restValues)
{
  return absmin(value, absmin(restValues...));
}


template <class ElemT>
HOST_DEVICE ElemT absmax(ElemT lhsValue, ElemT rhsValue)
{
  return thrust::max(lhsValue, rhsValue, detail::AbsMinComparator{});
}

template <class T, class... Rest>
HOST_DEVICE T absmax(T value, Rest... restValues)
{
  return absmax(value, absmax(restValues...));
}

template <class ElemT>
HOST_DEVICE ElemT sqr(ElemT value)
{
  return value * value;
}

struct ElemwiseMax
{
  HOST_DEVICE float2 operator()(float2 lhsValue, float2 rhsValue) const
  {
    return { thrust::max(lhsValue.x, rhsValue.x), thrust::max(lhsValue.y, rhsValue.y) };
  }

  HOST_DEVICE double2 operator()(double2 lhsValue, double2 rhsValue) const
  {
    return { thrust::max(lhsValue.x, rhsValue.x), thrust::max(lhsValue.y, rhsValue.y) };
  }
};

struct ElemwiseAbsMax
{
  HOST_DEVICE float4 operator()(float4 lhsValue, float4 rhsValue) const
  {
    return { absmax(lhsValue.x, rhsValue.x),
             absmax(lhsValue.y, rhsValue.y),
             absmax(lhsValue.z, rhsValue.z),
             absmax(lhsValue.w, rhsValue.w) };
  }

  HOST_DEVICE double4 operator()(double4 lhsValue, double4 rhsValue) const
  {
    return { absmax(lhsValue.x, rhsValue.x),
             absmax(lhsValue.y, rhsValue.y),
             absmax(lhsValue.z, rhsValue.z),
             absmax(lhsValue.w, rhsValue.w) };
  }
};

struct TransformCoordinates
{
  HOST_DEVICE float2 operator()(float2 coordinates, float2 n) const
  {
    return { coordinates.x * n.x + coordinates.y * n.y, -coordinates.x * n.y + coordinates.y * n.x };
  }

  HOST_DEVICE double2 operator()(double2 coordinates, double2 n) const
  {
    return { coordinates.x * n.x + coordinates.y * n.y, -coordinates.x * n.y + coordinates.y * n.x };
  }
};

} // namespace kae
