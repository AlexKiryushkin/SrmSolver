#pragma once

#include <SrmSolver/cuda_includes.h>

namespace tests {

template <class GpuGridT>
class CircleShape
{
public:

  using ElemType = typename GpuGridT::ElemType;

  constexpr static ElemType offset{ static_cast<ElemType>(2.0) };
  constexpr static ElemType radius{ static_cast<ElemType>(1.0) };

  __host__ __device__ static bool shouldApplyScheme(unsigned, unsigned) { return true; }

  __host__ __device__ ElemType operator()(unsigned i, unsigned j) const
  {
    ElemType x = i * GpuGridT::hx;
    ElemType y = j * GpuGridT::hy;
    return (x - offset) * (x - offset) + (y - offset) * (y - offset) - radius * radius;
  }

  __host__ __device__ static ElemType reinitializedValue(unsigned i, unsigned j)
  {
    ElemType x = i * GpuGridT::hx;
    ElemType y = j * GpuGridT::hy;
    return std::hypot(x - offset, y - offset) - radius;
  }

  __host__ __device__ static ElemType integratedValue(unsigned i, unsigned j, ElemType integrateTime)
  {
    ElemType x = i * GpuGridT::hx;
    ElemType y = j * GpuGridT::hy;
    return std::hypot(x - offset, y - offset) - radius - integrateTime;
  }
};

} // namespace tests
