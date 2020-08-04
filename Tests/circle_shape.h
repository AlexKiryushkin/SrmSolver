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

  HOST_DEVICE static bool shouldApplyScheme(unsigned, unsigned) { return true; }

  HOST_DEVICE ElemType operator()(unsigned i, unsigned j) const
  {
    ElemType x = i * GpuGridT::hx;
    ElemType y = j * GpuGridT::hy;
    return (x - offset) * (x - offset) + (y - offset) * (y - offset) - radius * radius;
  }

  HOST_DEVICE static ElemType reinitializedValue(unsigned i, unsigned j)
  {
    ElemType x = i * GpuGridT::hx;
    ElemType y = j * GpuGridT::hy;
    return std::hypot(x - offset, y - offset) - radius;
  }

  HOST_DEVICE static ElemType integratedValue(unsigned i, unsigned j, ElemType integrateTime)
  {
    ElemType x = i * GpuGridT::hx;
    ElemType y = j * GpuGridT::hy;
    return std::hypot(x - offset, y - offset) - radius - integrateTime;
  }
};

} // namespace tests
