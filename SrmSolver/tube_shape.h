#pragma once

#include <cmath>

#include <cuda_runtime_api.h>

#include "boundary_condition.h"
#include "math_utilities.h"

namespace kae {

template <class GpuGridT>
class TubeShape
{
public:

  using ElemType = typename GpuGridT::ElemType;

  __host__ __device__ ElemType operator()(unsigned i, unsigned j) const
  {
    const auto x = GpuGridT::hx * i;
    const auto y = GpuGridT::hy * j;

    const auto distanceToLeftWall   = xLeft - x;
    const auto distanceToRightWall  = x - (xLeft + length);
    const auto distanceToBottomWall = yBottom - y;
    const auto distanceToTopWall    = y - (yBottom + height);

    if (x < xLeft)
    {
      if (y < yBottom)
      {
        return std::hypot(distanceToLeftWall, distanceToBottomWall);
      }

      if (y > yTop)
      {
        return std::hypot(distanceToLeftWall, distanceToTopWall);
      }

      return distanceToLeftWall;
    }

    if (x > xRight)
    {
      if (y < yBottom)
      {
        return std::hypot(distanceToRightWall, distanceToBottomWall);
      }

      if (y > yTop)
      {
        return std::hypot(distanceToRightWall, distanceToTopWall);
      }

      return distanceToRightWall;
    }

    if (y < yBottom || y > yTop)
    {
      return absmin(distanceToBottomWall, distanceToTopWall);
    }

    return absmin(distanceToLeftWall, distanceToRightWall, distanceToBottomWall, distanceToTopWall);
  }

  __host__ __device__ static bool shouldApplyScheme(unsigned i, unsigned j) { return false; }

  __host__ __device__ static bool isPointOnGrain(ElemType x, ElemType y) { return false; }

  __host__ __device__ static EBoundaryCondition getBoundaryCondition(ElemType x, ElemType y)
  {
    if (std::fabs(x - xLeft) < static_cast<ElemType>(1e-6))
    {
      return EBoundaryCondition::eMassFlowInlet;
    }

    if (std::fabs(x - xRight) < static_cast<ElemType>(1e-5))
    {
      return EBoundaryCondition::ePressureOutlet;
    }

    return EBoundaryCondition::eWall;
  }

  __host__ __device__ static ElemType getRadius(unsigned i, unsigned j) { return j * GpuGridT::hy - yBottom; }

private:

  constexpr static ElemType height{ static_cast<ElemType>(1.0) };
  constexpr static ElemType yBottom{ static_cast<ElemType>(0.5) };
  constexpr static ElemType yTop{ yBottom + height };

  constexpr static ElemType length{ static_cast<ElemType>(2.0) };
  constexpr static ElemType xLeft{ static_cast<ElemType>(0.5) };
  constexpr static ElemType xRight{ xLeft + length };
};

} // namespace kae

#include "srm_shape_with_umbrella_def.h"
