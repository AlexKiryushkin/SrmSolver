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

  __host__ __device__ float operator()(unsigned i, unsigned j) const
  {
    auto x = GpuGridT::hx * i;
    auto y = GpuGridT::hy * j;

    auto distanceToLeftWall = xLeft - x;
    auto distanceToRightWall = x - (xLeft + length);
    auto distanceToBottomWall = yBottom - y;
    auto distanceToTopWall = y - (yBottom + height);

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

  __host__ __device__ static bool isPointOnGrain(float x, float y) { return false; }

  __host__ __device__ static EBoundaryCondition getBoundaryCondition(float x, float y)
  {
    if (std::fabs(x - xLeft) < 1e-6f)
    {
      return EBoundaryCondition::eMassFlowInlet;
    }

    if (std::fabs(x - xRight) < 1e-5f)
    {
      return EBoundaryCondition::ePressureOutlet;
    }

    return EBoundaryCondition::eWall;
  }

  __host__ __device__ static float getRadius(unsigned i, unsigned j) { return j * GpuGridT::hy - yBottom; }

private:

  constexpr static float height{ 1.0f };
  constexpr static float yBottom{ 0.5f };
  constexpr static float yTop{ yBottom + height };

  constexpr static float length{ 2.0f };
  constexpr static float xLeft{ 0.5f };
  constexpr static float xRight{ xLeft + length };
};

} // namespace kae

#include "srm_shape_with_umbrella_def.h"
