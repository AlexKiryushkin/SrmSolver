#pragma once

#include <SrmSolver/cuda_includes.h>

namespace kae_tests {

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

template <class GpuGridT>
class SquareShape
{
public:

  using ElemType = typename GpuGridT::ElemType;

  constexpr static ElemType offset{ static_cast<ElemType>(1.5) };
  constexpr static ElemType length{ static_cast<ElemType>(1.0) };

  HOST_DEVICE static bool shouldApplyScheme(unsigned, unsigned) { return true; }

  HOST_DEVICE ElemType operator()(unsigned i, unsigned j) const
  {
    return reinitializedValue(i, j);
  }

  HOST_DEVICE static ElemType reinitializedValue(unsigned i, unsigned j)
  {
    ElemType x = i * GpuGridT::hx - offset;
    ElemType y = j * GpuGridT::hy - offset;
    if ((x < 0) && (y >= 0) && (y <= length))
    {
      return std::fabs(x);
    }
    else if ((x > length) && (y >= 0) && (y <= length))
    {
      return std::fabs(x - length);
    }
    else if ((y < 0) && (x >= 0) && (x <= length))
    {
      return std::fabs(y);
    }
    else if ((y > length) && (x >= 0) && (x <= length))
    {
      return std::fabs(y - length);
    }
    else if ((x < 0) && (y < 0))
    {
      return std::hypot(x, y);
    }
    else if ((x < 0) && (y > length))
    {
      return std::hypot(x, y - length);
    }
    else if ((x > length) && (y < 0))
    {
      return std::hypot(x - length, y);
    }
    else if ((x > length) && (y > length))
    {
      return std::hypot(x - length, y - length);
    }
    else
    {
      const auto distanceX = std::min(std::fabs(x), std::fabs(length - x));
      const auto distanceY = std::min(std::fabs(y), std::fabs(length - y));
      return -1.0 * std::min(distanceX, distanceY);
    }
  }
};

} // namespace kae_tests
