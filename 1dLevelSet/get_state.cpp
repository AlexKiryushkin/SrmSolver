
#include "get_state.h"

#include <algorithm>

namespace kae
{

template <class FloatT>
FloatT initialCircleFunctionValue(FloatT x, FloatT y)
{
  constexpr auto two = static_cast<FloatT>(2.0);
  constexpr auto one = static_cast<FloatT>(1.0);
  return (x - two) * (x - two) + (y - two) * (y - two) - one;
}

template <class FloatT>
FloatT reinitializedGoldCircleFunctionValue(FloatT x, FloatT y)
{
  constexpr auto two = static_cast<FloatT>(2.0);
  constexpr auto one = static_cast<FloatT>(1.0);
  return std::sqrt((x - two) * (x - two) + (y - two) * (y - two)) - one;
}

template <class FloatT>
FloatT initialSquareFunctionValue(FloatT x, FloatT y)
{
  constexpr auto offset = static_cast<FloatT>(1.5);
  constexpr auto length = static_cast<FloatT>(1.0);
  x -= offset;
  y -= offset;
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
    return -static_cast<FloatT>(1.0) * std::min(distanceX, distanceY);
  }
}

template <class FloatT>
FloatT reinitializedGoldSquareFunctionValue(FloatT x, FloatT y)
{
  return initialSquareFunctionValue(x, y);
}

template <class FloatT>
FloatT initialGradientVaryingCircleFunctionValue(FloatT x, FloatT y)
{
  constexpr auto three = static_cast<FloatT>(3.0);
  constexpr auto two = static_cast<FloatT>(2.0);
  constexpr auto one = static_cast<FloatT>(1.0);
  const auto multiplier = (x - one) * (x - one) + (y - three) * (y - three) + static_cast<FloatT>(0.1);
  return multiplier * (std::hypot(x - two, y - two) - one);
}

template <class FloatT>
FloatT initialFunctionValue(FloatT x, FloatT y, EShape shape)
{
  switch (shape)
  {
    case EShape::eCircle:
    {
      return initialCircleFunctionValue(x, y);
    }
    case EShape::eSquare:
    {
      return initialSquareFunctionValue(x, y);
    }
    case EShape::eGradientVaryingCircle:
    {
      return initialGradientVaryingCircleFunctionValue(x, y);
    }
    default:
    {
      return static_cast<FloatT>(0.0);
    }
  }
}

template <class FloatT>
FloatT reinitializedGoldFunctionValue(FloatT x, FloatT y, EShape shape)
{
  switch (shape)
  {
    case EShape::eCircle:
    {
      return reinitializedGoldCircleFunctionValue(x, y);
    }
    case EShape::eSquare:
    {
      return reinitializedGoldSquareFunctionValue(x, y);
    }
    case EShape::eGradientVaryingCircle:
    {
      return reinitializedGoldCircleFunctionValue(x, y);
    }
    default:
    {
      return static_cast<FloatT>(0.0);
    }
  }
}

template float initialFunctionValue<float>(float x, float y, EShape shape);
template double initialFunctionValue<double>(double x, double y, EShape shape);

template float reinitializedGoldFunctionValue<float>(float x, float y, EShape shape);
template double reinitializedGoldFunctionValue<double>(double x, double y, EShape shape);

}