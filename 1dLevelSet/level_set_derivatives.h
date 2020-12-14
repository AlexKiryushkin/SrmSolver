#pragma once

#include <algorithm>
#include <cmath>

namespace kae
{

constexpr std::size_t derivativesCount = 5U;
constexpr std::size_t fluxesCount = 3U;

template <class FloatT>
constexpr FloatT a[fluxesCount] = { static_cast<FloatT>(0.1), static_cast<FloatT>(0.6), static_cast<FloatT>(0.3) };

template <class FloatT>
constexpr FloatT LC[fluxesCount][fluxesCount] =
{ static_cast<FloatT>( 1.0 / 3.0), static_cast<FloatT>(-7.0 / 6.0), static_cast<FloatT>(11.0 / 6.0),
  static_cast<FloatT>(-1.0 / 6.0), static_cast<FloatT>( 5.0 / 6.0), static_cast<FloatT>( 1.0 / 3.0),
  static_cast<FloatT>( 1.0 / 3.0), static_cast<FloatT>( 5.0 / 6.0), static_cast<FloatT>(-1.0 / 6.0) };

template <class FloatT>
constexpr FloatT WC[2U] = { static_cast<FloatT>(13.0 / 12.0), static_cast<FloatT>(0.25) };

template <class FloatT>
FloatT sqr(FloatT value)
{
  return value * value;
}

template <class FloatT>
FloatT absmax(FloatT lhs, FloatT rhs)
{
  return std::max(lhs, rhs, [](auto left, auto right) { return std::fabs(left) < std::fabs(right); });
}

template <class FloatT>
FloatT absmin(FloatT lhs, FloatT rhs)
{
  return std::min(lhs, rhs, [](auto left, auto right) { return std::fabs(left) < std::fabs(right); });
}

template <class FloatT>
FloatT minmod(FloatT lhs, FloatT rhs)
{
  if (lhs * rhs < 0)
  {
    return static_cast<FloatT>(0);
  }

  return absmin(lhs, rhs);
}

template <bool IsPlus, class FloatT>
FloatT getLevelSetDerivative(const FloatT* arr, const FloatT * roots, const std::size_t i, FloatT h, std::size_t step)
{
  constexpr auto largeValue = static_cast<FloatT>(1000.0);
  constexpr std::size_t nPoints = 7U;
  FloatT xp[nPoints] = { 0, h, 2 * h, 3 * h, 4 * h, 5 * h, 6 * h };
  FloatT fp[nPoints] = { arr[i - 3 * step], arr[i - 2 * step], arr[i - step], arr[i], arr[i + step], arr[i + 2 * step], arr[i + 3 * step] };

  FloatT phi1[nPoints - 1] = { ( fp[1] - fp[0] ) / ( xp[1] - xp[0] ),
                               ( fp[2] - fp[1] ) / ( xp[2] - xp[1] ),
                               ( fp[3] - fp[2] ) / ( xp[3] - xp[2] ),
                               ( fp[4] - fp[3] ) / ( xp[4] - xp[3] ),
                               ( fp[5] - fp[4] ) / ( xp[5] - xp[4] ),
                               ( fp[6] - fp[5] ) / ( xp[6] - xp[5] )};

  FloatT phi2[nPoints - 2] = { ( phi1[1] - phi1[0] ) / ( xp[2] - xp[0] ),
                               ( phi1[2] - phi1[1] ) / ( xp[3] - xp[1] ),
                               ( phi1[3] - phi1[2] ) / ( xp[4] - xp[2] ),
                               ( phi1[4] - phi1[3] ) / ( xp[5] - xp[3] ),
                               ( phi1[5] - phi1[4] ) / ( xp[6] - xp[4] ) };

  FloatT phi3[nPoints - 3] = { ( phi2[1] - phi2[0] ) / ( xp[3] - xp[0] ),
                               ( phi2[2] - phi2[1] ) / ( xp[4] - xp[1] ),
                               ( phi2[3] - phi2[2] ) / ( xp[5] - xp[2] ),
                               ( phi2[4] - phi2[3] ) / ( xp[6] - xp[3] ) };

  if (IsPlus)
  {
    const auto intersect = ( ( fp[3] < 0 ) && ( fp[4] > 0 ) ) || ( ( fp[3] > 0 ) && ( fp[4] < 0 ) );
    const auto delta = intersect ? roots[i] : (xp[4] - xp[3]);
    if (intersect && delta == 0)
    {
      return static_cast<FloatT>(0.0);
    }

    const FloatT a = intersect ? -fp[3] / delta : phi1[3];
    const FloatT b = (std::fabs(phi2[2]) < std::fabs(phi2[3])) ?
      - (xp[3] - xp[2]) * delta * absmin(phi3[1], phi3[2]) : 
      - delta * (xp[3] - xp[5]) * absmin(phi3[2], phi3[3]);
    
    const FloatT c = minmod(phi2[2], phi2[3]);
    return a - delta * c + b;
  }
  else
  {
    const auto intersect = ((fp[2] < 0) && (fp[3] > 0)) || ((fp[2] > 0) && (fp[3] < 0));
    const auto delta = intersect ? h - roots[i - step] : (xp[3] - xp[2]);
    if (intersect && delta == 0)
    {
      return static_cast<FloatT>(0.0);
    }

    const FloatT a = intersect ? fp[3] / delta : phi1[2];
    const FloatT b = (std::fabs(phi2[1]) < std::fabs(phi2[2])) ?
      delta * (xp[3] - xp[1]) * absmin(phi3[0], phi3[1]) :
      delta * (xp[3] - xp[4]) * absmin(phi3[1], phi3[2]);
    const FloatT c = minmod(phi2[1], phi2[2]);
    return a + delta * c + b;
  }
  
}

template <class FloatT>
FloatT getLevelSetDerivative(const FloatT* arr, const FloatT * roots, std::size_t i, bool isPositiveVelocity, FloatT h, std::size_t step)
{
  if (isPositiveVelocity)
  {
    FloatT val1 = std::max(getLevelSetDerivative<false>(arr, roots, i, h, step), static_cast<FloatT>(0.0));
    FloatT val2 = std::min(getLevelSetDerivative<true>(arr, roots, i, h, step), static_cast<FloatT>(0.0));

    return absmax(val1, val2);
  }

  FloatT val1 = std::min(getLevelSetDerivative<false>(arr, roots, i, h, step), static_cast<FloatT>(0.0));
  FloatT val2 = std::max(getLevelSetDerivative<true>(arr, roots, i, h, step), static_cast<FloatT>(0.0));

  return absmax(val1, val2);
}

template <class FloatT>
FloatT getLevelSetAbsGradient(const FloatT* arr, const FloatT * roots, std::size_t i, bool isPositiveVelocity, FloatT h)
{
  FloatT derivativeX = getLevelSetDerivative<1>(arr, roots, i, isPositiveVelocity, h);

  return std::fabs(derivativeX);
}

template <class FloatT>
FloatT getLevelSetAbsGradient(const FloatT * arr, 
                              const FloatT * xRoots, 
                              const FloatT * yRoots, 
                              std::size_t    i,
                              std::size_t    nx,
                              bool           isPositiveVelocity, 
                              FloatT         hx, 
                              FloatT         hy)
{
  FloatT derivativeX = getLevelSetDerivative(arr, xRoots, i, isPositiveVelocity, hx, 1);
  FloatT derivativeY = getLevelSetDerivative(arr, yRoots, i, isPositiveVelocity, hy, nx);

  return std::hypot(derivativeX, derivativeY);
}

} // namespace kae