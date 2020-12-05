#pragma once

#include <algorithm>
#include <cmath>

namespace kae
{

constexpr unsigned derivativesCount = 5U;
constexpr unsigned fluxesCount = 3U;

template <class ElemT>
constexpr ElemT a[fluxesCount] = { 0.1, 0.6, 0.3 };

template <class ElemT>
constexpr ElemT LC[fluxesCount][fluxesCount] =
{  1.0 / 3.0, -7.0 / 6.0, 11.0 / 6.0,
  -1.0 / 6.0,  5.0 / 6.0,  1.0 / 3.0,
   1.0 / 3.0,  5.0 / 6.0, -1.0 / 6.0 };

template <class ElemT>
constexpr ElemT WC[2U] = { 13.0 / 12.0, 0.25 };

template <class ElemT>
ElemT sqr(ElemT value)
{
  return value * value;
}

template <class FloatT>
constexpr FloatT ctAbs(FloatT value)
{
  return value > 0 ? value : -value;
}

template <unsigned Step, bool IsPlus>
struct Derivative
{
  template <class ElemT>
  static ElemT get(const ElemT* arr, const ElemT* roots, const unsigned i, const int offset, ElemT h)
  {
    const auto rightValue = arr[i + (offset + 1) * Step];
    const auto leftValue = arr[i + offset * Step];
    const auto defaultDerivative = (rightValue - leftValue) / h;
    if (rightValue * leftValue < 0)
    {
      const auto root = roots[i + offset * Step];
      const auto delta = root - (i + offset * Step) * h;
      return -leftValue / delta;
    }

    return defaultDerivative;
  }
};

template <unsigned Step>
struct Derivative<Step, false>
{
  template <class ElemT>
  static ElemT get(const ElemT* arr, const ElemT* roots, const unsigned i, const int offset, ElemT h)
  {
    const auto rightValue = arr[i - offset * Step];
    const auto leftValue = arr[i - (offset + 1) * Step];
    const auto defaultDerivative = (rightValue - leftValue) / h;
    if (rightValue * leftValue < 0)
    {
      const auto root = roots[i - (offset + 1) * Step];
      const auto delta = root - (i - offset * Step) * h;
      return -rightValue / delta;
    }

    return defaultDerivative;
  }
};

template <unsigned Step, bool IsPlus, class ElemT>
ElemT getLevelSetDerivative(const ElemT* arr, const ElemT * roots, const unsigned i, ElemT h)
{
  using Derivative = Derivative<Step, IsPlus>;
  constexpr auto largeWeight = static_cast<ElemT>(1000.0);

  const auto midIdx = IsPlus ? i : i - 1;
  const ElemT v[derivativesCount] = { Derivative::get(arr, roots, i,  2, h),
                                      Derivative::get(arr, roots, i,  1, h),
                                      Derivative::get(arr, roots, i,  0, h),
                                      Derivative::get(arr, roots, i, -1, h),
                                      Derivative::get(arr, roots, i, -2, h) };

  const ElemT flux[fluxesCount] =
  { LC<ElemT>[0][0] * v[0] + LC<ElemT>[0][1] * v[1] + LC<ElemT>[0][2] * v[2],
    LC<ElemT>[1][0] * v[1] + LC<ElemT>[1][1] * v[2] + LC<ElemT>[1][2] * v[3],
    LC<ElemT>[2][0] * v[2] + LC<ElemT>[2][1] * v[3] + LC<ElemT>[2][2] * v[4] };

  ElemT s[fluxesCount] =
  { WC<ElemT>[0] * sqr(v[0] - 2 * v[1] + v[2]) + WC<ElemT>[1] * sqr(v[0] - 4 * v[1] + 3 * v[2]),
    WC<ElemT>[0] * sqr(v[1] - 2 * v[2] + v[3]) + WC<ElemT>[1] * sqr(v[1] - v[3]),
    WC<ElemT>[0] * sqr(v[2] - 2 * v[3] + v[4]) + WC<ElemT>[1] * sqr(3 * v[2] - 4 * v[3] + v[4]) };
  if (!std::isinf(roots[midIdx - 1]))
  {
    s[0] = largeWeight;
  }
  if (!std::isinf(roots[midIdx]))
  {
    s[1] = largeWeight;
  }
  if (!std::isinf(roots[midIdx + 1]))
  {
    s[2] = largeWeight;
  }

  const ElemT epsilon = h * h;
  const ElemT alpha[fluxesCount] = {
    a<ElemT>[0] / sqr(s[0] + epsilon),
    a<ElemT>[1] / sqr(s[1] + epsilon),
    a<ElemT>[2] / sqr(s[2] + epsilon) };

  return (alpha[0] * flux[0] + alpha[1] * flux[1] + alpha[2] * flux[2]) / (alpha[0] + alpha[1] + alpha[2]);
}

template <unsigned Nx, class ElemT>
ElemT getLevelSetDerivative(const ElemT* arr, const ElemT * roots, unsigned i, bool isPositiveVelocity, ElemT h)
{
  if (isPositiveVelocity)
  {
    ElemT val1 = std::max(getLevelSetDerivative<Nx, false>(arr, roots, i, h), static_cast<ElemT>(0.0));
    ElemT val2 = std::min(getLevelSetDerivative<Nx, true>(arr, roots, i, h), static_cast<ElemT>(0.0));

    return std::max(val1, val2, [](auto left, auto right) { return ctAbs(left) < ctAbs(right); });
  }

  ElemT val1 = std::min(getLevelSetDerivative<Nx, false>(arr, roots, i, h), static_cast<ElemT>(0.0));
  ElemT val2 = std::max(getLevelSetDerivative<Nx, true>(arr, roots, i, h), static_cast<ElemT>(0.0));

  return std::max(val1, val2, [](auto left, auto right) { return ctAbs(left) < ctAbs(right); });
}

template <class ElemT>
ElemT getLevelSetAbsGradient(const ElemT* arr, const ElemT * roots, unsigned i, bool isPositiveVelocity, ElemT h)
{
  ElemT derivativeX = getLevelSetDerivative<1>(arr, roots, i, isPositiveVelocity, h);

  return std::fabs(derivativeX);
}

} // namespace kae