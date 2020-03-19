#pragma once

#include "math_utilities.h"

namespace kae {

namespace detail {

constexpr unsigned derivativesCount = 5U;
constexpr unsigned fluxesCount = 3U;
constexpr float a[fluxesCount] = { 0.1f, 0.6f, 0.3f };

constexpr float LC[fluxesCount][fluxesCount] = 
  {  1.0f / 3.0f, -7.0f / 6.0f, 11.0f / 6.0f,
    -1.0f / 6.0f,  5.0f / 6.0f,  1.0f / 3.0f,
     1.0f / 3.0f,  5.0f / 6.0f, -1.0f / 6.0f };

constexpr float WC[2U] = { 13.0f / 12.0f, 0.25f };

template <class GpuGridT, unsigned Step, bool IsPlus>
struct Derivative
{
  __host__ __device__ static float get(const float * arr, const unsigned i, const int offset)
  {
    return (arr[i + (offset + 1) * Step] - arr[i + offset * Step]) * GpuGridT::hxReciprocal;
  }
};

template <class GpuGridT, unsigned Step>
struct Derivative<GpuGridT, Step, false>
{
  __host__ __device__ static float get(const float * arr, const unsigned i, const int offset)
  {
    return (arr[i - offset * Step] - arr[i - (offset + 1) * Step]) * GpuGridT::hxReciprocal;
  }
};

template <class GpuGridT, unsigned Step, bool IsPlus>
__host__ __device__ float getLevelSetDerivative(const float * arr, const unsigned i)
{
  using Derivative = Derivative<GpuGridT, Step, IsPlus>;

  const float v[derivativesCount] = { Derivative::get(arr, i, 2),
                                      Derivative::get(arr, i, 1),
                                      Derivative::get(arr, i, 0),
                                      Derivative::get(arr, i, -1),
                                      Derivative::get(arr, i, -2) };

  const float flux[fluxesCount] = 
  { LC[0][0] * v[0] + LC[0][1] * v[1] + LC[0][2] * v[2],
    LC[1][0] * v[1] + LC[1][1] * v[2] + LC[1][2] * v[3],
    LC[2][0] * v[2] + LC[2][1] * v[3] + LC[2][2] * v[4] };

  const float s[fluxesCount] =
  { WC[0] * sqr(v[0] - 2.0f * v[1] + v[2]) + WC[1] * sqr(v[0] - 4.0f * v[1] + 3.0f * v[2]),
    WC[0] * sqr(v[1] - 2.0f * v[2] + v[3]) + WC[1] * sqr(v[1] - v[3]),
    WC[0] * sqr(v[2] - 2.0f * v[3] + v[4]) + WC[1] * sqr(3.0f * v[2] - 4.0f * v[3] + v[4]) };

  constexpr float epsilon = 1e-6f;
  const float alpha[fluxesCount] = {
    a[0] / sqr(s[0] + epsilon),
    a[1] / sqr(s[1] + epsilon),
    a[2] / sqr(s[2] + epsilon) };

  return (alpha[0] * flux[0] + alpha[1] * flux[1] + alpha[2] * flux[2]) / (alpha[0] + alpha[1] + alpha[2]);
}

template <class GpuGridT, unsigned Nx>
__host__ __device__ float getLevelSetDerivative(const float * arr, unsigned i, bool isPositiveVelocity)
{
  if (isPositiveVelocity)
  {
    float val1 = thrust::max(getLevelSetDerivative<GpuGridT, Nx, false>(arr, i), 0.0f);
    float val2 = thrust::min(getLevelSetDerivative<GpuGridT, Nx, true>(arr, i), 0.0f);

    return kae::absmax(val1, val2);
  }

  float val1 = thrust::min(getLevelSetDerivative<GpuGridT, Nx, false>(arr, i), 0.0f);
  float val2 = thrust::max(getLevelSetDerivative<GpuGridT, Nx, true>(arr, i), 0.0f);

  return kae::absmax(val1, val2);
}

template <class GpuGridT, unsigned Nx>
__host__ __device__ float getLevelSetGradient(const float * arr, unsigned i, bool isPositiveVelocity)
{
  float derivativeX = getLevelSetDerivative<GpuGridT, 1>(arr, i, isPositiveVelocity);
  float derivativeY = getLevelSetDerivative<GpuGridT, Nx>(arr, i, isPositiveVelocity);

  return std::sqrt(derivativeX * derivativeX + derivativeY * derivativeY);
}

} // namespace detail

} // namespace kae
