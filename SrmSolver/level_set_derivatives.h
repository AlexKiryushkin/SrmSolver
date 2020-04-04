#pragma once

#include "cuda_float_types.h"
#include "math_utilities.h"

namespace kae {

namespace detail {

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

template <class GpuGridT, unsigned Step, bool IsPlus>
struct Derivative
{
  template <class ElemT>
  __host__ __device__ static ElemT get(const ElemT * arr, const unsigned i, const int offset)
  {
    return (arr[i + (offset + 1) * Step] - arr[i + offset * Step]) * GpuGridT::hxReciprocal;
  }
};

template <class GpuGridT, unsigned Step>
struct Derivative<GpuGridT, Step, false>
{
  template <class ElemT>
  __host__ __device__ static ElemT get(const ElemT * arr, const unsigned i, const int offset)
  {
    return (arr[i - offset * Step] - arr[i - (offset + 1) * Step]) * GpuGridT::hxReciprocal;
  }
};

template <class GpuGridT, unsigned Step, bool IsPlus, class ElemT>
__host__ __device__ ElemT getLevelSetDerivative(const ElemT * arr, const unsigned i)
{
  using Derivative = Derivative<GpuGridT, Step, IsPlus>;

  const ElemT v[derivativesCount] = { Derivative::get(arr, i, 2),
                                      Derivative::get(arr, i, 1),
                                      Derivative::get(arr, i, 0),
                                      Derivative::get(arr, i, -1),
                                      Derivative::get(arr, i, -2) };

  const ElemT flux[fluxesCount] =
  { LC<ElemT>[0][0] * v[0] + LC<ElemT>[0][1] * v[1] + LC<ElemT>[0][2] * v[2],
    LC<ElemT>[1][0] * v[1] + LC<ElemT>[1][1] * v[2] + LC<ElemT>[1][2] * v[3],
    LC<ElemT>[2][0] * v[2] + LC<ElemT>[2][1] * v[3] + LC<ElemT>[2][2] * v[4] };

  const ElemT s[fluxesCount] =
  { WC<ElemT>[0] * sqr(v[0] - 2 * v[1] + v[2]) + WC<ElemT>[1] * sqr(v[0] - 4 * v[1] + 3 * v[2]),
    WC<ElemT>[0] * sqr(v[1] - 2 * v[2] + v[3]) + WC<ElemT>[1] * sqr(v[1] - v[3]),

    WC<ElemT>[0] * sqr(v[2] - 2 * v[3] + v[4]) + 
    WC<ElemT>[1] * sqr(3 * v[2] - 4 * v[3] + v[4]) };

  constexpr ElemT epsilon = std::is_same<ElemT, float>::value ? static_cast<ElemT>(1e-12) : static_cast<ElemT>(1e-24);
  const ElemT alpha[fluxesCount] = {
    a<ElemT>[0] / sqr(s[0] + epsilon),
    a<ElemT>[1] / sqr(s[1] + epsilon),
    a<ElemT>[2] / sqr(s[2] + epsilon) };

  return (alpha[0] * flux[0] + alpha[1] * flux[1] + alpha[2] * flux[2]) / (alpha[0] + alpha[1] + alpha[2]);
}

template <class GpuGridT, unsigned Nx, class ElemT>
__host__ __device__ ElemT getLevelSetDerivative(const ElemT * arr, unsigned i, bool isPositiveVelocity)
{
  if (isPositiveVelocity)
  {
    ElemT val1 = thrust::max(getLevelSetDerivative<GpuGridT, Nx, false>(arr, i), static_cast<ElemT>(0.0));
    ElemT val2 = thrust::min(getLevelSetDerivative<GpuGridT, Nx, true>(arr, i), static_cast<ElemT>(0.0));

    return kae::absmax(val1, val2);
  }

  ElemT val1 = thrust::min(getLevelSetDerivative<GpuGridT, Nx, false>(arr, i), static_cast<ElemT>(0.0));
  ElemT val2 = thrust::max(getLevelSetDerivative<GpuGridT, Nx, true>(arr, i), static_cast<ElemT>(0.0));

  return kae::absmax(val1, val2);
}

template <class GpuGridT, unsigned Nx, class ElemT>
__host__ __device__ CudaFloatT<2U, ElemT> getLevelSetGradient(const ElemT * arr, unsigned i, bool isPositiveVelocity)
{
  ElemT derivativeX = getLevelSetDerivative<GpuGridT, 1>(arr, i, isPositiveVelocity);
  ElemT derivativeY = getLevelSetDerivative<GpuGridT, Nx>(arr, i, isPositiveVelocity);

  return { derivativeX, derivativeY };
}

template <class GpuGridT, unsigned Nx, class ElemT>
__host__ __device__ ElemT getLevelSetAbsGradient(const ElemT * arr, unsigned i, bool isPositiveVelocity)
{
  ElemT derivativeX = getLevelSetDerivative<GpuGridT, 1>(arr, i, isPositiveVelocity);
  ElemT derivativeY = getLevelSetDerivative<GpuGridT, Nx>(arr, i, isPositiveVelocity);

  return std::hypot(derivativeX, derivativeY);
}

} // namespace detail

} // namespace kae
