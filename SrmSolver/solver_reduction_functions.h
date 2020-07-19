#pragma once

#include "cuda_includes.h"

#include "cuda_float_types.h"
#include "delta_dirac_function.h"
#include "float4_arithmetics.h"
#include "gas_state.h"
#include "math_utilities.h"

namespace kae {

namespace detail {

template <class ElemT>
thrust::device_vector<ElemT> generateIndexMatrix(unsigned n)
{
  thrust::device_vector<ElemT> indexMatrix(n);
  thrust::sequence(std::begin(indexMatrix), std::end(indexMatrix));
  return indexMatrix;
}

template <class GasStateT, class ElemT = typename GasStateT::ElemType>
CudaFloatT<2U, ElemT> getMaxWaveSpeeds(const thrust::device_vector<GasStateT>& values)
{
  const auto first = thrust::make_transform_iterator(std::begin(values), kae::WaveSpeedXY{});
  const auto last = thrust::make_transform_iterator(std::end(values), kae::WaveSpeedXY{});
  return thrust::reduce(first, last, CudaFloatT<2U, ElemT>{ 0, 0 }, kae::ElemwiseMax{});
}

template <class GpuGridT, class GasStateT, class ElemT = typename GasStateT::ElemType>
ElemT getDeltaT(const thrust::device_vector<GasStateT>& values,
                ElemT courant)
{
  CudaFloatT<2U, ElemT> lambdas = detail::getMaxWaveSpeeds(values);
  return courant * GpuGridT::hx * GpuGridT::hy / (GpuGridT::hx * lambdas.x + GpuGridT::hy * lambdas.y);
}

template <class GasStateT, class ElemT = typename GasStateT::ElemType>
CudaFloatT<4U, ElemT> getMaxEquationDerivatives(const thrust::device_vector<GasStateT>& prevValues,
                                                const thrust::device_vector<GasStateT>& currValues,
                                                ElemT dt)
{
  const auto zipFirst = thrust::make_zip_iterator(thrust::make_tuple(std::begin(prevValues), std::begin(currValues)));
  const auto zipLast = thrust::make_zip_iterator(thrust::make_tuple(std::end(prevValues), std::end(currValues)));

  const auto toDerivatives = [] __device__(const thrust::tuple<GasStateT, GasStateT> & conservativeVariables)
  {
    const auto prevState = thrust::get<0U>(conservativeVariables);
    const auto currState = thrust::get<1U>(conservativeVariables);
    return ConservativeVariables::get(currState) - ConservativeVariables::get(prevState);
  };

  return (1 / dt) * thrust::transform_reduce(zipFirst, zipLast, toDerivatives, CudaFloatT<4U, ElemT>{}, ElemwiseAbsMax{});
}

template <class GpuGridT, class ShapeT, class ElemT = typename GpuGridT::ElemType>
ElemT getChamberVolume(const thrust::device_vector<ElemT> & currPhi)
{
  static thread_local auto indexVector = generateIndexMatrix<unsigned>(currPhi.size());

  const auto zipFirst = thrust::make_zip_iterator(thrust::make_tuple(std::begin(indexVector), std::begin(currPhi)));
  const auto zipLast  = thrust::make_zip_iterator(thrust::make_tuple(std::end(indexVector), std::end(currPhi)));

  const auto toVolume = [] __device__ (const thrust::tuple<unsigned, ElemT> & tuple)
  {
    const auto i         = thrust::get<0U>(tuple) % GpuGridT::nx;
    const auto j         = thrust::get<0U>(tuple) / GpuGridT::nx;
    if ((i >= GpuGridT::nx) || (j >= GpuGridT::ny) || !ShapeT::isChamber(i * GpuGridT::hx, j * GpuGridT::hy))
    {
      return static_cast<ElemT>(0.0);
    }

    return ((thrust::get<1U>(tuple) > 0) ? static_cast<ElemT>(0.0) : 
      2 * static_cast<ElemT>(M_PI) * ShapeT::getRadius(i, j) * GpuGridT::hx * GpuGridT::hy);
  };

  return thrust::transform_reduce(zipFirst, zipLast, toVolume, static_cast<ElemT>(0.0), thrust::plus<ElemT>{});
}

template <class GpuGridT, class ShapeT, class GasStateT, class ElemT = typename GpuGridT::ElemType>
ElemT getPressureIntegral(const thrust::device_vector<GasStateT>& gasValues,
                          const thrust::device_vector<ElemT>& currPhi)
{
  static thread_local auto indexVector = generateIndexMatrix<unsigned>(currPhi.size());

  const auto zipFirst = thrust::make_zip_iterator(
    thrust::make_tuple(std::begin(gasValues), std::begin(indexVector), std::begin(currPhi)));
  const auto zipLast = thrust::make_zip_iterator(
    thrust::make_tuple(std::end(gasValues), std::end(indexVector), std::end(currPhi)));

  const auto toVolume = [] __device__ (const thrust::tuple<GasStateT, unsigned, ElemT> & tuple)
  {
    const auto i = thrust::get<1U>(tuple) % GpuGridT::nx;
    const auto j = thrust::get<1U>(tuple) / GpuGridT::nx;
    if ((i >= GpuGridT::nx) || (j >= GpuGridT::ny) || !ShapeT::isChamber(i * GpuGridT::hx, j * GpuGridT::hy))
    {
      return static_cast<ElemT>(0.0);
    }

    return ((thrust::get<2U>(tuple) > 0) ? static_cast<ElemT>(0.0) : 
      2 * static_cast<ElemT>(M_PI) * ShapeT::getRadius(i, j) * P::get(thrust::get<0U>(tuple)) * GpuGridT::hx * GpuGridT::hy);
  };

  return thrust::transform_reduce(zipFirst, zipLast, toVolume, static_cast<ElemT>(0.0), thrust::plus<ElemT>{});
}

template <class GpuGridT, class ShapeT, class GasStateT, class ElemT = typename GpuGridT::ElemType>
ElemT getMaxChamberPressure(const thrust::device_vector<GasStateT>& gasValues,
                            const thrust::device_vector<ElemT>& currPhi)
{
  static thread_local auto indexVector = generateIndexMatrix<unsigned>(currPhi.size());

  const auto zipFirst = thrust::make_zip_iterator(
    thrust::make_tuple(std::begin(gasValues), std::begin(indexVector), std::begin(currPhi)));
  const auto zipLast = thrust::make_zip_iterator(
    thrust::make_tuple(std::end(gasValues), std::end(indexVector), std::end(currPhi)));

  const auto toPressure = [] __device__(const thrust::tuple<GasStateT, unsigned, ElemT> & tuple)
  {
    const auto i = thrust::get<1U>(tuple) % GpuGridT::nx;
    const auto j = thrust::get<1U>(tuple) / GpuGridT::nx;
    if ((i >= GpuGridT::nx) || (j >= GpuGridT::ny) || !ShapeT::isChamber(i * GpuGridT::hx, j * GpuGridT::hy))
    {
      return static_cast<ElemT>(0.0);
    }

    return ((thrust::get<2U>(tuple) > 0) ? static_cast<ElemT>(0.0) : P::get(thrust::get<0U>(tuple)));
  };

  return thrust::transform_reduce(zipFirst, zipLast, toPressure, static_cast<ElemT>(0.0), thrust::maximum<ElemT>{});
}

template <class GpuGridT, class ShapeT, class GasStateT, class ElemT = typename GpuGridT::ElemType>
ElemT getCalculatedBoriPressure(const thrust::device_vector<GasStateT>& gasValues,
                                const thrust::device_vector<ElemT>& currPhi)
{
  return getPressureIntegral<GpuGridT, ShapeT>(gasValues, currPhi) / getChamberVolume<GpuGridT, ShapeT>(currPhi);
}

template <class GpuGridT, class ShapeT, class ElemT = typename GpuGridT::ElemType>
ElemT getBurningSurface(const thrust::device_vector<ElemT>& currPhi,
                        const thrust::device_vector<CudaFloatT<2U, ElemT>>& normals)
{
  static thread_local thrust::device_vector<unsigned> indexVector = generateIndexMatrix<unsigned>(currPhi.size());

  const auto zipFirst = thrust::make_zip_iterator(
    thrust::make_tuple(std::begin(indexVector), std::begin(currPhi), std::begin(normals)));
  const auto zipLast = thrust::make_zip_iterator(
    thrust::make_tuple(std::end(indexVector), std::end(currPhi), std::end(normals)));

  const auto toVolume = [] __device__(const thrust::tuple<unsigned, ElemT, CudaFloatT<2U, ElemT>> & tuple)
  {
    const auto level   = thrust::get<1U>(tuple);
    const auto normals = thrust::get<2U>(tuple);

    const auto i = thrust::get<0U>(tuple) % GpuGridT::nx;
    const auto j = thrust::get<0U>(tuple) / GpuGridT::nx;
    const auto xSurface = i * GpuGridT::hx - level * normals.x;
    const auto ySurface = j * GpuGridT::hy - level * normals.y;
    if ((i >= GpuGridT::nx) || (j >= GpuGridT::ny) || !ShapeT::isBurningSurface(xSurface, ySurface))
    {
      return static_cast<ElemT>(0.0);
    }

    const auto y = ShapeT::getRadius(i, j);
    return 2 * static_cast<ElemT>(M_PI) * y * deltaDiracFunction(level, GpuGridT::hx) * GpuGridT::hx * GpuGridT::hy;
  };

  return thrust::transform_reduce(zipFirst, zipLast, toVolume, static_cast<ElemT>(0.0), thrust::plus<ElemT>{});
}

template <class GpuGridT,
          class ShapeT,
          class PhysicalPropertiesT,
          class ElemT = typename GpuGridT::ElemType>
ElemT getTheoreticalBoriPressure(const thrust::device_vector<ElemT>& currPhi,
                                 const thrust::device_vector<CudaFloatT<2U, ElemT>>& normals)
{
  constexpr auto kappa = PhysicalPropertiesT::kappa;
  const auto burningSurface = getBurningSurface<GpuGridT, ShapeT>(currPhi, normals);
  const auto boriPressure = std::pow(
    burningSurface * PhysicalPropertiesT::mt * std::sqrt((kappa - 1) / kappa * PhysicalPropertiesT::H0) /
    PhysicalPropertiesT::gammaComplex / ShapeT::getFCritical(), 1 / (1 - PhysicalPropertiesT::nu));
  return boriPressure;
}

template <class GpuGridT,
          class ShapeT,
          class GasStateT,
          class ElemT      = typename GpuGridT::ElemType,
          class ReturnType = thrust::tuple<ElemT, ElemT, ElemT, ElemT>>
ReturnType getMotorThrust(const thrust::device_vector<GasStateT> & gasValues,
                          const thrust::device_vector<ElemT> &     currPhi)
{
  static thread_local thrust::device_vector<unsigned> indexVector = generateIndexMatrix<unsigned>(currPhi.size());

  const auto zipFirst = thrust::make_zip_iterator(
    thrust::make_tuple(std::begin(gasValues), std::begin(indexVector), std::begin(currPhi)));
  const auto zipLast = thrust::make_zip_iterator(
    thrust::make_tuple(std::end(gasValues), std::end(indexVector), std::end(currPhi)));

  const auto toThrust = [] __device__(const thrust::tuple<GasStateT, unsigned, ElemT> & tuple)
  {
    const auto i = thrust::get<1U>(tuple) % GpuGridT::nx;
    const auto j = thrust::get<1U>(tuple) / GpuGridT::nx;
    const auto r = ShapeT::getRadius(i, j);
    const auto isInside = thrust::get<2U>(tuple) < static_cast<ElemT>(0.0);
    const auto isNearOutlet = ShapeT::getOutletCoordinate() - i * GpuGridT::hx <= GpuGridT::hx;
    if (!isInside || !isNearOutlet)
    {
      return ReturnType{};
    }

    const auto & gasState = thrust::get<0U>(tuple);
    const auto dS = 2 * static_cast<ElemT>(M_PI) * r * GpuGridT::hy;
    const auto dUS = gasState.ux * dS;
    const auto dG = MassFluxX::get(gasState) * dS;
    const auto dPS = P::get(gasState) * dS;
    return ReturnType{ dS, dUS, dG, dPS };
  };

  const auto sumUp = [] __device__(const ReturnType & lhs, const ReturnType & rhs)
  {
    return ReturnType{ thrust::get<0U>(lhs) + thrust::get<0U>(rhs),
                       thrust::get<1U>(lhs) + thrust::get<1U>(rhs),
                       thrust::get<2U>(lhs) + thrust::get<2U>(rhs),
                       thrust::get<3U>(lhs) + thrust::get<3U>(rhs) };
  };

  return thrust::transform_reduce(zipFirst, zipLast, toThrust, ReturnType{}, sumUp);
}

} // namespace detail

} // namespace kae
