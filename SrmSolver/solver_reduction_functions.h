#pragma once

#include "cuda_includes.h"

#include "cuda_float_types.h"
#include "delta_dirac_function.h"
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
CudaFloatT<2U, ElemT> getMaxWaveSpeeds(const thrust::device_vector<GasStateT>& values,
                                       const thrust::device_vector<ElemT>& currPhi)
{
  const auto first = thrust::make_transform_iterator(std::begin(values), kae::WaveSpeedXY{});
  const auto last = thrust::make_transform_iterator(std::end(values), kae::WaveSpeedXY{});

  const auto zipFirst = thrust::make_zip_iterator(thrust::make_tuple(first, std::begin(currPhi)));
  const auto zipLast = thrust::make_zip_iterator(thrust::make_tuple(last, std::end(currPhi)));

  const auto takeInner = [] __host__ __device__(
    const thrust::tuple<CudaFloatT<2U, ElemT>, ElemT> & conservativeVariables)
  {
    const auto level = thrust::get<1U>(conservativeVariables);
    if (level >= 0)
      return CudaFloatT<2U, ElemT>{};

    return thrust::get<0U>(conservativeVariables);
  };

  const auto transformFirst = thrust::make_transform_iterator(zipFirst, takeInner);
  const auto transformLast = thrust::make_transform_iterator(zipLast, takeInner);

  return thrust::reduce(first, last, CudaFloatT<2U, ElemT>{ 0, 0 }, kae::ElemwiseMax{});
}

template <class GpuGridT, class GasStateT, class ElemT = typename GasStateT::ElemType>
ElemT getDeltaT(const thrust::device_vector<GasStateT>& values,
                const thrust::device_vector<ElemT>& currPhi,
                ElemT courant)
{
  CudaFloatT<2U, ElemT> lambdas = detail::getMaxWaveSpeeds(values, currPhi);
  return courant * GpuGridT::hx * GpuGridT::hy / (GpuGridT::hx * lambdas.x + GpuGridT::hy * lambdas.y);
}

template <class GasStateT, class ElemT = typename GasStateT::ElemType>
CudaFloatT<4U, ElemT> getMaxEquationDerivatives(const thrust::device_vector<GasStateT>& prevValues,
                                                const thrust::device_vector<GasStateT>& currValues,
                                                const thrust::device_vector<ElemT>& currPhi,
                                                ElemT dt)
{
  const auto prevFirst = thrust::make_transform_iterator(std::begin(prevValues), kae::ConservativeVariables{});
  const auto prevLast = thrust::make_transform_iterator(std::end(prevValues), kae::ConservativeVariables{});

  const auto currFirst = thrust::make_transform_iterator(std::begin(currValues), kae::ConservativeVariables{});
  const auto currLast = thrust::make_transform_iterator(std::end(currValues), kae::ConservativeVariables{});

  const auto zipFirst = thrust::make_zip_iterator(thrust::make_tuple(prevFirst, currFirst, std::begin(currPhi)));
  const auto zipLast = thrust::make_zip_iterator(thrust::make_tuple(prevLast, currLast, std::end(currPhi)));

  const auto toDerivatives = [dt] __host__ __device__(
    const thrust::tuple<CudaFloatT<4U, ElemT>, CudaFloatT<4U, ElemT>, ElemT> & conservativeVariables)
  {
    const auto prevVariable = thrust::get<0U>(conservativeVariables);
    const auto currVariable = thrust::get<1U>(conservativeVariables);
    const auto level = thrust::get<2U>(conservativeVariables);
    if (level >= 0)
      return CudaFloatT<4U, ElemT>{};

    return CudaFloatT<4U, ElemT>{ (currVariable.x - prevVariable.x) / dt,
      (currVariable.y - prevVariable.y) / dt,
      (currVariable.z - prevVariable.z) / dt,
      (currVariable.w - prevVariable.w) / dt };
  };

  const auto transformFirst = thrust::make_transform_iterator(zipFirst, toDerivatives);
  const auto transformLast = thrust::make_transform_iterator(zipLast, toDerivatives);
  return thrust::reduce(transformFirst, transformLast, CudaFloatT<4U, ElemT>{}, kae::ElemwiseAbsMax{});
}

template <class GpuGridT, class ShapeT, class ElemT = typename GpuGridT::ElemType>
ElemT getChamberVolume(const thrust::device_vector<ElemT> & currPhi)
{
  static thread_local thrust::device_vector<unsigned> indexVector = generateIndexMatrix<unsigned>(currPhi.size());

  const auto tupleFirst = thrust::make_tuple(std::begin(indexVector), std::begin(currPhi));
  const auto tupleLast = thrust::make_tuple(std::end(indexVector), std::end(currPhi));

  const auto zipFirst = thrust::make_zip_iterator(tupleFirst);
  const auto zipLast  = thrust::make_zip_iterator(tupleLast);

  const auto toVolume = [] __host__ __device__(
    const thrust::tuple<unsigned, ElemT> & tuple)
  {
    const auto index     = thrust::get<0U>(tuple);
    const auto i         = index % GpuGridT::nx;
    const auto j         = index / GpuGridT::nx;
    if ((i >= GpuGridT::nx) || (j >= GpuGridT::ny))
    {
      return static_cast<ElemT>(0.0);
    }

    const auto isChamber = ShapeT::isChamber(i * GpuGridT::hx, j * GpuGridT::hy);
    if (!isChamber)
    {
      return static_cast<ElemT>(0.0);
    }

    const auto y = ShapeT::getRadius(i, j);
    return ((thrust::get<1U>(tuple) > 0) ? 
               static_cast<ElemT>(0.0) : 
               2 * static_cast<ElemT>(M_PI) * y * GpuGridT::hx * GpuGridT::hy);
  };

  const auto transformFirst = thrust::make_transform_iterator(zipFirst, toVolume);
  const auto transformLast = thrust::make_transform_iterator(zipLast, toVolume);

  return thrust::reduce(transformFirst, transformLast, static_cast<ElemT>(0.0));
}

template <class GpuGridT, class ShapeT, class GasStateT, class ElemT = typename GpuGridT::ElemType>
ElemT getPressureIntegral(const thrust::device_vector<GasStateT>& gasValues,
                          const thrust::device_vector<ElemT>& currPhi)
{
  static thread_local thrust::device_vector<unsigned> indexVector = generateIndexMatrix<unsigned>(currPhi.size());

  const auto tupleFirst = thrust::make_tuple(std::begin(gasValues), std::begin(indexVector), std::begin(currPhi));
  const auto tupleLast = thrust::make_tuple(std::end(gasValues), std::end(indexVector), std::end(currPhi));

  const auto zipFirst = thrust::make_zip_iterator(tupleFirst);
  const auto zipLast = thrust::make_zip_iterator(tupleLast);

  const auto toVolume = [] __host__ __device__(
    const thrust::tuple<GasStateT, unsigned, ElemT> & tuple)
  {
    const auto index = thrust::get<1U>(tuple);
    const auto i = index % GpuGridT::nx;
    const auto j = index / GpuGridT::nx;
    if ((i >= GpuGridT::nx) || (j >= GpuGridT::ny))
    {
      return static_cast<ElemT>(0.0);
    }

    const auto isChamber = ShapeT::isChamber(i * GpuGridT::hx, j * GpuGridT::hy);
    if (!isChamber)
    {
      return static_cast<ElemT>(0.0);
    }

    const auto y = ShapeT::getRadius(i, j);
    return ((thrust::get<2U>(tuple) > 0) ? 
              static_cast<ElemT>(0.0) : 
              2 * static_cast<ElemT>(M_PI) * y * P::get(thrust::get<0U>(tuple)) * GpuGridT::hx * GpuGridT::hy);
  };

  const auto transformFirst = thrust::make_transform_iterator(zipFirst, toVolume);
  const auto transformLast = thrust::make_transform_iterator(zipLast, toVolume);

  return thrust::reduce(transformFirst, transformLast, static_cast<ElemT>(0.0));
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

  const auto tupleFirst = thrust::make_tuple(std::begin(indexVector), 
                                             std::begin(currPhi),
                                             std::begin(normals));
  const auto tupleLast = thrust::make_tuple(std::end(indexVector), 
                                            std::end(currPhi),
                                            std::end(normals));

  const auto zipFirst = thrust::make_zip_iterator(tupleFirst);
  const auto zipLast = thrust::make_zip_iterator(tupleLast);

  const auto toVolume = [] __host__ __device__(
    const thrust::tuple<unsigned, ElemT, CudaFloatT<2U, ElemT>> & tuple)
  {
    const auto index   = thrust::get<0U>(tuple);
    const auto level   = thrust::get<1U>(tuple);
    const auto normals = thrust::get<2U>(tuple);

    const auto i = index % GpuGridT::nx;
    const auto j = index / GpuGridT::nx;
    if ((i >= GpuGridT::nx) || (j >= GpuGridT::ny))
    {
      return static_cast<ElemT>(0.0);
    }

    const auto xSurface = i * GpuGridT::hx - level * normals.x;
    const auto ySurface = j * GpuGridT::hy - level * normals.y;

    const auto isBurningSurface = ShapeT::isBurningSurface(xSurface, ySurface);
    if (!isBurningSurface)
    {
      return static_cast<ElemT>(0.0);
    }

    const auto y = ShapeT::getRadius(i, j);
    return 2 * static_cast<ElemT>(M_PI) * y * deltaDiracFunction(level, GpuGridT::hx) * GpuGridT::hx * GpuGridT::hy;
  };

  const auto transformFirst = thrust::make_transform_iterator(zipFirst, toVolume);
  const auto transformLast = thrust::make_transform_iterator(zipLast, toVolume);

  return thrust::reduce(transformFirst, transformLast, static_cast<ElemT>(0.0));
}

template <class GpuGridT,
          class ShapeT,
          class PropellantPropertiesT,
          class ElemT = typename GpuGridT::ElemType>
ElemT getTheoreticalBoriPressure(const thrust::device_vector<ElemT>& currPhi,
                                 const thrust::device_vector<CudaFloatT<2U, ElemT>>& normals)
{
  constexpr auto kappa = PropellantPropertiesT::kappa;
  const auto burningSurface = getBurningSurface<GpuGridT, ShapeT>(currPhi, normals);
  const auto boriPressure = std::pow(
    burningSurface * PropellantPropertiesT::mt * std::sqrt((kappa - 1) / kappa * PropellantPropertiesT::H0) /
    PropellantPropertiesT::gammaComplex / ShapeT::getFCritical(), 1 / (1 - PropellantPropertiesT::nu));
  return boriPressure;
}

} // namespace detail

} // namespace kae
