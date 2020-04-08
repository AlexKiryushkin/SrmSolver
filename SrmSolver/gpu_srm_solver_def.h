#pragma once

#include "cuda_includes.h"

#include "gas_state.h"
#include "gpu_build_ghost_to_closest_map_kernel.h"
#include "gpu_gas_dynamic_kernel.h"
#include "gpu_set_ghost_points_kernel.h"
#include "math_utilities.h"

namespace kae {

namespace detail {

template <class GasStateT, class ElemT = typename GasStateT::ElemType>
CudaFloatT<2U, ElemT> getMaxWaveSpeeds(const thrust::device_vector<GasStateT> & values,
                                       const thrust::device_vector<ElemT> & currPhi)
{
  const auto first = thrust::make_transform_iterator(std::begin(values), kae::WaveSpeedXY{});
  const auto last  = thrust::make_transform_iterator(std::end(values), kae::WaveSpeedXY{});

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
ElemT getDeltaT(const thrust::device_vector<GasStateT> & values,
                const thrust::device_vector<ElemT> & currPhi, 
                ElemT courant)
{
  CudaFloatT<2U, ElemT> lambdas = detail::getMaxWaveSpeeds(values, currPhi);
  return courant * GpuGridT::hx * GpuGridT::hy / (GpuGridT::hx * lambdas.x + GpuGridT::hy * lambdas.y);
}

template <class GasStateT, class ElemT = typename GasStateT::ElemType>
CudaFloatT<4U, ElemT> getMaxEquationDerivatives(const thrust::device_vector<GasStateT> & prevValues,
                                                const thrust::device_vector<GasStateT> & currValues,
                                                const thrust::device_vector<ElemT> & currPhi,
                                                ElemT dt)
{
  const auto prevFirst = thrust::make_transform_iterator(std::begin(prevValues), kae::ConservativeVariables{});
  const auto prevLast  = thrust::make_transform_iterator(std::end(prevValues),   kae::ConservativeVariables{});

  const auto currFirst = thrust::make_transform_iterator(std::begin(currValues), kae::ConservativeVariables{});
  const auto currLast  = thrust::make_transform_iterator(std::end(currValues),   kae::ConservativeVariables{});

  const auto zipFirst  = thrust::make_zip_iterator(thrust::make_tuple(prevFirst, currFirst, std::begin(currPhi)));
  const auto zipLast   = thrust::make_zip_iterator(thrust::make_tuple(prevLast,  currLast,  std::end(currPhi)));

  const auto toDerivatives = [dt] __host__ __device__(
    const thrust::tuple<CudaFloatT<4U, ElemT>, CudaFloatT<4U, ElemT>, ElemT> & conservativeVariables)
  {
    const auto prevVariable = thrust::get<0U>(conservativeVariables);
    const auto currVariable = thrust::get<1U>(conservativeVariables);
    const auto level        = thrust::get<2U>(conservativeVariables);
    if (level >= 0)
      return CudaFloatT<4U, ElemT>{};

    return CudaFloatT<4U, ElemT>{ (currVariable.x - prevVariable.x) / dt,
                                  (currVariable.y - prevVariable.y) / dt,
                                  (currVariable.z - prevVariable.z) / dt,
                                  (currVariable.w - prevVariable.w) / dt };
  };

  const auto transformFirst = thrust::make_transform_iterator(zipFirst, toDerivatives);
  const auto transformLast  = thrust::make_transform_iterator(zipLast, toDerivatives);
  return thrust::reduce(transformFirst, transformLast, CudaFloatT<4U, ElemT>{}, kae::ElemwiseAbsMax{});
}

template <class GpuGridT, class ShapeT, class PropellantPropertiesT, class GasStateT, class ElemT = typename GasStateT::ElemType>
void srmIntegrateTVDSubStepWrapper(thrust::device_ptr<GasStateT>                pPrevValue,
                                   thrust::device_ptr<const GasStateT>          pFirstValue,
                                   thrust::device_ptr<GasStateT>                pCurrValue,
                                   thrust::device_ptr<const ElemT>              pCurrentPhi,
                                   thrust::device_ptr<const unsigned>           pClosestIndices,
                                   thrust::device_ptr<const EBoundaryCondition> pBoundaryConditions,
                                   thrust::device_ptr<CudaFloatT<2U, ElemT>>    pNormals,
                                   ElemT dt, CudaFloatT<2U, ElemT> lambda, ElemT prevWeight)
{
  detail::setFirstOrderGhostValuesWrapper<GpuGridT, GasStateT, PropellantPropertiesT>(
    pPrevValue,
    pCurrentPhi,
    pClosestIndices,
    pBoundaryConditions,
    pNormals);

  detail::gasDynamicIntegrateTVDSubStepWrapper<GpuGridT, ShapeT, GasStateT>(
    pPrevValue,
    pFirstValue,
    pCurrValue,
    pCurrentPhi,
    dt, lambda, prevWeight);
}

} // namespace detail

template <class GpuGridT, class ShapeT, class GasStateT, class PropellantPropertiesT>
GpuSrmSolver<GpuGridT, ShapeT, GasStateT, PropellantPropertiesT>::GpuSrmSolver(
  ShapeT    shape, 
  GasStateT initialState,
  unsigned  iterationCount,
  ElemType  courant)
  : m_boundaryConditions{ EBoundaryCondition::eWall                               },
    m_closestIndices    { 0U                                                      },
    m_normals           { CudaFloatT<2U, ElemType>{ 0, 0 }                        },
    m_currState         { initialState                                            },
    m_prevState         { initialState                                            },
    m_firstState        { initialState                                            },
    m_secondState       { initialState                                            },
    m_levelSetSolver    { shape, iterationCount, ETimeDiscretizationOrder::eThree },
    m_courant           { courant                                                 }
{
}

template <class GpuGridT, class ShapeT, class GasStateT, class PropellantPropertiesT>
template <class CallbackT>
void GpuSrmSolver<GpuGridT, ShapeT, GasStateT, PropellantPropertiesT>::dynamicIntegrate(
  unsigned iterationCount, ElemType deltaT, ETimeDiscretizationOrder timeOrder, CallbackT callback)
{
  auto t{ static_cast<ElemType>(0.0) };
  for (unsigned i{ 0U }; i < iterationCount; ++i)
  {
    const auto deltaTGasDynamic = staticIntegrate(deltaT, timeOrder, callback);
    const auto maxDerivatives = getMaxEquationDerivatives();
    callback(m_currState, currPhi(), i, t, maxDerivatives);
    const auto dt = integrateInTime(deltaTGasDynamic);
    t += dt;
  }
}

template <class GpuGridT, class ShapeT, class GasStateT, class PropellantPropertiesT>
auto GpuSrmSolver<GpuGridT, ShapeT, GasStateT, PropellantPropertiesT>::staticIntegrate(
  unsigned iterationCount,
  ETimeDiscretizationOrder timeOrder) -> ElemType
{
  detail::findClosestIndicesWrapper<GpuGridT, ShapeT>(
    getDevicePtr(currPhi()),
    getDevicePtr(m_closestIndices),
    getDevicePtr(m_boundaryConditions),
    getDevicePtr(m_normals));

  auto t{ static_cast<ElemType>(0.0) };
  for (unsigned i{ 0U }; i < iterationCount; ++i)
  {
    const auto dt = staticIntegrateStep(timeOrder);
    t += dt;
  }

  return t;
}

template <class GpuGridT, class ShapeT, class GasStateT, class PropellantPropertiesT>
template <class CallbackT>
auto GpuSrmSolver<GpuGridT, ShapeT, GasStateT, PropellantPropertiesT>::staticIntegrate(
  ElemType deltaT,
  ETimeDiscretizationOrder timeOrder, 
  CallbackT callback) -> ElemType
{
  detail::findClosestIndicesWrapper<GpuGridT, ShapeT>(
    getDevicePtr(currPhi()),
    getDevicePtr(m_closestIndices),
    getDevicePtr(m_boundaryConditions),
    getDevicePtr(m_normals));

  unsigned i{ 0U };
  auto t{ static_cast<ElemType>(0.0) };
  while (t < deltaT)
  {
    const CudaFloatT<2U, ElemType> lambdas = detail::getMaxWaveSpeeds(m_currState.values(), currPhi().values());
    const CudaFloatT<2U, ElemType> multipliedLambdas{ static_cast<ElemType>(1.2) * lambdas.x, static_cast<ElemType>(1.2) * lambdas.y };
    const auto maxDt = m_courant * GpuGridT::hx * GpuGridT::hy / (GpuGridT::hx * lambdas.x + GpuGridT::hy * lambdas.y);
    const auto remainingTime = deltaT - t;
    auto dt = std::min(maxDt, remainingTime);
    dt = staticIntegrateStep(timeOrder, dt, multipliedLambdas);
    t += dt;
    ++i;
    if (i % 100U == 0U)
    {
      callback(m_currState);
    }
  }

  return t;
}

template <class GpuGridT, class ShapeT, class GasStateT, class PropellantPropertiesT>
auto GpuSrmSolver<GpuGridT, ShapeT, GasStateT, PropellantPropertiesT>::staticIntegrateStep(ETimeDiscretizationOrder timeOrder) -> ElemType
{
  const auto lambdas = detail::getMaxWaveSpeeds(m_currState.values());
  const auto dt = m_courant * GpuGridT::hx * GpuGridT::hy / (GpuGridT::hx * lambdas.x + GpuGridT::hy * lambdas.y);
  return staticIntegrateStep(timeOrder, dt, lambdas);
}

template <class GpuGridT, class ShapeT, class GasStateT, class PropellantPropertiesT>
auto GpuSrmSolver<GpuGridT, ShapeT, GasStateT, PropellantPropertiesT>::integrateInTime(ElemType deltaT) -> ElemType
{
  return m_levelSetSolver.template integrateInTime<PropellantPropertiesT>(
    m_currState,
    m_closestIndices,
    deltaT,
    ETimeDiscretizationOrder::eThree);
}

template <class GpuGridT, class ShapeT, class GasStateT, class PropellantPropertiesT>
auto GpuSrmSolver<GpuGridT, ShapeT, GasStateT, PropellantPropertiesT>::getMaxEquationDerivatives() const -> CudaFloatT<4U, ElemType>
{
  return detail::getMaxEquationDerivatives(
    m_prevState.values(),
    m_currState.values(),
    currPhi().values(),
    detail::getDeltaT<GpuGridT>(m_currState.values(), currPhi().values(), m_courant));
}

template <class GpuGridT, class ShapeT, class GasStateT, class PropellantPropertiesT>
auto GpuSrmSolver<GpuGridT, ShapeT, GasStateT, PropellantPropertiesT>::staticIntegrateStep(
  ETimeDiscretizationOrder timeOrder,
  ElemType dt,
  CudaFloatT<2U, ElemType> lambdas) -> ElemType
{
  thrust::swap(m_prevState.values(), m_currState.values());
  switch (timeOrder)
  {
  case ETimeDiscretizationOrder::eOne:
    detail::srmIntegrateTVDSubStepWrapper<GpuGridT, ShapeT, PropellantPropertiesT, GasStateT>(
      getDevicePtr(m_prevState),
      {},
      getDevicePtr(m_currState),
      getDevicePtr(currPhi()),
      getDevicePtr(m_closestIndices),
      getDevicePtr(m_boundaryConditions),
      getDevicePtr(m_normals),
      dt, lambdas, static_cast<ElemType>(1.0));
    break;

  case ETimeDiscretizationOrder::eTwo:
    detail::srmIntegrateTVDSubStepWrapper<GpuGridT, ShapeT, PropellantPropertiesT, GasStateT>(
      getDevicePtr(m_prevState),
      {},
      getDevicePtr(m_firstState),
      getDevicePtr(currPhi()),
      getDevicePtr(m_closestIndices),
      getDevicePtr(m_boundaryConditions),
      getDevicePtr(m_normals),
      dt, lambdas, static_cast<ElemType>(1.0));

    detail::srmIntegrateTVDSubStepWrapper<GpuGridT, ShapeT, PropellantPropertiesT, GasStateT>(
      getDevicePtr(m_firstState),
      getDevicePtr(m_prevState),
      getDevicePtr(m_currState),
      getDevicePtr(currPhi()),
      getDevicePtr(m_closestIndices),
      getDevicePtr(m_boundaryConditions),
      getDevicePtr(m_normals),
      dt, lambdas, static_cast<ElemType>(0.5));
    break;
  case ETimeDiscretizationOrder::eThree:
    detail::srmIntegrateTVDSubStepWrapper<GpuGridT, ShapeT, PropellantPropertiesT, GasStateT>(
      getDevicePtr(m_prevState),
      {},
      getDevicePtr(m_firstState),
      getDevicePtr(currPhi()),
      getDevicePtr(m_closestIndices),
      getDevicePtr(m_boundaryConditions),
      getDevicePtr(m_normals),
      dt, lambdas, static_cast<ElemType>(1.0));

    detail::srmIntegrateTVDSubStepWrapper<GpuGridT, ShapeT, PropellantPropertiesT, GasStateT>(
      getDevicePtr(m_firstState),
      getDevicePtr(m_prevState),
      getDevicePtr(m_secondState),
      getDevicePtr(currPhi()),
      getDevicePtr(m_closestIndices),
      getDevicePtr(m_boundaryConditions),
      getDevicePtr(m_normals),
      dt, lambdas, static_cast<ElemType>(0.25));

    detail::srmIntegrateTVDSubStepWrapper<GpuGridT, ShapeT, PropellantPropertiesT, GasStateT>(
      getDevicePtr(m_secondState),
      getDevicePtr(m_prevState),
      getDevicePtr(m_currState),
      getDevicePtr(currPhi()),
      getDevicePtr(m_closestIndices),
      getDevicePtr(m_boundaryConditions),
      getDevicePtr(m_normals),
      dt, lambdas, static_cast<ElemType>(2.0 / 3.0));
    break;
  default:
    break;
  }

  return dt;
}

} // namespace 
