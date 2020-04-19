#pragma once

#include "std_includes.h"
#include "cuda_includes.h"

#include "gpu_integrate_kernel.h"
#include "gpu_reinitialize_kernel.h"
#include "propellant_properties.h"

namespace kae {

template <class GpuGridT, class ShapeT>
GpuLevelSetSolver<GpuGridT, ShapeT>::GpuLevelSetSolver(ShapeT shape, 
                                                       unsigned iterationCount, 
                                                       ETimeDiscretizationOrder timeOrder)
  : m_currState(shape), m_prevState(shape), m_firstState(shape), m_secondState(shape)
{
  reinitialize(iterationCount, timeOrder);
}

template <class GpuGridT, class ShapeT>
template <class PropellantPropertiesT, class GasStateT>
auto GpuLevelSetSolver<GpuGridT, ShapeT>::integrateInTime(
  const GpuMatrix<GpuGridT, GasStateT> & gasValues,
  const GpuMatrix<GpuGridT, unsigned>  & closestIndices,
  unsigned                               iterationCount,
  ETimeDiscretizationOrder               timeOrder) -> ElemType
{
  ElemType t{ 0 };
  constexpr unsigned numOfReinitializeIterations{ 10U };
  for (unsigned i{ 0U }; i < iterationCount; ++i)
  {
    const auto dt = integrateInTimeStep<PropellantPropertiesT>(gasValues, closestIndices, timeOrder);
    t += dt;

    reinitialize(numOfReinitializeIterations, timeOrder);
  }

  return t;
}

template <class GpuGridT, class ShapeT>
template <class PropellantPropertiesT, class GasStateT>
auto GpuLevelSetSolver<GpuGridT, ShapeT>::integrateInTime(
  const GpuMatrix<GpuGridT, GasStateT> & gasValues,
  const GpuMatrix<GpuGridT, unsigned>  & closestIndices,
  ElemType                                  deltaT,
  ETimeDiscretizationOrder               timeOrder) -> ElemType
{
  constexpr unsigned numOfReinitializeIterations{ 10U };

  ElemType t{ 0 };
  while (t < deltaT)
  {
    const auto maxBurningSpeed = getMaxBurningRate<PropellantPropertiesT>(gasValues.values());
    const auto courant = static_cast<ElemType>(0.4);
    const auto maxDt = courant * GpuGridT::hx * GpuGridT::hy / (GpuGridT::hx + GpuGridT::hy) / maxBurningSpeed;
    const auto remainingTime = deltaT - t;
    auto dt = std::min(maxDt, remainingTime);
    dt = integrateInTimeStep<PropellantPropertiesT>(gasValues, closestIndices, timeOrder, dt);
    t += dt;

    reinitialize(numOfReinitializeIterations, timeOrder);
  }

  return t;
}

template <class GpuGridT, class ShapeT>
void GpuLevelSetSolver<GpuGridT, ShapeT>::reinitialize(unsigned iterationCount, ETimeDiscretizationOrder timeOrder)
{
  for (unsigned i{ 0U }; i < iterationCount; ++i)
  {
    reinitializeStep(timeOrder);
  }
}

template <class GpuGridT, class ShapeT>
template <class PropellantPropertiesT, class GasStateT>
auto GpuLevelSetSolver<GpuGridT, ShapeT>::integrateInTimeStep(
  const GpuMatrix<GpuGridT, GasStateT> & gasValues,
  const GpuMatrix<GpuGridT, unsigned>  & closestIndices,
  ETimeDiscretizationOrder               timeOrder) -> ElemType
{
  const auto courant = static_cast<ElemType>(0.4);
  const auto maxBurningSpeed = getMaxBurningRate<PropellantPropertiesT>(gasValues.values());
  const auto dt = courant * GpuGridT::hx * GpuGridT::hy / (GpuGridT::hx + GpuGridT::hy) / maxBurningSpeed;
  return integrateInTimeStep<PropellantPropertiesT>(gasValues, closestIndices, timeOrder, dt);
}

template <class GpuGridT, class ShapeT>
template <class PropellantPropertiesT, class GasStateT>
auto GpuLevelSetSolver<GpuGridT, ShapeT>::integrateInTimeStep(
  const GpuMatrix<GpuGridT, GasStateT> & gasValues,
  const GpuMatrix<GpuGridT, unsigned>  & closestIndices,
  ETimeDiscretizationOrder               timeOrder,
  ElemType dt) -> ElemType
{
  thrust::swap(m_prevState.values(), m_currState.values());
  switch (timeOrder)
  {
  case ETimeDiscretizationOrder::eOne:
    detail::integrateEqTvdSubStepWrapper<GpuGridT, ShapeT, PropellantPropertiesT>(
      getConstDevicePtr(m_prevState),
      thrust::device_ptr<const ElemType>{},
      getDevicePtr(m_currState),
      getConstDevicePtr(gasValues),
      getConstDevicePtr(closestIndices),
      dt, static_cast<ElemType>(1.0));
    break;

  case ETimeDiscretizationOrder::eTwo:
    detail::integrateEqTvdSubStepWrapper<GpuGridT, ShapeT, PropellantPropertiesT>(
      getConstDevicePtr(m_prevState),
      thrust::device_ptr<const ElemType>{},
      getDevicePtr(m_firstState),
      getConstDevicePtr(gasValues),
      getConstDevicePtr(closestIndices),
      dt, static_cast<ElemType>(1.0));

    detail::integrateEqTvdSubStepWrapper<GpuGridT, ShapeT, PropellantPropertiesT>(
      getConstDevicePtr(m_firstState),
      getConstDevicePtr(m_prevState),
      getDevicePtr(m_currState),
      getConstDevicePtr(gasValues),
      getConstDevicePtr(closestIndices),
      dt, static_cast<ElemType>(0.5));
    break;
  case ETimeDiscretizationOrder::eThree:
    detail::integrateEqTvdSubStepWrapper<GpuGridT, ShapeT, PropellantPropertiesT>(
      getConstDevicePtr(m_prevState),
      thrust::device_ptr<const ElemType>{},
      getDevicePtr(m_firstState),
      getConstDevicePtr(gasValues),
      getConstDevicePtr(closestIndices),
      dt, static_cast<ElemType>(1.0));

    detail::integrateEqTvdSubStepWrapper<GpuGridT, ShapeT, PropellantPropertiesT>(
      getConstDevicePtr(m_firstState),
      getConstDevicePtr(m_prevState),
      getDevicePtr(m_secondState),
      getConstDevicePtr(gasValues),
      getConstDevicePtr(closestIndices),
      dt, static_cast<ElemType>(0.25));

    detail::integrateEqTvdSubStepWrapper<GpuGridT, ShapeT, PropellantPropertiesT>(
      getConstDevicePtr(m_secondState),
      getConstDevicePtr(m_prevState),
      getDevicePtr(m_currState),
      getConstDevicePtr(gasValues),
      getConstDevicePtr(closestIndices),
      dt, static_cast<ElemType>(2.0 / 3.0));
    break;
  default:
    break;
  }

  return dt;
}

template <class GpuGridT, class ShapeT>
void GpuLevelSetSolver<GpuGridT, ShapeT>::reinitializeStep(ETimeDiscretizationOrder timeOrder)
{
  const auto courant = static_cast<ElemType>(0.8);
  const auto dt = courant * GpuGridT::hx * GpuGridT::hy / (GpuGridT::hx + GpuGridT::hy);

  thrust::swap(m_prevState.values(), m_currState.values());
  switch (timeOrder)
  {
  case ETimeDiscretizationOrder::eOne:
    detail::reinitializeTVDSubStepWrapper<GpuGridT, ShapeT>(
      getConstDevicePtr(m_prevState),
      thrust::device_ptr<const ElemType>{},
      getDevicePtr(m_currState),
      dt, static_cast<ElemType>(1.0));
    break;
  case ETimeDiscretizationOrder::eTwo:
    detail::reinitializeTVDSubStepWrapper<GpuGridT, ShapeT>(
      getConstDevicePtr(m_prevState),
      thrust::device_ptr<const ElemType>{},
      getDevicePtr(m_firstState),
      dt, static_cast<ElemType>(1.0));

    detail::reinitializeTVDSubStepWrapper<GpuGridT, ShapeT>(
      getConstDevicePtr(m_firstState),
      getConstDevicePtr(m_prevState),
      getDevicePtr(m_currState),
      dt, static_cast<ElemType>(0.5));
    break;
  case ETimeDiscretizationOrder::eThree:
    detail::reinitializeTVDSubStepWrapper<GpuGridT, ShapeT>(
      getConstDevicePtr(m_prevState),
      thrust::device_ptr<const ElemType>{},
      getDevicePtr(m_firstState),
      dt, static_cast<ElemType>(1.0));

    detail::reinitializeTVDSubStepWrapper<GpuGridT, ShapeT>(
      getConstDevicePtr(m_firstState),
      getConstDevicePtr(m_prevState),
      getDevicePtr(m_secondState),
      dt, static_cast<ElemType>(0.25));

    detail::reinitializeTVDSubStepWrapper<GpuGridT, ShapeT>(
      getConstDevicePtr(m_secondState),
      getConstDevicePtr(m_prevState),
      getDevicePtr(m_currState),
      dt, static_cast<ElemType>(2.0 / 3.0));
    break;
  default:
    break;
  }
}

template <class GpuGridT, class ShapeT>
template <class PropellantPropertiesT, class GasStateT>
auto GpuLevelSetSolver<GpuGridT, ShapeT>::getMaxBurningRate(const thrust::device_vector<GasStateT> & gasValues)
  -> ElemType
{
  auto first = thrust::make_transform_iterator(std::begin(gasValues), kae::BurningRate<PropellantPropertiesT>{});
  auto last = thrust::make_transform_iterator(std::end(gasValues), kae::BurningRate<PropellantPropertiesT>{});
  return thrust::reduce(first, last, static_cast<ElemType>(0.0), thrust::maximum<ElemType>{});
}

} // namespace kae
