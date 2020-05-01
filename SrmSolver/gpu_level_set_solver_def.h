#pragma once

#include "std_includes.h"
#include "cuda_includes.h"

#include "gpu_integrate_kernel.h"
#include "gpu_reinitialize_kernel.h"

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
auto GpuLevelSetSolver<GpuGridT, ShapeT>::integrateInTime(
  const GpuMatrix<GpuGridT, ElemType> & velocities,
  unsigned                              iterationCount,
  ETimeDiscretizationOrder              timeOrder) -> ElemType
{
  ElemType t{ 0 };
  constexpr unsigned numOfReinitializeIterations{ 10U };
  for (unsigned i{ 0U }; i < iterationCount; ++i)
  {
    const auto dt = integrateInTimeStep(velocities, timeOrder);
    t += dt;

    reinitialize(numOfReinitializeIterations, timeOrder);
  }

  return t;
}

template <class GpuGridT, class ShapeT>
auto GpuLevelSetSolver<GpuGridT, ShapeT>::integrateInTime(
  const GpuMatrix<GpuGridT, ElemType> & velocities,
  ElemType                              deltaT,
  ETimeDiscretizationOrder              timeOrder) -> ElemType
{
  constexpr unsigned numOfReinitializeIterations{ 10U };

  ElemType t{ 0 };
  while (t < deltaT)
  {
    const auto maxBurningSpeed = getMaxVelocity(velocities.values());
    const auto courant = static_cast<ElemType>(0.4);
    const auto maxDt = courant * GpuGridT::hx * GpuGridT::hy / (GpuGridT::hx + GpuGridT::hy) / maxBurningSpeed;
    const auto remainingTime = deltaT - t;
    auto dt = std::min(maxDt, remainingTime);
    dt = integrateInTimeStep(velocities, timeOrder, dt);
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
auto GpuLevelSetSolver<GpuGridT, ShapeT>::integrateInTimeStep(
  const GpuMatrix<GpuGridT, ElemType> & velocities,
  ETimeDiscretizationOrder              timeOrder) -> ElemType
{
  const auto courant = static_cast<ElemType>(0.4);
  const auto maxBurningSpeed = getMaxVelocity(velocities.values());
  const auto dt = courant * GpuGridT::hx * GpuGridT::hy / (GpuGridT::hx + GpuGridT::hy) / maxBurningSpeed;
  return integrateInTimeStep(velocities, timeOrder, dt);
}

template <class GpuGridT, class ShapeT>
auto GpuLevelSetSolver<GpuGridT, ShapeT>::integrateInTimeStep(
  const GpuMatrix<GpuGridT, ElemType> & velocities,
  ETimeDiscretizationOrder               timeOrder,
  ElemType dt) -> ElemType
{
  thrust::swap(m_prevState.values(), m_currState.values());
  switch (timeOrder)
  {
  case ETimeDiscretizationOrder::eOne:
    detail::integrateEqTvdSubStepWrapper<GpuGridT>(
      getConstDevicePtr(m_prevState),
      thrust::device_ptr<const ElemType>{},
      getDevicePtr(m_currState),
      getConstDevicePtr(velocities),
      dt, static_cast<ElemType>(1.0));
    break;

  case ETimeDiscretizationOrder::eTwo:
    detail::integrateEqTvdSubStepWrapper<GpuGridT>(
      getConstDevicePtr(m_prevState),
      thrust::device_ptr<const ElemType>{},
      getDevicePtr(m_firstState),
      getConstDevicePtr(velocities),
      dt, static_cast<ElemType>(1.0));

    detail::integrateEqTvdSubStepWrapper<GpuGridT>(
      getConstDevicePtr(m_firstState),
      getConstDevicePtr(m_prevState),
      getDevicePtr(m_currState),
      getConstDevicePtr(velocities),
      dt, static_cast<ElemType>(0.5));
    break;
  case ETimeDiscretizationOrder::eThree:
    detail::integrateEqTvdSubStepWrapper<GpuGridT>(
      getConstDevicePtr(m_prevState),
      thrust::device_ptr<const ElemType>{},
      getDevicePtr(m_firstState),
      getConstDevicePtr(velocities),
      dt, static_cast<ElemType>(1.0));

    detail::integrateEqTvdSubStepWrapper<GpuGridT>(
      getConstDevicePtr(m_firstState),
      getConstDevicePtr(m_prevState),
      getDevicePtr(m_secondState),
      getConstDevicePtr(velocities),
      dt, static_cast<ElemType>(0.25));

    detail::integrateEqTvdSubStepWrapper<GpuGridT>(
      getConstDevicePtr(m_secondState),
      getConstDevicePtr(m_prevState),
      getDevicePtr(m_currState),
      getConstDevicePtr(velocities),
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
auto GpuLevelSetSolver<GpuGridT, ShapeT>::getMaxVelocity(const thrust::device_vector<ElemType> & velocities)
  -> ElemType
{
  return thrust::reduce(std::begin(velocities),
                        std::end(velocities), 
                        static_cast<ElemType>(0), 
                        thrust::maximum<ElemType>{});
}

} // namespace kae
