#pragma once

#include "std_includes.h"
#include "cuda_includes.h"

#include "kernel/gpu_integrate_kernel.h"
#include "kernel/gpu_reinitialize_kernel.h"

namespace kae {

template <class ElemT, class ShapeT>
GpuLevelSetSolver<ElemT, ShapeT>::GpuLevelSetSolver(const GpuGridT<ElemType>& grid, ShapeT shape,
    unsigned iterationCount,
    ETimeDiscretizationOrder timeOrder)
    : m_grid{grid}, m_currState(m_grid.nx, m_grid.ny, shape), m_prevState(m_grid.nx, m_grid.ny, shape), m_firstState(m_grid.nx, m_grid.ny, shape), m_secondState(m_grid.nx, m_grid.ny, shape)
{
    reinitialize(iterationCount, timeOrder);
}

template <class ElemT, class ShapeT>
auto GpuLevelSetSolver<ElemT, ShapeT>::integrateInTime(
  const GpuMatrix<ElemType> & velocities,
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

template <class ElemT, class ShapeT>
auto GpuLevelSetSolver<ElemT, ShapeT>::integrateInTime(
  const GpuMatrix<ElemType> & velocities,
  ElemType                              deltaT,
  ETimeDiscretizationOrder              timeOrder) -> ElemType
{
  constexpr unsigned numOfReinitializeIterations{ 10U };

  ElemType t{ 0 };
  while (t < deltaT)
  {
    const auto maxBurningSpeed = getMaxVelocity(velocities.values());
    const auto courant = static_cast<ElemType>(0.4);
    const auto maxDt = courant * m_grid.hx * m_grid.hy / (m_grid.hx + m_grid.hy) / maxBurningSpeed;
    const auto remainingTime = deltaT - t;
    auto dt = std::min(maxDt, remainingTime);
    dt = integrateInTimeStep(velocities, timeOrder, dt);
    t += dt;

    reinitialize(numOfReinitializeIterations, timeOrder);
  }

  return t;
}

template <class ElemT, class ShapeT>
void GpuLevelSetSolver<ElemT, ShapeT>::reinitialize(unsigned iterationCount, ETimeDiscretizationOrder timeOrder)
{
  for (unsigned i{ 0U }; i < iterationCount; ++i)
  {
    reinitializeStep(timeOrder);
  }
}

template <class ElemT, class ShapeT>
auto GpuLevelSetSolver<ElemT, ShapeT>::integrateInTimeStep(
  const GpuMatrix<ElemType> & velocities,
  ETimeDiscretizationOrder              timeOrder) -> ElemType
{
  const auto courant = static_cast<ElemType>(0.4);
  const auto maxBurningSpeed = getMaxVelocity(velocities.values());
  const auto dt = courant * m_grid.hx * m_grid.hy / (m_grid.hx + m_grid.hy) / maxBurningSpeed;
  return integrateInTimeStep(velocities, timeOrder, dt);
}

template <class ElemT, class ShapeT>
auto GpuLevelSetSolver<ElemT, ShapeT>::integrateInTimeStep(
  const GpuMatrix<ElemType> & velocities,
  ETimeDiscretizationOrder               timeOrder,
  ElemType dt) -> ElemType
{
  thrust::swap(m_prevState.values(), m_currState.values());
  switch (timeOrder)
  {
  case ETimeDiscretizationOrder::eOne:
      detail::integrateEqTvdSubStepWrapper(
          getConstDevicePtr(m_prevState),
          thrust::device_ptr<const ElemType>{},
          getDevicePtr(m_currState),
          getConstDevicePtr(velocities), m_grid,
          dt, static_cast<ElemType>(1.0));
    break;

  case ETimeDiscretizationOrder::eTwo:
    detail::integrateEqTvdSubStepWrapper(
      getConstDevicePtr(m_prevState),
      thrust::device_ptr<const ElemType>{},
      getDevicePtr(m_firstState),
      getConstDevicePtr(velocities), m_grid,
      dt, static_cast<ElemType>(1.0));

    detail::integrateEqTvdSubStepWrapper(
      getConstDevicePtr(m_firstState),
      getConstDevicePtr(m_prevState),
      getDevicePtr(m_currState),
      getConstDevicePtr(velocities), m_grid,
      dt, static_cast<ElemType>(0.5));
    break;
  case ETimeDiscretizationOrder::eThree:
      detail::integrateEqTvdSubStepWrapper(
          getConstDevicePtr(m_prevState),
          thrust::device_ptr<const ElemType>{},
          getDevicePtr(m_firstState),
          getConstDevicePtr(velocities), m_grid,
          dt, static_cast<ElemType>(1.0));

    detail::integrateEqTvdSubStepWrapper(
        getConstDevicePtr(m_firstState),
        getConstDevicePtr(m_prevState),
        getDevicePtr(m_secondState),
        getConstDevicePtr(velocities), m_grid,
        dt, static_cast<ElemType>(0.25));

    detail::integrateEqTvdSubStepWrapper(
        getConstDevicePtr(m_secondState),
        getConstDevicePtr(m_prevState),
        getDevicePtr(m_currState),
        getConstDevicePtr(velocities), m_grid,
        dt, static_cast<ElemType>(2.0 / 3.0));
    break;
  default:
    break;
  }

  return dt;
}

template <class ElemT, class ShapeT>
void GpuLevelSetSolver<ElemT, ShapeT>::reinitializeStep(ETimeDiscretizationOrder timeOrder)
{
  const auto courant = static_cast<ElemType>(0.8);
  const auto dt = courant * m_grid.hx * m_grid.hy / (m_grid.hx + m_grid.hy);

  thrust::swap(m_prevState.values(), m_currState.values());
  switch (timeOrder)
  {
  case ETimeDiscretizationOrder::eOne:
    detail::reinitializeTVDSubStepWrapper<ShapeT>(
      getConstDevicePtr(m_prevState),
      thrust::device_ptr<const ElemType>{},
      getDevicePtr(m_currState), m_grid,
      dt, static_cast<ElemType>(1.0));
    break;
  case ETimeDiscretizationOrder::eTwo:
    detail::reinitializeTVDSubStepWrapper<ShapeT>(
      getConstDevicePtr(m_prevState),
      thrust::device_ptr<const ElemType>{},
      getDevicePtr(m_firstState), m_grid,
      dt, static_cast<ElemType>(1.0));

    detail::reinitializeTVDSubStepWrapper<ShapeT>(
      getConstDevicePtr(m_firstState),
      getConstDevicePtr(m_prevState),
      getDevicePtr(m_currState), m_grid,
      dt, static_cast<ElemType>(0.5));
    break;
  case ETimeDiscretizationOrder::eThree:
    detail::reinitializeTVDSubStepWrapper<ShapeT>(
      getConstDevicePtr(m_prevState),
      thrust::device_ptr<const ElemType>{},
      getDevicePtr(m_firstState), m_grid,
      dt, static_cast<ElemType>(1.0));

    detail::reinitializeTVDSubStepWrapper<ShapeT>(
      getConstDevicePtr(m_firstState),
      getConstDevicePtr(m_prevState),
      getDevicePtr(m_secondState), m_grid,
      dt, static_cast<ElemType>(0.25));

    detail::reinitializeTVDSubStepWrapper<ShapeT>(
      getConstDevicePtr(m_secondState),
      getConstDevicePtr(m_prevState),
      getDevicePtr(m_currState), m_grid,
      dt, static_cast<ElemType>(2.0 / 3.0));
    break;
  default:
    break;
  }
}

template <class ElemT, class ShapeT>
auto GpuLevelSetSolver<ElemT, ShapeT>::getMaxVelocity(const thrust::device_vector<ElemType> & velocities)
  -> ElemType
{
  return thrust::reduce(std::begin(velocities),
                        std::end(velocities), 
                        static_cast<ElemType>(0), 
                        thrust::maximum<ElemType>{});
}

} // namespace kae
