#pragma once

#include "std_includes.h"
#include "cuda_includes.h"

#include "kernel/gpu_integrate_kernel.h"
#include "kernel/gpu_reinitialize_kernel.h"

namespace kae {

template <class ElemT>
GpuLevelSetSolver<ElemT>::GpuLevelSetSolver(const GpuGridT<ElemType>& grid, thrust::host_vector<ElemType> signedDistances, Shape<ElemType> shape,
    unsigned iterationCount,
    ETimeDiscretizationOrder timeOrder)
    : m_grid{ grid }, m_currState(m_grid.nx, m_grid.ny, signedDistances), m_prevState(m_grid.nx, m_grid.ny, signedDistances), m_firstState(m_grid.nx, m_grid.ny, signedDistances), m_secondState(m_grid.nx, m_grid.ny, signedDistances), m_shape{shape}
{
    reinitialize(iterationCount, timeOrder);
}

template <class ElemT>
auto GpuLevelSetSolver<ElemT>::integrateInTime(
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

template <class ElemT>
void GpuLevelSetSolver<ElemT>::integrateInTime(
    const GpuMatrix<ElemType>& velocities,
    ElemType                              deltaT,
    ETimeDiscretizationOrder              timeOrder)
{
    constexpr unsigned numOfReinitializeIterations{ 10U };

    std::size_t i{};
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
        ++i;

        reinitialize(numOfReinitializeIterations, timeOrder);
    }
    std::cout << "Integration is done in " << i << " iterations" << "\n";
}

template <class ElemT>
void GpuLevelSetSolver<ElemT>::reinitialize(unsigned iterationCount, ETimeDiscretizationOrder timeOrder)
{
  for (unsigned i{ 0U }; i < iterationCount; ++i)
  {
    reinitializeStep(timeOrder);
  }
}

template <class ElemT>
auto GpuLevelSetSolver<ElemT>::integrateInTimeStep(
  const GpuMatrix<ElemType> & velocities,
  ETimeDiscretizationOrder              timeOrder) -> ElemType
{
  const auto courant = static_cast<ElemType>(0.4);
  const auto maxBurningSpeed = getMaxVelocity(velocities.values());
  const auto dt = courant * m_grid.hx * m_grid.hy / (m_grid.hx + m_grid.hy) / maxBurningSpeed;
  return integrateInTimeStep(velocities, timeOrder, dt);
}

template <class ElemT>
auto GpuLevelSetSolver<ElemT>::integrateInTimeStep(
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

template <class ElemT>
void GpuLevelSetSolver<ElemT>::reinitializeStep(ETimeDiscretizationOrder timeOrder)
{
  const auto courant = static_cast<ElemType>(0.8);
  const auto dt = courant * m_grid.hx * m_grid.hy / (m_grid.hx + m_grid.hy);

  thrust::swap(m_prevState.values(), m_currState.values());
  switch (timeOrder)
  {
  case ETimeDiscretizationOrder::eOne:
    detail::reinitializeTVDSubStepWrapper(
      getConstDevicePtr(m_prevState),
      thrust::device_ptr<const ElemType>{},
      getDevicePtr(m_currState), m_shape, m_grid,
      dt, static_cast<ElemType>(1.0));
    break;
  case ETimeDiscretizationOrder::eTwo:
    detail::reinitializeTVDSubStepWrapper(
      getConstDevicePtr(m_prevState),
      thrust::device_ptr<const ElemType>{},
      getDevicePtr(m_firstState), m_shape, m_grid,
      dt, static_cast<ElemType>(1.0));

    detail::reinitializeTVDSubStepWrapper(
      getConstDevicePtr(m_firstState),
      getConstDevicePtr(m_prevState),
      getDevicePtr(m_currState), m_shape, m_grid,
      dt, static_cast<ElemType>(0.5));
    break;
  case ETimeDiscretizationOrder::eThree:
    detail::reinitializeTVDSubStepWrapper(
      getConstDevicePtr(m_prevState),
      thrust::device_ptr<const ElemType>{},
      getDevicePtr(m_firstState), m_shape, m_grid,
      dt, static_cast<ElemType>(1.0));

    detail::reinitializeTVDSubStepWrapper(
      getConstDevicePtr(m_firstState),
      getConstDevicePtr(m_prevState),
      getDevicePtr(m_secondState), m_shape, m_grid,
      dt, static_cast<ElemType>(0.25));

    detail::reinitializeTVDSubStepWrapper(
      getConstDevicePtr(m_secondState),
      getConstDevicePtr(m_prevState),
      getDevicePtr(m_currState), m_shape, m_grid,
      dt, static_cast<ElemType>(2.0 / 3.0));
    break;
  default:
    break;
  }
}

template <class ElemT>
auto GpuLevelSetSolver<ElemT>::getMaxVelocity(const thrust::device_vector<ElemType> & velocities)
  -> ElemType
{
  return thrust::reduce(std::begin(velocities),
                        std::end(velocities), 
                        static_cast<ElemType>(0), 
                        thrust::maximum<ElemType>{});
}

} // namespace kae
