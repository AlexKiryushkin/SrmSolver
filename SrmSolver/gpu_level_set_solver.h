#pragma once

#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

#include "discretization_order.h"
#include "gpu_integrate_kernel.h"
#include "gpu_matrix.h"
#include "gpu_reinitialize_kernel.h"
#include "propellant_properties.h"

namespace kae {

template <class GpuGridT, class ShapeT>
class GpuLevelSetSolver
{
public:

  explicit GpuLevelSetSolver(ShapeT shape,
                             unsigned iterationCount = 0,
                             ETimeDiscretizationOrder timeOrder = ETimeDiscretizationOrder::eThree);

  template <class PropellantPropertiesT, class GasStateT>
  float integrateInTime(const GpuMatrix<GpuGridT, GasStateT> & gasValues,
                        const GpuMatrix<GpuGridT, unsigned>  & closestIndices,
                        unsigned                               iterationCount,
                        ETimeDiscretizationOrder               timeOrder = ETimeDiscretizationOrder::eThree);

  void reinitialize(unsigned iterationCount, ETimeDiscretizationOrder timeOrder = ETimeDiscretizationOrder::eThree);

  const GpuMatrix<GpuGridT, float> & currState() const { return m_currState; }

private:

  template <class PropellantPropertiesT, class GasStateT>
  float integrateInTimeStep(const GpuMatrix<GpuGridT, GasStateT> & gasValues,
                            const GpuMatrix<GpuGridT, unsigned>  & closestIndices,
                            ETimeDiscretizationOrder               timeOrder);

  void reinitializeStep(ETimeDiscretizationOrder timeOrder);

  template <class PropellantPropertiesT, class GasStateT>
  float getMaxBurningRate(const thrust::device_vector<GasStateT> & gasValues);

private:
  GpuMatrix<GpuGridT, float> m_currState;
  GpuMatrix<GpuGridT, float> m_prevState;
  GpuMatrix<GpuGridT, float> m_firstState;
  GpuMatrix<GpuGridT, float> m_secondState;
};

template <class GpuGridT, class ShapeT>
GpuLevelSetSolver<GpuGridT, ShapeT>::GpuLevelSetSolver(ShapeT shape, unsigned iterationCount, ETimeDiscretizationOrder timeOrder)
  : m_currState(shape), m_prevState(shape), m_firstState(shape), m_secondState(shape)
{
  reinitialize(iterationCount, timeOrder);
}

template <class GpuGridT, class ShapeT>
template <class PropellantPropertiesT, class GasStateT>
float GpuLevelSetSolver<GpuGridT, ShapeT>::integrateInTime(
  const GpuMatrix<GpuGridT, GasStateT> & gasValues,
  const GpuMatrix<GpuGridT, unsigned>  & closestIndices,
  unsigned                               iterationCount,
  ETimeDiscretizationOrder               timeOrder)
{
  float t{ 0.0f };
  const unsigned numOfReinitializeIterations{ 10U };
  for (unsigned i{ 0U }; i < iterationCount; ++i)
  {
    const float dt = integrateInTimeStep<PropellantPropertiesT>(gasValues, closestIndices, timeOrder);
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
float GpuLevelSetSolver<GpuGridT, ShapeT>::integrateInTimeStep(
  const GpuMatrix<GpuGridT, GasStateT> & gasValues,
  const GpuMatrix<GpuGridT, unsigned>  & closestIndices,
  ETimeDiscretizationOrder               timeOrder)
{
  const float maxBurningSpeed = getMaxBurningRate<PropellantPropertiesT>(gasValues.values());

  const float courant = 0.4f;
  const float dt = courant * GpuGridT::hx * GpuGridT::hy / (GpuGridT::hx + GpuGridT::hy) / maxBurningSpeed;

  thrust::swap(m_prevState.values(), m_currState.values());
  switch (timeOrder)
  {
  case ETimeDiscretizationOrder::eOne:
    detail::integrateEqTvdSubStepWrapper<GpuGridT, ShapeT, PropellantPropertiesT>(
      getDevicePtr(m_prevState),
      {},
      getDevicePtr(m_currState),
      getDevicePtr(gasValues),
      getDevicePtr(closestIndices),
      dt, 1.0f);
    break;

  case ETimeDiscretizationOrder::eTwo:
    detail::integrateEqTvdSubStepWrapper<GpuGridT, ShapeT, PropellantPropertiesT>(
      getDevicePtr(m_prevState),
      {},
      getDevicePtr(m_firstState),
      getDevicePtr(gasValues),
      getDevicePtr(closestIndices),
      dt, 1.0f);

    detail::integrateEqTvdSubStepWrapper<GpuGridT, ShapeT, PropellantPropertiesT>(
      getDevicePtr(m_firstState),
      getDevicePtr(m_prevState),
      getDevicePtr(m_currState),
      getDevicePtr(gasValues),
      getDevicePtr(closestIndices),
      dt, 0.5f);
    break;
  case ETimeDiscretizationOrder::eThree:
    detail::integrateEqTvdSubStepWrapper<GpuGridT, ShapeT, PropellantPropertiesT>(
      getDevicePtr(m_prevState),
      {},
      getDevicePtr(m_firstState),
      getDevicePtr(gasValues),
      getDevicePtr(closestIndices),
      dt, 1.0f);

    detail::integrateEqTvdSubStepWrapper<GpuGridT, ShapeT, PropellantPropertiesT>(
      getDevicePtr(m_firstState),
      getDevicePtr(m_prevState),
      getDevicePtr(m_secondState),
      getDevicePtr(gasValues),
      getDevicePtr(closestIndices),
      dt, 0.25f);

    detail::integrateEqTvdSubStepWrapper<GpuGridT, ShapeT, PropellantPropertiesT>(
      getDevicePtr(m_secondState),
      getDevicePtr(m_prevState),
      getDevicePtr(m_currState),
      getDevicePtr(gasValues),
      getDevicePtr(closestIndices),
      dt, 2.0f / 3.0f);
    break;
  default:
    break;
  }

  return dt;
}

template <class GpuGridT, class ShapeT>
void GpuLevelSetSolver<GpuGridT, ShapeT>::reinitializeStep(ETimeDiscretizationOrder timeOrder)
{
  const float courant = 0.8f;
  const float dt = courant * GpuGridT::hx * GpuGridT::hy / (GpuGridT::hx + GpuGridT::hy);

  thrust::swap(m_prevState.values(), m_currState.values());
  switch (timeOrder)
  {
  case ETimeDiscretizationOrder::eOne:
    detail::reinitializeTVDSubStepWrapper<GpuGridT, ShapeT>(
      getDevicePtr(m_prevState),
      {},
      getDevicePtr(m_currState),
      dt, 1.0f);
    break;
  case ETimeDiscretizationOrder::eTwo:
    detail::reinitializeTVDSubStepWrapper<GpuGridT, ShapeT>(
      getDevicePtr(m_prevState),
      {},
      getDevicePtr(m_firstState),
      dt, 1.0f);

    detail::reinitializeTVDSubStepWrapper<GpuGridT, ShapeT>(
      getDevicePtr(m_firstState),
      getDevicePtr(m_prevState),
      getDevicePtr(m_currState),
      dt, 0.5f);
    break;
  case ETimeDiscretizationOrder::eThree:
    detail::reinitializeTVDSubStepWrapper<GpuGridT, ShapeT>(
      getDevicePtr(m_prevState),
      {},
      getDevicePtr(m_firstState),
      dt, 1.0f);

    detail::reinitializeTVDSubStepWrapper<GpuGridT, ShapeT>(
      getDevicePtr(m_firstState),
      getDevicePtr(m_prevState),
      getDevicePtr(m_secondState),
      dt, 0.25f);

    detail::reinitializeTVDSubStepWrapper<GpuGridT, ShapeT>(
      getDevicePtr(m_secondState),
      getDevicePtr(m_prevState),
      getDevicePtr(m_currState),
      dt, 2.0f / 3.0f);
    break;
  default:
    break;
  }
}

template <class GpuGridT, class ShapeT>
template <class PropellantPropertiesT, class GasStateT>
float GpuLevelSetSolver<GpuGridT, ShapeT>::getMaxBurningRate(const thrust::device_vector<GasStateT> & gasValues)
{
  auto first = thrust::make_transform_iterator(std::begin(gasValues), kae::BurningRate<PropellantPropertiesT>{});
  auto last  = thrust::make_transform_iterator(std::end(gasValues),   kae::BurningRate<PropellantPropertiesT>{});
  return thrust::reduce(first, last, 0.0f, thrust::maximum<float>{});
}

} // namespace kae
