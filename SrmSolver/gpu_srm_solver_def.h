#pragma once

#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

#include "gas_state.h"
#include "gpu_build_ghost_to_closest_map_kernel.h"
#include "gpu_gas_dynamic_kernel.h"
#include "gpu_set_ghost_points_kernel.h"
#include "math_utilities.h"

namespace kae {

namespace detail {

template <class GasStateT>
float2 getMaxWaveSpeeds(const thrust::device_vector<GasStateT> & values)
{
  const auto first = thrust::make_transform_iterator(std::begin(values), kae::WaveSpeedXY{});
  const auto last  = thrust::make_transform_iterator(std::end(values), kae::WaveSpeedXY{});
  return thrust::reduce(first, last, float2{ 0.0f, 0.0f }, kae::ElemwiseMax{});
}

template <class GpuGridT, class GasStateT>
float getDeltaT(const thrust::device_vector<GasStateT> & values, float courant)
{
  float2 lambdas = detail::getMaxWaveSpeeds(values);
  return courant * GpuGridT::hx * GpuGridT::hy / (GpuGridT::hx * lambdas.x + GpuGridT::hy * lambdas.y);
}

template <class GasStateT>
float4 getMaxEquationDerivatives(const thrust::device_vector<GasStateT> & prevValues,
                                 const thrust::device_vector<GasStateT> & currValues,
                                 const thrust::device_vector<float> & currPhi,
                                 float dt)
{
  const auto prevFirst = thrust::make_transform_iterator(std::begin(prevValues), kae::ConservativeVariables{});
  const auto prevLast  = thrust::make_transform_iterator(std::end(prevValues),   kae::ConservativeVariables{});

  const auto currFirst = thrust::make_transform_iterator(std::begin(currValues), kae::ConservativeVariables{});
  const auto currLast  = thrust::make_transform_iterator(std::end(currValues),   kae::ConservativeVariables{});

  const auto zipFirst  = thrust::make_zip_iterator(thrust::make_tuple(prevFirst, currFirst, std::begin(currPhi)));
  const auto zipLast   = thrust::make_zip_iterator(thrust::make_tuple(prevLast,  currLast,  std::end(currPhi)));

  const auto toDerivatives = [dt] __host__ __device__(const thrust::tuple<float4, float4, float> & conservativeVariables)
  {
    const auto prevVariable = thrust::get<0U>(conservativeVariables);
    const auto currVariable = thrust::get<1U>(conservativeVariables);
    const auto level        = thrust::get<2U>(conservativeVariables);
    if (level >= 0.0f)
      return float4{};

    return float4{ (currVariable.x - prevVariable.x) / dt,
                   (currVariable.y - prevVariable.y) / dt,
                   (currVariable.z - prevVariable.z) / dt,
                   (currVariable.w - prevVariable.w) / dt };
  };

  const auto transformFirst = thrust::make_transform_iterator(zipFirst, toDerivatives);
  const auto transformLast  = thrust::make_transform_iterator(zipLast, toDerivatives);
  return thrust::reduce(transformFirst, transformLast, float4{}, kae::ElemwiseAbsMax{});
}

template <class GpuGridT, class ShapeT, class PropellantPropertiesT, class GasStateT>
void srmIntegrateTVDSubStepWrapper(thrust::device_ptr<GasStateT>                pPrevValue,
                                   thrust::device_ptr<const GasStateT>          pFirstValue,
                                   thrust::device_ptr<GasStateT>                pCurrValue,
                                   thrust::device_ptr<const float>              pCurrentPhi,
                                   thrust::device_ptr<const unsigned>           pClosestIndices,
                                   thrust::device_ptr<const EBoundaryCondition> pBoundaryConditions,
                                   thrust::device_ptr<float2>                   pNormals,
                                   float dt, float2 lambda, float prevWeight)
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
  float     courant)
  : m_boundaryConditions{ EBoundaryCondition::eWall                               },
    m_closestIndices    { 0U                                                      },
    m_normals           { float2{ 0.0f, 0.0f }                                    },
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
  unsigned iterationCount, ETimeDiscretizationOrder timeOrder, CallbackT callback)
{
  float t{ 0.0f };
  callback(m_currState, currPhi(), 0U, t, float4{});
  for (unsigned i{ 1U }; i <= iterationCount; ++i)
  {
    const float deltaTGasDynamic = staticIntegrate(80000U, timeOrder);
    const float dt = m_levelSetSolver.template integrateInTime<PropellantPropertiesT>(
      m_currState,
      m_closestIndices,
      2U,
      ETimeDiscretizationOrder::eThree);

    t += dt;

    const auto maxDerivatives = detail::getMaxEquationDerivatives(
      m_prevState.values(),
      m_currState.values(),
      currPhi().values(),
      detail::getDeltaT<GpuGridT>(m_currState.values(), m_courant));
    callback(m_currState, currPhi(), i, t, maxDerivatives);
  }
}

template <class GpuGridT, class ShapeT, class GasStateT, class PropellantPropertiesT>
float GpuSrmSolver<GpuGridT, ShapeT, GasStateT, PropellantPropertiesT>::staticIntegrate(
  unsigned iterationCount,
  ETimeDiscretizationOrder timeOrder)
{
  detail::findClosestIndicesWrapper<GpuGridT, ShapeT>(
    getDevicePtr(currPhi()),
    getDevicePtr(m_closestIndices),
    getDevicePtr(m_boundaryConditions),
    getDevicePtr(m_normals));

  float t = 0.0f;
  for (unsigned i{ 0U }; i < iterationCount; ++i)
  {
    const float dt = staticIntegrateStep(timeOrder);
    t += dt;
  }

  return t;
}

template <class GpuGridT, class ShapeT, class GasStateT, class PropellantPropertiesT>
float GpuSrmSolver<GpuGridT, ShapeT, GasStateT, PropellantPropertiesT>::staticIntegrateStep(ETimeDiscretizationOrder timeOrder)
{
  float2 lambdas = detail::getMaxWaveSpeeds(m_currState.values());
  float dt = m_courant * GpuGridT::hx * GpuGridT::hy / (GpuGridT::hx * lambdas.x + GpuGridT::hy * lambdas.y);

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
      dt, lambdas, 1.0f);
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
      dt, lambdas, 1.0f);

    detail::srmIntegrateTVDSubStepWrapper<GpuGridT, ShapeT, PropellantPropertiesT, GasStateT>(
      getDevicePtr(m_firstState),
      getDevicePtr(m_prevState),
      getDevicePtr(m_currState),
      getDevicePtr(currPhi()),
      getDevicePtr(m_closestIndices),
      getDevicePtr(m_boundaryConditions),
      getDevicePtr(m_normals),
      dt, lambdas, 0.5f);
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
      dt, lambdas, 1.0f);

    detail::srmIntegrateTVDSubStepWrapper<GpuGridT, ShapeT, PropellantPropertiesT, GasStateT>(
      getDevicePtr(m_firstState),
      getDevicePtr(m_prevState),
      getDevicePtr(m_secondState),
      getDevicePtr(currPhi()),
      getDevicePtr(m_closestIndices),
      getDevicePtr(m_boundaryConditions),
      getDevicePtr(m_normals),
      dt, lambdas, 0.25f);

    detail::srmIntegrateTVDSubStepWrapper<GpuGridT, ShapeT, PropellantPropertiesT, GasStateT>(
      getDevicePtr(m_secondState),
      getDevicePtr(m_prevState),
      getDevicePtr(m_currState),
      getDevicePtr(currPhi()),
      getDevicePtr(m_closestIndices),
      getDevicePtr(m_boundaryConditions),
      getDevicePtr(m_normals),
      dt, lambdas, 2.0f / 3.0f);
    break;
  default:
    break;
  }

  return dt;
}

} // namespace 
