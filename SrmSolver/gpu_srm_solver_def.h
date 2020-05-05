#pragma once

#include "cuda_includes.h"

#include "gas_state.h"
#include "gpu_build_ghost_to_closest_map_kernel.h"
#include "gpu_gas_dynamic_kernel.h"
#include "gpu_matrix_writer.h"
#include "gpu_set_ghost_points_kernel.h"
#include "solver_reduction_functions.h"

namespace kae {

namespace detail {

template <class GpuGridT,
          class ShapeT,
          class PropellantPropertiesT,
          class GasStateT,
          class ElemT = typename GasStateT::ElemType>
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

template <class ShapeT,
          class PropellantPropertiesT,
          class GpuGridT,
          class GasStateT,
          class ElemT = typename GasStateT::ElemType>
GpuMatrix<GpuGridT, ElemT> getBurningRates(const GpuMatrix<GpuGridT, GasStateT>            & currState,
                                           const GpuMatrix<GpuGridT, ElemT>                & currPhi,
                                           const GpuMatrix<GpuGridT, CudaFloatT<2U, ElemT>> & normals)
{
  const auto indices = generateIndexMatrix<unsigned>(GpuGridT::n);
  const auto first   = thrust::make_tuple(std::begin(currState.values()),
                                          std::begin(indices),
                                          std::begin(currPhi.values()),
                                          std::begin(normals.values()));
  const auto last    = thrust::make_tuple(std::end(currState.values()),
                                          std::end(indices),
                                          std::end(currPhi.values()),
                                          std::end(normals.values()));

  const auto zipFirst = thrust::make_zip_iterator(first);
  const auto zipLast = thrust::make_zip_iterator(last);

  const auto toBurningRate = [] __host__ __device__
    (const thrust::tuple<GasStateT, unsigned, ElemT, CudaFloatT<2U, ElemT>> & tuple)
  {
    const auto index = thrust::get<1U>(tuple);
    const auto i = index % GpuGridT::nx;
    const auto j = index / GpuGridT::nx;
    if ((i >= GpuGridT::nx) || (j >= GpuGridT::ny))
    {
      return static_cast<ElemT>(0.0);
    }

    const auto level = thrust::get<2U>(tuple);
    const auto normal = thrust::get<3U>(tuple);
    const auto isBurningSurface = ShapeT::isPointOnGrain(i * GpuGridT::hx - level * normal.x,
                                                         j * GpuGridT::hy - level * normal.y);
    const auto burningRate = BurningRate<PropellantPropertiesT>{}(thrust::get<0U>(tuple));
    return (isBurningSurface ? burningRate : static_cast<ElemT>(0));
  };

  GpuMatrix<GpuGridT, ElemT> burningRates;
  thrust::transform(zipFirst, zipLast, std::begin(burningRates.values()), toBurningRate);
  return burningRates;
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
void GpuSrmSolver<GpuGridT, ShapeT, GasStateT, PropellantPropertiesT>::quasiStationaryDynamicIntegrate(
  unsigned iterationCount, ElemType maximumChamberPressure, ETimeDiscretizationOrder timeOrder, CallbackT callback)
{
  auto t{ static_cast<ElemType>(0.0) };

  ElemType desiredIntegrateTime{};
  ElemType currP{};
  ElemType prevP{};

  const auto levelSetDeltaT = GpuGridT::hx * GpuGridT::hy / (GpuGridT::hx + GpuGridT::hy) /
    BurningRate<PropellantPropertiesType>::get(maximumChamberPressure);

  auto && phiValues = currPhi().values();
  detail::findClosestIndicesWrapper<GpuGridT, ShapeT>(
    getDevicePtr(currPhi()),
    getDevicePtr(m_closestIndices),
    getDevicePtr(m_boundaryConditions),
    getDevicePtr(m_normals));
  for (unsigned i{ 0U }; i < iterationCount; ++i)
  {
    prevP = std::exchange(currP,
      detail::getTheoreticalBoriPressure<GpuGridT, ShapeT, PropellantPropertiesType>(phiValues, m_normals.values()));

    const auto chamberVolume = detail::getChamberVolume<GpuGridT, ShapeT>(phiValues);
    desiredIntegrateTime += 900 * std::fabs(prevP - currP) * chamberVolume + levelSetDeltaT / 50;
    const auto gasDynamicDeltaT = std::min(desiredIntegrateTime, levelSetDeltaT);
    desiredIntegrateTime -= gasDynamicDeltaT;

    staticIntegrate(gasDynamicDeltaT, timeOrder, callback);
    const auto maxDerivatives = getMaxEquationDerivatives();
    callback(m_currState, currPhi(), i, t, maxDerivatives, ShapeT{});
    const auto dt = integrateInTime(levelSetDeltaT);
    t += dt;
  }
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
    callback(m_currState, currPhi(), i, t, maxDerivatives, ShapeT{});
    const auto dt = integrateInTime(deltaTGasDynamic);
    t += dt;
  }
}

template <class GpuGridT, class ShapeT, class GasStateT, class PropellantPropertiesT>
template <class CallbackT>
auto GpuSrmSolver<GpuGridT, ShapeT, GasStateT, PropellantPropertiesT>::staticIntegrate(
  unsigned iterationCount,
  ETimeDiscretizationOrder timeOrder,
  CallbackT callback) -> ElemType
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
    if (i % 100U == 0U)
    {
      callback(m_currState);
    }
    if (i % 5000U == 0U)
    {
      std::cout << i << ": " << t << '\n';
    }
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
    const auto maxDt = m_courant * GpuGridT::hx * GpuGridT::hy / (GpuGridT::hx * lambdas.x + GpuGridT::hy * lambdas.y);
    const auto remainingTime = deltaT - t;
    auto dt = std::min(maxDt, remainingTime);
    dt = staticIntegrateStep(timeOrder, dt, lambdas);
    t += dt;
    ++i;
    if (i % 100U == 0U)
    {
      callback(m_currState);
    }
    if (i % 5000U == 0U)
    {
      std::cout << i << ": " << t << '\n';
    }
  }

  return t;
}

template <class GpuGridT, class ShapeT, class GasStateT, class PropellantPropertiesT>
bool GpuSrmSolver<GpuGridT, ShapeT, GasStateT, PropellantPropertiesT>::isCurrentStateValid() const
{
  return thrust::all_of(std::begin(m_currState.values()), std::end(m_currState.values()), kae::IsValid{});
}

template <class GpuGridT, class ShapeT, class GasStateT, class PropellantPropertiesT>
void GpuSrmSolver<GpuGridT, ShapeT, GasStateT, PropellantPropertiesT>::writeIfNotValid() const
{
  if (isCurrentStateValid())
  {
    return;
  }

  writeMatrixToFile(m_prevState, 
                    "prev_error_p.dat", 
                    "prev_error_ux.dat", 
                    "prev_error_uy.dat", 
                    "prev_error_mach.dat", 
                    "prev_error_t.dat");
  writeMatrixToFile(m_currState, 
                    "curr_error_p.dat", 
                    "curr_error_ux.dat", 
                    "curr_error_uy.dat", 
                    "curr_error_mach.dat", 
                    "curr_error_t.dat");
  writeMatrixToFile(currPhi(), "sgd.dat");
  throw std::runtime_error("Gas state has become invalid");
}

template <class GpuGridT, class ShapeT, class GasStateT, class PropellantPropertiesT>
auto GpuSrmSolver<GpuGridT, ShapeT, GasStateT, PropellantPropertiesT>::staticIntegrateStep(
  ETimeDiscretizationOrder timeOrder) -> ElemType
{
  const auto lambdas = detail::getMaxWaveSpeeds(m_currState.values());
  const auto dt = m_courant * GpuGridT::hx * GpuGridT::hy / (GpuGridT::hx * lambdas.x + GpuGridT::hy * lambdas.y);
  return staticIntegrateStep(timeOrder, dt, lambdas);
}

template <class GpuGridT, class ShapeT, class GasStateT, class PropellantPropertiesT>
auto GpuSrmSolver<GpuGridT, ShapeT, GasStateT, PropellantPropertiesT>::integrateInTime(ElemType deltaT) -> ElemType
{
  const auto burningRates = detail::getBurningRates<ShapeT, PropellantPropertiesT>(
    m_currState, currPhi(), m_normals);
  return m_levelSetSolver.integrateInTime(
    burningRates,
    deltaT,
    ETimeDiscretizationOrder::eThree);
}

template <class GpuGridT, class ShapeT, class GasStateT, class PropellantPropertiesT>
auto GpuSrmSolver<GpuGridT, ShapeT, GasStateT, PropellantPropertiesT>::getMaxEquationDerivatives() const
  -> CudaFloatT<4U, ElemType>
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
