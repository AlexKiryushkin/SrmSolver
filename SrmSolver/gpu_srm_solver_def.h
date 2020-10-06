#pragma once

#include "std_includes.h"
#include "cuda_includes.h"

#include "gas_state.h"
#include "gpu_build_ghost_to_closest_map_kernel.h"
#include "gpu_calculate_ghost_point_data_kernel.h"
#include "gpu_gas_dynamic_kernel.h"
#include "gpu_matrix_writer.h"
#include "gpu_set_first_order_ghost_points_kernel.h"
#include "gpu_set_ghost_points_kernel.h"
#include "solver_reduction_functions.h"

template <class T>
using DevicePtr = thrust::device_ptr<T>;

namespace kae {

namespace detail {

template <class PhysicalPropertiesT, class ShapeT, class ElemT = typename ShapeT::ElemType>
ElemT getDeltaT(ElemT prevP, ElemT currP, ElemT sBurn, ElemT chamberVolume)
{
  constexpr auto rt = (PhysicalPropertiesT::kappa - 1) / PhysicalPropertiesT::kappa * PhysicalPropertiesT::H0;
  const ElemT a = -PhysicalPropertiesT::mt * sBurn * rt / chamberVolume;
  const ElemT b = ShapeT::getFCritical() * std::sqrt(rt) * PhysicalPropertiesT::gammaComplex / chamberVolume;
  return 1 / (1 - PhysicalPropertiesT::nu) / b * std::log(
    (std::pow(prevP, 1 - PhysicalPropertiesT::nu) - a / b) /
    (std::pow(currP, 1 - PhysicalPropertiesT::nu) - a / b)
  );
}

template <class GpuGridT,
          class ShapeT,
          class PhysicalPropertiesT,
          unsigned order,
          class GasStateT,
          class IndexMatrixT,
          class ElemT = typename GpuGridT::ElemType>
void srmIntegrateTVDSubStepWrapper(DevicePtr<GasStateT>                              pPrevValue,
                                   DevicePtr<const GasStateT>                        pFirstValue,
                                   DevicePtr<GasStateT>                              pCurrValue,
                                   DevicePtr<const ElemT>                            pCurrentPhi,
                                   DevicePtr<const thrust::pair<unsigned, unsigned>> pClosestIndicesMap,
                                   DevicePtr<const EBoundaryCondition>               pBoundaryConditions,
                                   DevicePtr<CudaFloat2T<ElemT>>                     pNormals,
                                   DevicePtr<CudaFloat2T<ElemT>>                     pSurfacePoints,
                                   DevicePtr<IndexMatrixT>                           pIndexMatrices,
                                   unsigned nClosestIndexElems, ElemT dt, CudaFloat2T<ElemT> lambda, ElemT prevWeight)
{
  constexpr std::uint64_t startIdx{ 200000U };
  static thread_local std::uint64_t counter{};

  /*if (counter > startIdx)
  {
    detail::setGhostValuesWrapper<GpuGridT, GasStateT, PhysicalPropertiesT, order>(
      pPrevValue,
      pClosestIndicesMap,
      pBoundaryConditions,
      pNormals,
      pSurfacePoints,
      pIndexMatrices,
      nClosestIndexElems);
  }
  else*/
  {
    //++counter;
    detail::setFirstOrderGhostValuesWrapper<GpuGridT, GasStateT, PhysicalPropertiesT>(
      pPrevValue,
      pCurrentPhi,
      pClosestIndicesMap,
      pBoundaryConditions,
      pNormals,
      nClosestIndexElems);
  }


  detail::gasDynamicIntegrateTVDSubStepWrapper<GpuGridT, ShapeT, GasStateT>(
    pPrevValue,
    pFirstValue,
    pCurrValue,
    pCurrentPhi,
    dt, lambda, prevWeight);
}

template <class ShapeT,
          class PhysicalPropertiesT,
          class GpuGridT,
          class GasStateT,
          class ElemT = typename GasStateT::ElemType>
GpuMatrix<GpuGridT, ElemT> getBurningRates(const GpuMatrix<GpuGridT, GasStateT>            & currState,
                                           const GpuMatrix<GpuGridT, ElemT>                & currPhi,
                                           const GpuMatrix<GpuGridT, CudaFloat2T<ElemT>>   & normals)
{
  const static thread_local auto indices = generateIndexMatrix<unsigned>(GpuGridT::n);

  const auto zipFirst = thrust::make_zip_iterator(
    thrust::make_tuple(std::begin(currState.values()), 
                       std::begin(indices), 
                       std::begin(currPhi.values()), 
                       std::begin(normals.values())));
  const auto zipLast = thrust::make_zip_iterator(
    thrust::make_tuple(std::end(currState.values()), 
                       std::end(indices), 
                       std::end(currPhi.values()), 
                       std::end(normals.values())));

  const auto toBurningRate = [] __device__
    (const thrust::tuple<GasStateT, unsigned, ElemT, CudaFloat2T<ElemT>> & tuple)
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
    const auto burningRate = BurningRate<PhysicalPropertiesT>{}(thrust::get<0U>(tuple));
    return (isBurningSurface ? burningRate : static_cast<ElemT>(0));
  };

  GpuMatrix<GpuGridT, ElemT> burningRates;
  thrust::transform(zipFirst, zipLast, std::begin(burningRates.values()), toBurningRate);
  return burningRates;
}

} // namespace detail

template <class GpuGridT, class ShapeT, class GasStateT, class PhysicalPropertiesT>
GpuSrmSolver<GpuGridT, ShapeT, GasStateT, PhysicalPropertiesT>::GpuSrmSolver(
  ShapeT    shape, 
  GasStateT initialState,
  unsigned  iterationCount,
  ElemType  courant)
  : m_boundaryConditions{ EBoundaryCondition::eWall                               },
    m_normals           { CudaFloat2T<ElemType>{ 0, 0 }                           },
    m_surfacePoints     { CudaFloat2T<ElemType>{ 0, 0 }                           },
    m_indexMatrices     { IndexMatrixT{}                                          },
    m_currState         { initialState                                            },
    m_prevState         { initialState                                            },
    m_firstState        { initialState                                            },
    m_secondState       { initialState                                            },
    m_levelSetSolver    { shape, iterationCount, ETimeDiscretizationOrder::eThree },
    m_closestIndicesMap ( GpuGridT::n, thrust::make_pair(0U, 0U)                  ),
    m_calculateBlocks   ( maxSizeX * maxSizeY, 0                                  ),
    m_courant           { courant                                                 }
{
  findClosestIndices();
}

template <class GpuGridT, class ShapeT, class GasStateT, class PhysicalPropertiesT>
template <class CallbackT>
void GpuSrmSolver<GpuGridT, ShapeT, GasStateT, PhysicalPropertiesT>::quasiStationaryDynamicIntegrate(
  unsigned iterationCount, ElemType levelSetDeltaT, ETimeDiscretizationOrder timeOrder, CallbackT && callback)
{
  auto t{ static_cast<ElemType>(0.0) };

  ElemType desiredIntegrateTime{};
  ElemType currP{};
  ElemType prevP{};

  auto && phiValues = currPhi().values();
  for (unsigned i{ 0U }; i < iterationCount; ++i)
  {
    prevP = std::exchange(currP,
      detail::getTheoreticalBoriPressure<GpuGridT, ShapeT, PhysicalPropertiesT>(phiValues, m_normals.values()));
    const auto sBurn = detail::getBurningSurface<GpuGridT, ShapeT>(phiValues, m_normals.values());
    const auto chamberVolume = detail::getChamberVolume<GpuGridT, ShapeT>(phiValues);
    desiredIntegrateTime += 900 * std::fabs(prevP - currP) * chamberVolume + levelSetDeltaT / 50;
    const auto gasDynamicDeltaT = std::min(desiredIntegrateTime, levelSetDeltaT);
    desiredIntegrateTime -= gasDynamicDeltaT;

    staticIntegrate(gasDynamicDeltaT, timeOrder, callback);
    if (i % 100 == 0)
    {
      callback(m_currState, currPhi(), i, t, getMaxEquationDerivatives(), sBurn, ShapeT{});
    }
    const auto dt = integrateInTime(levelSetDeltaT);
    t += dt;
  }
}

template <class GpuGridT, class ShapeT, class GasStateT, class PhysicalPropertiesT>
template <class CallbackT>
void GpuSrmSolver<GpuGridT, ShapeT, GasStateT, PhysicalPropertiesT>::dynamicIntegrate(
  unsigned iterationCount, ElemType deltaT, ETimeDiscretizationOrder timeOrder, CallbackT && callback)
{
  auto t{ static_cast<ElemType>(0.0) };
  for (unsigned i{ 0U }; i < iterationCount; ++i)
  {
    const auto deltaTGasDynamic = staticIntegrate(deltaT, timeOrder, callback);
    const auto maxDerivatives = getMaxEquationDerivatives();
    const auto sBurn = detail::getBurningSurface<GpuGridT, ShapeT>(currPhi().values(), m_normals.values());
    if (i % 10 == 0)
    {
      callback(m_currState, currPhi(), i, t, getMaxEquationDerivatives(), sBurn, ShapeT{});
    }
    const auto dt = integrateInTime(deltaTGasDynamic);
    t += dt;
  }
}

template <class GpuGridT, class ShapeT, class GasStateT, class PhysicalPropertiesT>
template <class CallbackT>
auto GpuSrmSolver<GpuGridT, ShapeT, GasStateT, PhysicalPropertiesT>::staticIntegrate(
  unsigned iterationCount,
  ETimeDiscretizationOrder timeOrder,
  CallbackT && callback) -> ElemType
{
  findClosestIndices();

  auto t{ static_cast<ElemType>(0.0) };
  CudaFloat2T<ElemType> lambdas{};
  for (unsigned i{ 0U }; i < iterationCount; ++i)
  {
    if ((i <= 1000U) || (i % 500U == 0U))
    {
      lambdas = detail::getMaxWaveSpeeds(m_currState.values());
    }

    const auto dt = m_courant * GpuGridT::hx * GpuGridT::hy / (GpuGridT::hx * lambdas.x + GpuGridT::hy * lambdas.y);
    staticIntegrateStep(timeOrder, dt, lambdas);
    t += dt;

    if (i % 200U == 0U)
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

template <class GpuGridT, class ShapeT, class GasStateT, class PhysicalPropertiesT>
template <class CallbackT>
auto GpuSrmSolver<GpuGridT, ShapeT, GasStateT, PhysicalPropertiesT>::staticIntegrate(
  ElemType deltaT,
  ETimeDiscretizationOrder timeOrder, 
  CallbackT && callback) -> ElemType
{
  findClosestIndices();

  unsigned i{ 0U };
  auto t{ static_cast<ElemType>(0.0) };
  CudaFloat2T<ElemType> lambdas;
  while (t < deltaT)
  {
    if ((i <= 1000U) || (i % 500U == 0U))
    {
      lambdas = detail::getMaxWaveSpeeds(m_currState.values());
    }

    const auto maxDt = m_courant * GpuGridT::hx * GpuGridT::hy / (GpuGridT::hx * lambdas.x + GpuGridT::hy * lambdas.y);
    const auto remainingTime = deltaT - t;
    auto dt = std::min(maxDt, remainingTime);
    dt = staticIntegrateStep(timeOrder, dt, lambdas);
    t += dt;
    ++i;
    if (i % 200U == 0U)
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

template <class GpuGridT, class ShapeT, class GasStateT, class PhysicalPropertiesT>
void GpuSrmSolver<GpuGridT, ShapeT, GasStateT, PhysicalPropertiesT>::findClosestIndices()
{
  m_closestIndicesMap.resize(GpuGridT::n);
  thrust::fill(std::begin(m_closestIndicesMap), 
               std::end(m_closestIndicesMap), 
               thrust::pair<unsigned, unsigned>{0U, 0U});
  detail::calculateGhostPointDataWrapper<GpuGridT, ShapeT, order>(
    getDevicePtr(currPhi()),
    m_closestIndicesMap.data(),
    getDevicePtr(m_boundaryConditions),
    getDevicePtr(m_normals),
    getDevicePtr(m_surfacePoints),
    getDevicePtr(m_indexMatrices));

  const auto removeIter = thrust::remove_if(std::begin(m_closestIndicesMap), 
                                            std::end(m_closestIndicesMap),
                                            thrust::placeholders::_1 == thrust::pair<unsigned, unsigned>{0U, 0U});
  m_closestIndicesMap.erase(removeIter, std::end(m_closestIndicesMap));
  fillCalculateBlockMatrix();
}

template <class GpuGridT, class ShapeT, class GasStateT, class PhysicalPropertiesT>
void GpuSrmSolver<GpuGridT, ShapeT, GasStateT, PhysicalPropertiesT>::fillCalculateBlockMatrix()
{
  static_assert(maxSizeX >= GpuGridT::gridSize.x, "Error. Max size is too small!");
  static_assert(maxSizeY >= GpuGridT::gridSize.y, "Error. Max size is too small!");

  thrust::host_vector<ElemType> currentLevel = currPhi().values();

  for (unsigned blockIdxX{ 0U }; blockIdxX < GpuGridT::gridSize.x; ++blockIdxX)
  {
    const auto startX = blockIdxX * GpuGridT::blockSize.x;
    for (unsigned blockIdxY{ 0U }; blockIdxY < GpuGridT::gridSize.y; ++blockIdxY)
    {
      const auto startY = blockIdxY * GpuGridT::blockSize.y;
      bool calculateBlock{ false };
      for (unsigned xIdx{ 0U }; (xIdx < GpuGridT::blockSize.x) && !calculateBlock; ++xIdx)
      {
        for (unsigned yIdx{ 0U }; (yIdx < GpuGridT::blockSize.y) && !calculateBlock; ++yIdx)
        {
          const auto index = (startY + yIdx) * GpuGridT::nx + startX + xIdx;
          if ((index < GpuGridT::n) && (currentLevel[index] < 0))
          {
            calculateBlock = true;
          }
        }
      }
      if (calculateBlock)
      {
        m_calculateBlocks.at(blockIdxY * maxSizeX + blockIdxX) = 1;
      }
    }
  }

  cudaMemcpyToSymbol(calculateBlockMatrix, m_calculateBlocks.data(), sizeof(int8_t) * maxSizeX * maxSizeY);
}

template <class GpuGridT, class ShapeT, class GasStateT, class PhysicalPropertiesT>
void GpuSrmSolver<GpuGridT, ShapeT, GasStateT, PhysicalPropertiesT>::writeIfNotValid() const
{
  if (thrust::all_of(std::begin(m_currState.values()), std::end(m_currState.values()), kae::IsValid{}))
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

template <class GpuGridT, class ShapeT, class GasStateT, class PhysicalPropertiesT>
auto GpuSrmSolver<GpuGridT, ShapeT, GasStateT, PhysicalPropertiesT>::integrateInTime(ElemType deltaT) -> ElemType
{
  const auto burningRates = detail::getBurningRates<ShapeT, PhysicalPropertiesT>(
    m_currState, currPhi(), m_normals);
  return m_levelSetSolver.integrateInTime(
    burningRates, deltaT, ETimeDiscretizationOrder::eThree);
}

template <class GpuGridT, class ShapeT, class GasStateT, class PhysicalPropertiesT>
auto GpuSrmSolver<GpuGridT, ShapeT, GasStateT, PhysicalPropertiesT>::getMaxEquationDerivatives() const
  -> CudaFloat4T<ElemType>
{
  return detail::getMaxEquationDerivatives(
    m_prevState.values(),
    m_currState.values(),
    detail::getDeltaT<GpuGridT>(m_currState.values(), m_courant));
}

template <class GpuGridT, class ShapeT, class GasStateT, class PhysicalPropertiesT>
auto GpuSrmSolver<GpuGridT, ShapeT, GasStateT, PhysicalPropertiesT>::staticIntegrateStep(
  ETimeDiscretizationOrder timeOrder,
  ElemType dt,
  CudaFloat2T<ElemType> lambdas) -> ElemType
{
  thrust::swap(m_prevState.values(), m_currState.values());
  switch (timeOrder)
  {
  case ETimeDiscretizationOrder::eOne:
    detail::srmIntegrateTVDSubStepWrapper<GpuGridT, ShapeT, PhysicalPropertiesT, order, GasStateT>(
      getDevicePtr(m_prevState),
      {},
      getDevicePtr(m_currState),
      getDevicePtr(currPhi()),
      m_closestIndicesMap.data(),
      getDevicePtr(m_boundaryConditions),
      getDevicePtr(m_normals),
      getDevicePtr(m_surfacePoints),
      getDevicePtr(m_indexMatrices),
      static_cast<unsigned>(m_closestIndicesMap.size()), dt, lambdas, static_cast<ElemType>(1.0));
    break;

  case ETimeDiscretizationOrder::eTwo:
    detail::srmIntegrateTVDSubStepWrapper<GpuGridT, ShapeT, PhysicalPropertiesT, order, GasStateT>(
      getDevicePtr(m_prevState),
      {},
      getDevicePtr(m_firstState),
      getDevicePtr(currPhi()),
      m_closestIndicesMap.data(),
      getDevicePtr(m_boundaryConditions),
      getDevicePtr(m_normals),
      getDevicePtr(m_surfacePoints),
      getDevicePtr(m_indexMatrices),
      static_cast<unsigned>(m_closestIndicesMap.size()), dt, lambdas, static_cast<ElemType>(1.0));

    detail::srmIntegrateTVDSubStepWrapper<GpuGridT, ShapeT, PhysicalPropertiesT, order, GasStateT>(
      getDevicePtr(m_firstState),
      getDevicePtr(m_prevState),
      getDevicePtr(m_currState),
      getDevicePtr(currPhi()),
      m_closestIndicesMap.data(),
      getDevicePtr(m_boundaryConditions),
      getDevicePtr(m_normals),
      getDevicePtr(m_surfacePoints),
      getDevicePtr(m_indexMatrices),
      static_cast<unsigned>(m_closestIndicesMap.size()), dt, lambdas, static_cast<ElemType>(0.5));
    break;
  case ETimeDiscretizationOrder::eThree:
    detail::srmIntegrateTVDSubStepWrapper<GpuGridT, ShapeT, PhysicalPropertiesT, order, GasStateT>(
      getDevicePtr(m_prevState),
      {},
      getDevicePtr(m_firstState),
      getDevicePtr(currPhi()),
      m_closestIndicesMap.data(),
      getDevicePtr(m_boundaryConditions),
      getDevicePtr(m_normals),
      getDevicePtr(m_surfacePoints),
      getDevicePtr(m_indexMatrices),
      static_cast<unsigned>(m_closestIndicesMap.size()), dt, lambdas, static_cast<ElemType>(1.0));

    detail::srmIntegrateTVDSubStepWrapper<GpuGridT, ShapeT, PhysicalPropertiesT, order, GasStateT>(
      getDevicePtr(m_firstState),
      getDevicePtr(m_prevState),
      getDevicePtr(m_secondState),
      getDevicePtr(currPhi()),
      m_closestIndicesMap.data(),
      getDevicePtr(m_boundaryConditions),
      getDevicePtr(m_normals),
      getDevicePtr(m_surfacePoints),
      getDevicePtr(m_indexMatrices),
      static_cast<unsigned>(m_closestIndicesMap.size()), dt, lambdas, static_cast<ElemType>(0.25));

    detail::srmIntegrateTVDSubStepWrapper<GpuGridT, ShapeT, PhysicalPropertiesT, order, GasStateT>(
      getDevicePtr(m_secondState),
      getDevicePtr(m_prevState),
      getDevicePtr(m_currState),
      getDevicePtr(currPhi()),
      m_closestIndicesMap.data(),
      getDevicePtr(m_boundaryConditions),
      getDevicePtr(m_normals),
      getDevicePtr(m_surfacePoints),
      getDevicePtr(m_indexMatrices),
      static_cast<unsigned>(m_closestIndicesMap.size()), dt, lambdas, static_cast<ElemType>(2.0 / 3.0));
    break;
  default:
    break;
  }

  return dt;
}

} // namespace 
