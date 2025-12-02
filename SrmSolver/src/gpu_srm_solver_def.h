#pragma once

#include "std_includes.h"
#include "cuda_includes.h"

#include "gas_state.h"
#include "kernel/gpu_build_ghost_to_closest_map_kernel.cuh"
#include "kernel/gpu_calculate_ghost_point_data_kernel.h"
#include "kernel/gpu_gas_dynamic_kernel.h"
#include "gpu_matrix_writer.h"
#include "kernel/gpu_set_first_order_ghost_points_kernel.h"
#include "kernel/gpu_set_ghost_points_kernel.h"
#include "solver_reduction_functions.h"

template <class T>
using DevicePtr = thrust::device_ptr<T>;

namespace kae {

namespace detail {

template <class ShapeT, class ElemT = typename ShapeT::ElemType>
ElemT getDeltaT(ElemT prevP, ElemT currP, ElemT sBurn, ElemT chamberVolume, PhysicalPropertiesData<ElemT> physicalProperties)
{
  constexpr auto rt = (physicalProperties.kappa - 1) / physicalProperties.kappa * physicalProperties.H0;
  const ElemT a = -physicalProperties.mt * sBurn * rt / chamberVolume;
  const ElemT b = ShapeT::getFCritical() * std::sqrt(rt) * physicalProperties.gammaComplex / chamberVolume;
  return 1 / (1 - physicalProperties.nu) / b * std::log(
    (std::pow(prevP, 1 - physicalProperties.nu) - a / b) /
    (std::pow(currP, 1 - physicalProperties.nu) - a / b)
  );
}

template <class ShapeT,
          unsigned order,
          class GasStateT,
          class IndexMatrixT,
          class ElemT = typename GasStateT::ElemType>
void srmIntegrateTVDSubStepWrapper(DevicePtr<GasStateT>                              pPrevValue,
                                   DevicePtr<const GasStateT>                        pFirstValue,
                                   DevicePtr<GasStateT>                              pCurrValue,
                                   DevicePtr<const ElemT>                            pCurrentPhi, GasParameters<ElemT> gasParameters,
                                   DevicePtr<const thrust::pair<unsigned, unsigned>> pClosestIndicesMap,
                                   DevicePtr<const EBoundaryCondition>               pBoundaryConditions,
                                   DevicePtr<CudaFloat2T<ElemT>>                     pNormals,
                                   DevicePtr<CudaFloat2T<ElemT>>                     pSurfacePoints,
                                   DevicePtr<IndexMatrixT>                           pIndexMatrices,
                                   GpuGridT<ElemT> grid, PhysicalPropertiesData<ElemT> physicalProperties,
                                   unsigned nClosestIndexElems, ElemT dt, CudaFloat2T<ElemT> lambda, ElemT prevWeight)
{
  // constexpr std::uint64_t startIdx{ 200U };
  static thread_local std::uint64_t counter{};

  /*if (counter > startIdx)
  {
    detail::setGhostValuesWrapper<GpuGridT, ShapeT, GasStateT, PhysicalPropertiesT, order>(
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
    ++counter;
    detail::setFirstOrderGhostValuesWrapper<GasStateT>(
      pPrevValue,
      pCurrentPhi,
      pClosestIndicesMap,
      pBoundaryConditions,
      pNormals, gasParameters, physicalProperties,
      nClosestIndexElems);
  }


  detail::gasDynamicIntegrateTVDSubStepWrapper<ShapeT, GasStateT>(
    pPrevValue,
    pFirstValue,
    pCurrValue,
    pCurrentPhi, gasParameters, grid,
    dt, lambda, prevWeight);
}

template <class ShapeT,
          class GasStateT,
          class ElemT = typename GasStateT::ElemType>
GpuMatrix<ElemT> getBurningRates(const GpuMatrix<GasStateT>            & currState,
                                           const GpuMatrix<ElemT>                & currPhi,
                                           const GpuMatrix<CudaFloat2T<ElemT>>   & normals, PhysicalPropertiesData<ElemT> physicalProperties, unsigned nx, unsigned ny, ElemT hx, ElemT hy)
{
  const static thread_local auto indices = generateIndexMatrix<unsigned>(nx * ny);

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

  const auto toBurningRate = [physicalProperties, nx, ny, hx, hy] __device__
    (const thrust::tuple<GasStateT, unsigned, ElemT, CudaFloat2T<ElemT>> & tuple)
  {
    const auto index = thrust::get<1U>(tuple);
    const auto i = index % nx;
    const auto j = index / nx;
    if ((i >= nx) || (j >= ny))
    {
      return static_cast<ElemT>(0.0);
    }

    const auto level = thrust::get<2U>(tuple);
    const auto normal = thrust::get<3U>(tuple);
    const auto isBurningSurface = ShapeT::isPointOnGrain(i * hx - level * normal.x,
                                                         j * hy - level * normal.y, hx);
    const auto burningRate = BurningRate{}(thrust::get<0U>(tuple), physicalProperties.nu, physicalProperties.mt, physicalProperties.rhoP);
    return (isBurningSurface ? burningRate : static_cast<ElemT>(0));
  };

  GpuMatrix<ElemT> burningRates{ nx, ny, ElemT{} };
  thrust::transform(zipFirst, zipLast, std::begin(burningRates.values()), toBurningRate);

  return burningRates;
}

} // namespace detail

template <class ShapeT, class GasStateT>
GpuSrmSolver<ShapeT, GasStateT>::GpuSrmSolver(
    GpuGridT<ElemType> grid, PhysicalPropertiesData<ElemType> physicalProperties, ShapeT    shape,
    GasStateT initialState,
    GasParameters<ElemType> gasParameters,
    unsigned  iterationCount,
    ElemType  courant)
    : m_grid{ grid }, m_physicalProperties{ physicalProperties },
    m_boundaryConditions{m_grid.nx, m_grid.ny, EBoundaryCondition::eWall},
    m_normals{ m_grid.nx, m_grid.ny, CudaFloat2T<ElemType>{ 0, 0 } },
    m_surfacePoints{ m_grid.nx, m_grid.ny, CudaFloat2T<ElemType>{ 0, 0 } },
    m_indexMatrices{ m_grid.nx, m_grid.ny, IndexMatrixT{} },
    m_currState{ m_grid.nx, m_grid.ny, initialState },
    m_prevState{ m_grid.nx, m_grid.ny, initialState },
    m_firstState{ m_grid.nx, m_grid.ny, initialState },
    m_secondState{ m_grid.nx, m_grid.ny, initialState },
    m_levelSetSolver{ m_grid, shape, iterationCount, ETimeDiscretizationOrder::eThree },
    m_gasParameters{ gasParameters },
    m_closestIndicesMap(m_grid.n, thrust::make_pair(0U, 0U)),
    m_calculateBlocks(maxSizeX* maxSizeY, 0),
    m_courant{ courant }
{
    findClosestIndices();
}

template <class ShapeT, class GasStateT>
template <class CallbackT>
void GpuSrmSolver<ShapeT, GasStateT>::quasiStationaryDynamicIntegrate(
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
          detail::getTheoreticalBoriPressure<ShapeT>(phiValues, m_normals.values(), m_grid.nx, m_grid.ny, m_grid.hx, m_grid.hy,
              m_physicalProperties.kappa, m_physicalProperties.mt, m_physicalProperties.nu, m_physicalProperties.H0, m_physicalProperties.gammaComplex));
    const auto sBurn = detail::getBurningSurface<ShapeT>(phiValues, m_normals.values(), m_grid.nx, m_grid.ny, m_grid.hx, m_grid.hy);
    const auto chamberVolume = detail::getChamberVolume<ShapeT>(phiValues, m_grid.nx, m_grid.ny, m_grid.hx, m_grid.hy);
    desiredIntegrateTime += 450 * std::fabs(prevP - currP) * chamberVolume + levelSetDeltaT / 100;
    const auto gasDynamicDeltaT = std::min(desiredIntegrateTime, levelSetDeltaT);
    desiredIntegrateTime -= gasDynamicDeltaT;

    staticIntegrate(gasDynamicDeltaT, timeOrder, callback);
    if (i % 100 == 0)
    {
      cudaStreamSynchronize(nullptr);
      callback(m_currState, currPhi(), i, t, getMaxEquationDerivatives(), sBurn, ShapeT{}, m_grid.nx, m_grid.ny, m_grid.hx, m_grid.hy);
    }
    const auto dt = integrateInTime(levelSetDeltaT);
    t += dt;
  }
}

template <class ShapeT, class GasStateT>
template <class CallbackT>
void GpuSrmSolver<ShapeT, GasStateT>::dynamicIntegrate(
  unsigned iterationCount, ElemType deltaT, ETimeDiscretizationOrder timeOrder, CallbackT && callback)
{
  auto t{ static_cast<ElemType>(0.0) };
  for (unsigned i{ 0U }; i < iterationCount; ++i)
  {
    const auto deltaTGasDynamic = staticIntegrate(deltaT, timeOrder, callback);
    const auto maxDerivatives = getMaxEquationDerivatives();
    const auto sBurn = detail::getBurningSurface<ShapeT>(currPhi().values(), m_normals.values(), m_grid.nx, m_grid.ny, m_grid.hx, m_grid.hy);
    if (i % 10 == 0)
    {
      callback.operator()<ShapeT>(m_currState, m_gasParameters, currPhi(), i, t, getMaxEquationDerivatives(), sBurn, m_grid.nx, m_grid.ny, m_grid.hx, m_grid.hy);
    }
    const auto dt = integrateInTime(deltaTGasDynamic);
    t += dt;
  }
}

template <class ShapeT, class GasStateT>
template <class CallbackT>
auto GpuSrmSolver<ShapeT, GasStateT>::staticIntegrate(
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
      lambdas = detail::getMaxWaveSpeeds(m_currState.values(), m_gasParameters);
    }

    const auto dt = m_courant * m_grid.hx * m_grid.hy / (m_grid.hx * lambdas.x + m_grid.hy * lambdas.y);
    staticIntegrateStep(timeOrder, dt, lambdas);
    t += dt;

    if (i % 200U == 0U)
    {
      cudaStreamSynchronize(nullptr);
      callback(m_currState, currPhi(), m_grid.hx, m_grid.hy);
    }
    if (i % 5000U == 0U)
    {
      std::cout << i << ": " << t << '\n';
    }
  }

  return t;
}

template <class ShapeT, class GasStateT>
template <class CallbackT>
auto GpuSrmSolver<ShapeT, GasStateT>::staticIntegrate(
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
      lambdas = detail::getMaxWaveSpeeds(m_currState.values(), m_gasParameters);
    }

    const auto maxDt = m_courant * m_grid.hx * m_grid.hy / (m_grid.hx * lambdas.x + m_grid.hy * lambdas.y);
    const auto remainingTime = deltaT - t;
    auto dt = std::min(maxDt, remainingTime);
    dt = staticIntegrateStep(timeOrder, dt, lambdas);
    t += dt;
    ++i;
    if (i % 200U == 0U)
    {
      callback(m_currState, currPhi(), m_grid.hx, m_grid.hy);
    }
    if (i % 5000U == 0U)
    {
      std::cout << i << ": " << t << '\n';
    }
  }

  return t;
}

template <class ShapeT, class GasStateT>
void GpuSrmSolver<ShapeT, GasStateT>::findClosestIndices()
{
  m_closestIndicesMap.resize(m_grid.n);
  thrust::fill(std::begin(m_closestIndicesMap), 
               std::end(m_closestIndicesMap), 
               thrust::pair<unsigned, unsigned>{0U, 0U});
  detail::calculateGhostPointDataWrapper<ShapeT, order>(
      getDevicePtr(currPhi()),
      m_closestIndicesMap.data(),
      getDevicePtr(m_boundaryConditions),
      getDevicePtr(m_normals),
      getDevicePtr(m_surfacePoints),
      getDevicePtr(m_indexMatrices), m_grid);

  const auto removeIter = thrust::remove_if(std::begin(m_closestIndicesMap), 
                                            std::end(m_closestIndicesMap),
                                            thrust::placeholders::_1 == thrust::pair<unsigned, unsigned>{0U, 0U});
  m_closestIndicesMap.erase(removeIter, std::end(m_closestIndicesMap));
  fillCalculateBlockMatrix();
}

template <class ShapeT, class GasStateT>
void GpuSrmSolver<ShapeT, GasStateT>::fillCalculateBlockMatrix()
{
  assert(maxSizeX >= m_grid.gridSize.x && "Error. Max size is too small!");
  assert(maxSizeY >= m_grid.gridSize.y && "Error. Max size is too small!");

  thrust::host_vector<ElemType> currentLevel = currPhi().values();

  for (unsigned blockIdxX{ 0U }; blockIdxX < m_grid.gridSize.x; ++blockIdxX)
  {
    const auto startX = blockIdxX * m_grid.blockSize.x;
    for (unsigned blockIdxY{ 0U }; blockIdxY < m_grid.gridSize.y; ++blockIdxY)
    {
      const auto startY = blockIdxY * m_grid.blockSize.y;
      bool calculateBlock{ false };
      for (unsigned xIdx{ 0U }; (xIdx < m_grid.blockSize.x) && !calculateBlock; ++xIdx)
      {
        for (unsigned yIdx{ 0U }; (yIdx < m_grid.blockSize.y) && !calculateBlock; ++yIdx)
        {
          const auto index = (startY + yIdx) * m_grid.nx + startX + xIdx;
          if ((index < m_grid.n) && (currentLevel[index] < 0))
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

template <class ShapeT, class GasStateT>
void GpuSrmSolver<ShapeT, GasStateT>::writeIfNotValid() const
{
  if (thrust::all_of(std::begin(m_currState.values()), std::end(m_currState.values()), kae::IsValid{}))
  {
    return;
  }

  writeMatrixToFile(m_prevState, m_gasParameters, m_grid.hx, m_grid.hy,
                    "prev_error_p.dat", 
                    "prev_error_ux.dat", 
                    "prev_error_uy.dat", 
                    "prev_error_mach.dat", 
                    "prev_error_t.dat");
  writeMatrixToFile(m_currState, m_gasParameters, m_grid.hx, m_grid.hy,
                    "curr_error_p.dat", 
                    "curr_error_ux.dat", 
                    "curr_error_uy.dat", 
                    "curr_error_mach.dat", 
                    "curr_error_t.dat");
  writeMatrixToFile(currPhi(), m_grid.hx, m_grid.hy, "sgd.dat");
  throw std::runtime_error("Gas state has become invalid");
}

template <class ShapeT, class GasStateT>
auto GpuSrmSolver<ShapeT, GasStateT>::integrateInTime(ElemType deltaT) -> ElemType
{
  const auto burningRates = detail::getBurningRates<ShapeT>(
    m_currState, currPhi(), m_normals, m_physicalProperties, m_grid.nx, m_grid.ny, m_grid.hx, m_grid.hy);
  return m_levelSetSolver.integrateInTime(
    burningRates, deltaT, ETimeDiscretizationOrder::eThree);
}

template <class ShapeT, class GasStateT>
auto GpuSrmSolver<ShapeT, GasStateT>::getMaxEquationDerivatives() const
  -> CudaFloat4T<ElemType>
{
  return detail::getMaxEquationDerivatives(
    m_prevState.values(),
    m_currState.values(), m_gasParameters,
    detail::getDeltaT(m_currState.values(), m_gasParameters, m_courant, m_grid.hx, m_grid.hy));
}

template <class ShapeT, class GasStateT>
auto GpuSrmSolver<ShapeT, GasStateT>::staticIntegrateStep(
  ETimeDiscretizationOrder timeOrder,
  ElemType dt,
  CudaFloat2T<ElemType> lambdas) -> ElemType
{
  thrust::swap(m_prevState.values(), m_currState.values());
  switch (timeOrder)
  {
  case ETimeDiscretizationOrder::eOne:
    detail::srmIntegrateTVDSubStepWrapper<ShapeT, order, GasStateT>(
      getDevicePtr(m_prevState),
      {},
      getDevicePtr(m_currState),
      getDevicePtr(currPhi()), m_gasParameters,
      m_closestIndicesMap.data(),
      getDevicePtr(m_boundaryConditions),
      getDevicePtr(m_normals),
      getDevicePtr(m_surfacePoints),
      getDevicePtr(m_indexMatrices), m_grid, m_physicalProperties,
      static_cast<unsigned>(m_closestIndicesMap.size()), dt, lambdas, static_cast<ElemType>(1.0));
    break;

  case ETimeDiscretizationOrder::eTwo:
    detail::srmIntegrateTVDSubStepWrapper<ShapeT, order, GasStateT>(
      getDevicePtr(m_prevState),
      {},
      getDevicePtr(m_firstState),
      getDevicePtr(currPhi()), m_gasParameters,
      m_closestIndicesMap.data(),
      getDevicePtr(m_boundaryConditions),
      getDevicePtr(m_normals),
      getDevicePtr(m_surfacePoints),
      getDevicePtr(m_indexMatrices), m_grid, m_physicalProperties,
      static_cast<unsigned>(m_closestIndicesMap.size()), dt, lambdas, static_cast<ElemType>(1.0));

    detail::srmIntegrateTVDSubStepWrapper<ShapeT, order, GasStateT>(
      getDevicePtr(m_firstState),
      getDevicePtr(m_prevState),
      getDevicePtr(m_currState),
      getDevicePtr(currPhi()), m_gasParameters,
      m_closestIndicesMap.data(),
      getDevicePtr(m_boundaryConditions),
      getDevicePtr(m_normals),
      getDevicePtr(m_surfacePoints),
      getDevicePtr(m_indexMatrices), m_grid, m_physicalProperties,
      static_cast<unsigned>(m_closestIndicesMap.size()), dt, lambdas, static_cast<ElemType>(0.5));
    break;
  case ETimeDiscretizationOrder::eThree:
    detail::srmIntegrateTVDSubStepWrapper<ShapeT, order, GasStateT>(
      getDevicePtr(m_prevState),
      {},
      getDevicePtr(m_firstState),
      getDevicePtr(currPhi()), m_gasParameters,
      m_closestIndicesMap.data(),
      getDevicePtr(m_boundaryConditions),
      getDevicePtr(m_normals),
      getDevicePtr(m_surfacePoints),
      getDevicePtr(m_indexMatrices), m_grid, m_physicalProperties,
      static_cast<unsigned>(m_closestIndicesMap.size()), dt, lambdas, static_cast<ElemType>(1.0));

    detail::srmIntegrateTVDSubStepWrapper<ShapeT, order, GasStateT>(
      getDevicePtr(m_firstState),
      getDevicePtr(m_prevState),
      getDevicePtr(m_secondState),
      getDevicePtr(currPhi()), m_gasParameters,
      m_closestIndicesMap.data(),
      getDevicePtr(m_boundaryConditions),
      getDevicePtr(m_normals),
      getDevicePtr(m_surfacePoints),
      getDevicePtr(m_indexMatrices), m_grid, m_physicalProperties,
      static_cast<unsigned>(m_closestIndicesMap.size()), dt, lambdas, static_cast<ElemType>(0.25));

    detail::srmIntegrateTVDSubStepWrapper<ShapeT, order, GasStateT>(
      getDevicePtr(m_secondState),
      getDevicePtr(m_prevState),
      getDevicePtr(m_currState),
      getDevicePtr(currPhi()), m_gasParameters,
      m_closestIndicesMap.data(),
      getDevicePtr(m_boundaryConditions),
      getDevicePtr(m_normals),
      getDevicePtr(m_surfacePoints),
      getDevicePtr(m_indexMatrices), m_grid, m_physicalProperties,
      static_cast<unsigned>(m_closestIndicesMap.size()), dt, lambdas, static_cast<ElemType>(2.0 / 3.0));
    break;
  default:
    break;
  }

  return dt;
}

} // namespace 
