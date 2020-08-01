#pragma once

#include "cuda_includes.h"

#include "boundary_condition.h"
#include "cuda_float_types.h"
#include "gas_state.h"
#include "get_coordinates_matrix.h"
#include "get_extrapolated_ghost_value.h"
#include "linear_system_solver.h"

template <class T>
using DevicePtr = thrust::device_ptr<T>;

template <class ElemT, unsigned order>
using LhsMatrix = Eigen::Matrix<ElemT, order* order, order* (order + 1U) / 2U>;

template <class ElemT, unsigned order>
using RhsMatrix = Eigen::Matrix<ElemT, order* order, 4U>;

template <class ElemT, unsigned order>
using SquareMatrix = Eigen::Matrix<ElemT, order, order>;

namespace kae {

namespace detail {

template <class GpuGridT, class GasStateT, class PhysicalPropertiesT, unsigned order, unsigned smSizeX,
          class InputMatrixT, class ElemT = typename GasStateT::ElemType>
__global__ void setGhostValues(DevicePtr<GasStateT>                              pGasValues,
                               DevicePtr<const thrust::pair<unsigned, unsigned>> pClosestIndicesMap,
                               DevicePtr<const EBoundaryCondition>               pBoundaryConditions,
                               DevicePtr<CudaFloat2T<ElemT>>                     pNormals,
                               DevicePtr<CudaFloat2T<ElemT>>                     pSurfacePoints,
                               DevicePtr<InputMatrixT>                           pIndexMatrix,
                               unsigned                                          nClosestIndexElems)
{
  const auto i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i >= nClosestIndexElems)
  {
    return;
  }

  __shared__ InputMatrixT indexMatrices[smSizeX];
  __shared__ LhsMatrix<ElemT, order> lhsMatrices[smSizeX];
  __shared__ RhsMatrix<ElemT, order> rhsMatrices[smSizeX];

  const auto indexMap         = pClosestIndicesMap.get()[i];
  const auto globalIdx        = indexMap.first;
  const auto closestGlobalIdx = indexMap.second;
  const auto surfacePoint     = pSurfacePoints.get()[globalIdx];
  const auto normal           = pNormals.get()[globalIdx];

  indexMatrices[threadIdx.x]  = pIndexMatrix[globalIdx];
  lhsMatrices[threadIdx.x]    = getCoordinatesMatrix<GpuGridT, order>(surfacePoint, normal, indexMatrices[threadIdx.x]);
  rhsMatrices[threadIdx.x]    = getRightHandSideMatrix<GpuGridT, order>(normal, pGasValues.get(), indexMatrices[threadIdx.x]);

  const auto A = lhsMatrices[threadIdx.x].transpose() * lhsMatrices[threadIdx.x];
  const auto x = lhsMatrices[threadIdx.x].transpose() * rhsMatrices[threadIdx.x];
  const auto l = choleskyDecompositionL(A);

  const auto boundaryCondition = pBoundaryConditions.get()[globalIdx];
  const auto rotatedState      = Rotate::get(pGasValues.get()[closestGlobalIdx], normal.x, normal.y);
  const auto extrapolatedState = getFirstOrderExtrapolatedGhostValue<PhysicalPropertiesT>(rotatedState, 
                                                                                            boundaryCondition);
  pGasValues[globalIdx]        = ReverseRotate::get(extrapolatedState, normal.x, normal.y);
}

template <class GpuGridT, class GasStateT, class PhysicalPropertiesT, unsigned order,
          class InputMatrixT, class ElemT = typename GasStateT::ElemType>
void setGhostValuesWrapper(DevicePtr<GasStateT>                              pGasValues,
                           DevicePtr<const thrust::pair<unsigned, unsigned>> pClosestIndices,
                           DevicePtr<const EBoundaryCondition>               pBoundaryConditions,
                           DevicePtr<CudaFloat2T<ElemT>>                     pNormals,
                           DevicePtr<CudaFloat2T<ElemT>>                     pSurfacePoints,
                           DevicePtr<InputMatrixT>                           pIndexMatrix,
                           unsigned                                          nClosestIndexElems)
{
  constexpr unsigned blockSize = 64U;
  const unsigned gridSize = (nClosestIndexElems + blockSize - 1U) / blockSize;
  setGhostValues<GpuGridT, GasStateT, PhysicalPropertiesT, order, blockSize><<<gridSize, blockSize>>>
  (pGasValues, pClosestIndices, pBoundaryConditions, pNormals, pSurfacePoints, pIndexMatrix, nClosestIndexElems);
}

} // namespace detail

} // namespace kae

