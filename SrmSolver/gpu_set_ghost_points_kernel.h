#pragma once

#include "cuda_includes.h"

#include "boundary_condition.h"
#include "cuda_float_types.h"
#include "gas_state.h"
#include "get_extrapolated_ghost_value.h"

namespace kae {

namespace detail {

template <class GpuGridT, class GasStateT, class PropellantPropertiesT, class ElemT = typename GasStateT::ElemType>
__global__ void setFirstOrderGhostValues(thrust::device_ptr<GasStateT>                pGasValues,
                                         thrust::device_ptr<const ElemT>              pCurrPhi,
                                         thrust::device_ptr<const unsigned>           pClosestIndices,
                                         thrust::device_ptr<const EBoundaryCondition> pBoundaryConditions,
                                         thrust::device_ptr<CudaFloatT<2U, ElemT>>    pNormals)
{
  const unsigned i         = threadIdx.x + blockDim.x*blockIdx.x;
  const unsigned j         = threadIdx.y + blockDim.y*blockIdx.y;
  const unsigned globalIdx = j * GpuGridT::nx + i;
  if ((i >= GpuGridT::nx) || (j >= GpuGridT::ny))
  {
    return;
  }

  const ElemT level       = pCurrPhi[globalIdx];
  const bool pointIsGhost = (level >= 0) && (std::fabs(level) < 4 * GpuGridT::hx);
  if (!pointIsGhost)
  {
    return;
  }

  const unsigned closestGlobalIdx            = pClosestIndices[globalIdx];
  const EBoundaryCondition boundaryCondition = pBoundaryConditions[globalIdx];
  const CudaFloatT<2U, ElemT> normal         = pNormals[globalIdx];
  const GasStateT rotatedState               = Rotate::get(pGasValues.get()[closestGlobalIdx], normal.x, normal.y);
  const GasStateT extrapolatedState          = getFirstOrderExtrapolatedGhostValue<PropellantPropertiesT>(rotatedState, boundaryCondition);
  pGasValues[globalIdx]                      = ReverseRotate::get(extrapolatedState, normal.x, normal.y);
}

template <class GpuGridT, class GasStateT, class PropellantPropertiesT, class ElemT = typename GasStateT::ElemType>
void setFirstOrderGhostValuesWrapper(thrust::device_ptr<GasStateT>                pGasValues,
                                     thrust::device_ptr<const ElemT>              pCurrPhi,
                                     thrust::device_ptr<const unsigned>           pClosestIndices,
                                     thrust::device_ptr<const EBoundaryCondition> pBoundaryConditions,
                                     thrust::device_ptr<CudaFloatT<2U, ElemT>>    pNormals)
{
  setFirstOrderGhostValues<GpuGridT, GasStateT, PropellantPropertiesT><<<GpuGridT::gridSize, GpuGridT::blockSize>>>
  (pGasValues, pCurrPhi, pClosestIndices, pBoundaryConditions, pNormals);
  cudaDeviceSynchronize();
}

} // namespace detail

} // namespace kae
