#pragma once

#include "cuda_includes.h"

#include "boundary_condition.h"
#include "cuda_float_types.h"
#include "gas_state.h"
#include "get_extrapolated_ghost_value.h"

namespace kae {

namespace detail {

template <class GasStateT, class ElemT = typename GasStateT::ElemType>
__global__ void setFirstOrderGhostValues(thrust::device_ptr<GasStateT>                              pGasValues,
                                         thrust::device_ptr<const ElemT>                            pCurrPhi,
                                         thrust::device_ptr<const thrust::pair<unsigned, unsigned>> pClosestIndicesMap,
                                         thrust::device_ptr<const EBoundaryCondition>               pBoundaryConditions,
                                         thrust::device_ptr<CudaFloat2T<ElemT>>                     pNormals,
                                         GasParameters<ElemT> gasParameters,
                                         PhysicalPropertiesData<ElemT> physicalProps,
                                         unsigned                                                   nClosestIndexElems)
{
  const unsigned i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i >= nClosestIndexElems)
  {
    return;
  }

  const auto indexMap             = pClosestIndicesMap.get()[i];
  const unsigned globalIdx        = indexMap.first;
  const unsigned closestGlobalIdx = indexMap.second;
  const auto boundaryCondition = pBoundaryConditions.get()[globalIdx];
  const auto normal            = pNormals.get()[globalIdx];
  const auto rotatedState      = Rotate::get(pGasValues.get()[closestGlobalIdx], normal.x, normal.y);
  const auto extrapolatedState = getFirstOrderExtrapolatedGhostValue(rotatedState,
      rotatedState, gasParameters,
      boundaryCondition, physicalProps.nu, physicalProps.mt, physicalProps.H0, physicalProps.P0);
  pGasValues[globalIdx]        = ReverseRotate::get(extrapolatedState, normal.x, normal.y);
}

template <class GasStateT, class ElemT>
void setFirstOrderGhostValuesWrapper(thrust::device_ptr<GasStateT>                              pGasValues,
                                     thrust::device_ptr<const ElemT>                            pCurrPhi,
                                     thrust::device_ptr<const thrust::pair<unsigned, unsigned>> pClosestIndices,
                                     thrust::device_ptr<const EBoundaryCondition>               pBoundaryConditions,
                                     thrust::device_ptr<CudaFloat2T<ElemT>>                  pNormals, GasParameters<ElemT> gasParameters, PhysicalPropertiesData<ElemT> physicalProps,
                                     unsigned nClosestIndexElems)
{
  constexpr unsigned blockSize = 256U;
  const unsigned gridSize = (nClosestIndexElems + blockSize - 1U) / blockSize;
  setFirstOrderGhostValues<GasStateT><<<gridSize, blockSize>>>
  (pGasValues, pCurrPhi, pClosestIndices, pBoundaryConditions, pNormals, gasParameters, physicalProps, nClosestIndexElems);
}

} // namespace detail

} // namespace kae
