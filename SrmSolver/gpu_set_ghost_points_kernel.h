#pragma once

#include <cstdio>

#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>

#include "boundary_condition.h"
#include "gas_state.h"
#include "get_closest_rotated_state_index.h"
#include "get_extrapolated_ghost_value.h"
#include "level_set_derivatives.h"
#include "math_utilities.h"

namespace kae {

namespace detail {

template <class GpuGridT, class GasStateT, class PropellantPropertiesT>
__global__ void setFirstOrderGhostValues(thrust::device_ptr<GasStateT>                pGasValues,
                                         thrust::device_ptr<const float>              pCurrPhi,
                                         thrust::device_ptr<const unsigned>           pClosestIndices,
                                         thrust::device_ptr<const EBoundaryCondition> pBoundaryConditions,
                                         thrust::device_ptr<float2>                   pNormals)
{
  const unsigned i         = threadIdx.x + blockDim.x*blockIdx.x;
  const unsigned j         = threadIdx.y + blockDim.y*blockIdx.y;
  const unsigned globalIdx = j * GpuGridT::nx + i;
  if ((i >= GpuGridT::nx) || (j >= GpuGridT::ny))
  {
    return;
  }

  const float level       = pCurrPhi[globalIdx];
  const bool pointIsGhost = (level >= 0.0f) && (std::fabs(level) < 4 * GpuGridT::hx);
  if (!pointIsGhost)
  {
    return;
  }

  const unsigned closestGlobalIdx            = pClosestIndices[globalIdx];
  const EBoundaryCondition boundaryCondition = pBoundaryConditions[globalIdx];
  const float2 normal                        = pNormals[globalIdx];
  const GasStateT rotatedState               = Rotate::get(pGasValues.get()[closestGlobalIdx], normal.x, normal.y);
  const GasStateT extrapolatedState          = getFirstOrderExtrapolatedGhostValue<PropellantPropertiesT>(rotatedState, boundaryCondition);
  pGasValues[globalIdx]                      = ReverseRotate::get(extrapolatedState, normal.x, normal.y);
}

template <class GpuGridT, class GasStateT, class PropellantPropertiesT>
void setFirstOrderGhostValuesWrapper(thrust::device_ptr<GasStateT>                pGasValues,
                                     thrust::device_ptr<const float>              pCurrPhi,
                                     thrust::device_ptr<const unsigned>           pClosestIndices,
                                     thrust::device_ptr<const EBoundaryCondition> pBoundaryConditions,
                                     thrust::device_ptr<float2>                   pNormals)
{
  setFirstOrderGhostValues<GpuGridT, GasStateT, PropellantPropertiesT><<<GpuGridT::gridSize, GpuGridT::blockSize>>>
  (pGasValues, pCurrPhi, pClosestIndices, pBoundaryConditions, pNormals);
  cudaDeviceSynchronize();
}

} // namespace detail

} // namespace kae
