#pragma once

#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>

#include "boundary_condition.h"
#include "get_closest_rotated_state_index.h"
#include "get_extrapolated_ghost_value.h"
#include "level_set_derivatives.h"

namespace kae {

namespace detail {

template <class GpuGridT, class ShapeT>
__global__ void findClosestIndices(thrust::device_ptr<const float>        pCurrPhi,
                                   thrust::device_ptr<unsigned>           pClosestIndices,
                                   thrust::device_ptr<EBoundaryCondition> pBoundaryConditions,
                                   thrust::device_ptr<float2>             pNormals)
{
  const unsigned i         = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned j         = threadIdx.y + blockDim.y * blockIdx.y;
  const unsigned globalIdx = j * GpuGridT::nx + i;
  if ((i >= GpuGridT::nx) || (j >= GpuGridT::ny))
  {
    return;
  }

  const float level = pCurrPhi[globalIdx];
  const bool pointIsGhost = (level >= 0.0f) && (std::fabs(level) < 4 * GpuGridT::hx);
  if (!pointIsGhost)
  {
    return;
  }

  float nx = getLevelSetDerivative<GpuGridT, 1U>(pCurrPhi.get(), globalIdx, true);
  float ny = getLevelSetDerivative<GpuGridT, GpuGridT::nx>(pCurrPhi.get(), globalIdx, true);
  const float length = std::hypot(nx, ny);
  nx /= length;
  ny /= length;
  pNormals[globalIdx] = { nx, ny };

  const EBoundaryCondition boundaryCondition = ShapeT::getBoundaryCondition(i * GpuGridT::hx - nx * level, j * GpuGridT::hy - ny * level);
  if (boundaryCondition == EBoundaryCondition::eWall)
  {
    const float iMirror  = i - 2 * nx * level * GpuGridT::hxReciprocal;
    const float jMirror  = j - 2 * ny * level * GpuGridT::hxReciprocal;

    const int iMirrorInt = std::round(iMirror);
    const int jMirrorInt = std::round(jMirror);

    const float sum      = std::fabs(iMirror - iMirrorInt) + std::fabs(jMirror - jMirrorInt);
    if (sum < 0.01f * GpuGridT::hx)
    {
      const unsigned mirrorGlobalIdx = jMirrorInt * GpuGridT::nx + iMirrorInt;
      pClosestIndices[globalIdx]     = mirrorGlobalIdx;
      pBoundaryConditions[globalIdx] = EBoundaryCondition::eMirror;
      return;
    }
  }

  const unsigned closestGlobalIdx = getClosestRotatedStateIndex<GpuGridT, ShapeT>(pCurrPhi.get(), i, j, nx, ny);
  pClosestIndices[globalIdx]      = closestGlobalIdx;
  pBoundaryConditions[globalIdx]  = boundaryCondition;
}

template <class GpuGridT, class ShapeT>
void findClosestIndicesWrapper(thrust::device_ptr<const float> pCurrPhi,
                               thrust::device_ptr<unsigned> pClosestIndices,
                               thrust::device_ptr<EBoundaryCondition> pBoundaryConditions,
                               thrust::device_ptr<float2> pNormals)
{
  findClosestIndices<GpuGridT, ShapeT><<<GpuGridT::gridSize, GpuGridT::blockSize>>>
    (pCurrPhi, pClosestIndices, pBoundaryConditions, pNormals);
  cudaDeviceSynchronize();
}

} // namespace detail

} // namespace kae
