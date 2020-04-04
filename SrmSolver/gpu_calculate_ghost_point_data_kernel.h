#pragma once

#include "cuda_includes.h"

#include <Eigen/Core>

#include "boundary_condition.h"
#include "cuda_float_types.h"
#include "get_closest_index.h"
#include "get_extrapolated_ghost_value.h"
#include "level_set_derivatives.h"

namespace kae {

template <class GpuGridT, class ShapeT, class ElemT>
__global__ void calculateGhostPointData(thrust::device_ptr<const ElemT>                  pCurrPhi,
                                        thrust::device_ptr<unsigned>                     pClosestIndices,
                                        thrust::device_ptr<EBoundaryCondition>           pBoundaryConditions,
                                        thrust::device_ptr<CudaFloatT<2U, ElemT>>        pNormals,
                                        thrust::device_ptr<Eigen::Matrix<ElemT, 9U, 9U>> pLhsMatrices)
{
  const unsigned i = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned j = threadIdx.y + blockDim.y * blockIdx.y;
  const unsigned globalIdx = j * GpuGridT::nx + i;
  if ((i >= GpuGridT::nx) || (j >= GpuGridT::ny))
  {
    return;
  }

  const ElemT level = pCurrPhi[globalIdx];
  const bool pointIsGhost = (level >= 0) && (std::fabs(level) < 4 * GpuGridT::hx);
  if (!pointIsGhost)
  {
    return;
  }

  ElemT nx = getLevelSetDerivative<GpuGridT, 1U>(pCurrPhi.get(), globalIdx, true);
  ElemT ny = getLevelSetDerivative<GpuGridT, GpuGridT::nx>(pCurrPhi.get(), globalIdx, true);
  const ElemT length = std::hypot(nx, ny);
  nx /= length;
  ny /= length;
  pNormals[globalIdx] = { nx, ny };

  const ElemT xSurface = i * GpuGridT::hx - nx * level;
  const ElemT ySurface = j * GpuGridT::hy - ny * level;
  const EBoundaryCondition boundaryCondition = ShapeT::getBoundaryCondition(xSurface, ySurface);
  if (boundaryCondition == EBoundaryCondition::eWall)
  {
    const ElemT iMirror = i - 2 * nx * level * GpuGridT::hxReciprocal;
    const ElemT jMirror = j - 2 * ny * level * GpuGridT::hxReciprocal;

    const int iMirrorInt = std::round(iMirror);
    const int jMirrorInt = std::round(jMirror);

    const ElemT sum = std::fabs(iMirror - iMirrorInt) + std::fabs(jMirror - jMirrorInt);
    if (sum < static_cast<ElemT>(0.01) * GpuGridT::hx)
    {
      const unsigned mirrorGlobalIdx = jMirrorInt * GpuGridT::nx + iMirrorInt;
      pClosestIndices[globalIdx] = mirrorGlobalIdx;
      pBoundaryConditions[globalIdx] = EBoundaryCondition::eMirror;
      return;
    }
  }

  const unsigned closestGlobalIdx = getClosestIndex<GpuGridT, ShapeT>(pCurrPhi.get(), i, j, nx, ny);
  pClosestIndices[globalIdx] = closestGlobalIdx;
  pBoundaryConditions[globalIdx] = boundaryCondition;
}

} // namespace kae
