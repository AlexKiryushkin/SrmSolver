#pragma once

#include "cuda_includes.h"

#include <Eigen/Core>

#include "boundary_condition.h"
#include "cuda_float_types.h"
#include "get_closest_index.h"
#include "get_extrapolated_ghost_value.h"
#include "level_set_derivatives.h"
#include "math_utilities.h"

namespace kae {

template <class GpuGridT, class ShapeT, class ElemT>
__global__ void calculateGhostPointData(thrust::device_ptr<const ElemT>                  pCurrPhi,
                                        thrust::device_ptr<unsigned>                     pClosestIndices,
                                        thrust::device_ptr<EBoundaryCondition>           pBoundaryConditions,
                                        thrust::device_ptr<CudaFloat2T<ElemT>>           pNormals,
                                        thrust::device_ptr<Eigen::Matrix<ElemT, 9U, 3U>> pLhsMatrices)
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

  unsigned counter{ 0U };
  Eigen::Matrix<ElemT, 9U, 3U> coordinateMatrix{};
  for (unsigned  iIdx{ i - 1 }; iIdx <= i + 1; ++iIdx)
  {
    for (unsigned jIdx{ j - 1 }; jIdx <= j + 1; ++jIdx)
    {
      const auto currentGlobalIdx{ jIdx * GpuGridT::nx + iIdx };
      if (pCurrPhi[currentGlobalIdx] < 0)
      {
        const CudaFloat2T<ElemT> globalDeltas{ iIdx * GpuGridT::hx - xSurface, jIdx * GpuGridT::hy - ySurface };
        const CudaFloat2T<ElemT> localDeltas{ TransformCoordinates{}(globalDeltas, { nx, ny }) };
        coordinateMatrix(counter, 0) = static_cast<ElemT>(1.0);
        coordinateMatrix(counter, 1) = localDeltas.x;
        coordinateMatrix(counter, 2) = localDeltas.y;
        ++counter;
      }
    }
  }

  const unsigned closestGlobalIdx = getClosestIndex<GpuGridT>(pCurrPhi.get(), i, j, nx, ny);
  pClosestIndices[globalIdx] = closestGlobalIdx;
  pBoundaryConditions[globalIdx] = boundaryCondition;
}

} // namespace kae
