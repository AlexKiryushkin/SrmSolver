#pragma once

#include "cuda_includes.h"

#include "boundary_condition.h"
#include "cuda_float_types.h"
#include "get_closest_index.h"
#include "get_coordinates_matrix.h"
#include "get_extrapolated_ghost_value.h"
#include "level_set_derivatives.h"
#include "math_utilities.h"
#include "matrix.h"
#include "matrix_operations.h"

namespace kae {

namespace detail {

template <class GpuGridT, class ShapeT, unsigned order, class ElemT>
__global__ void calculateGhostPointData(const ElemT *                         pCurrPhi,
                                        thrust::pair<unsigned, unsigned> *    pClosestIndices,
                                        EBoundaryCondition *                  pBoundaryConditions,
                                        CudaFloat2T<ElemT> *                  pNormals,
                                        CudaFloat2T<ElemT> *                  pSurfacePoints,
                                        kae::Matrix<unsigned, order, order> * pStencilIndices)
{
  const unsigned i         = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned j         = threadIdx.y + blockDim.y * blockIdx.y;
  const unsigned globalIdx = j * GpuGridT::nx + i;
  if ((i >= GpuGridT::nx) || (j >= GpuGridT::ny))
  {
    return;
  }

  if ((i < 10U) || (j < 10U) || (i >= GpuGridT::nx - 10) || (j >= GpuGridT::ny - 10))
  {
    return;
  }

  ElemT nx = getLevelSetDerivative<GpuGridT, 1U>(pCurrPhi, globalIdx, true);
  ElemT ny = getLevelSetDerivative<GpuGridT, GpuGridT::nx>(pCurrPhi, globalIdx, true);
  const ElemT length = std::hypot(nx, ny);
  nx /= length;
  ny /= length;
  pNormals[globalIdx] = { nx, ny };

  const ElemT level = pCurrPhi[globalIdx];
  const bool pointIsGhost = (level >= 0) && (std::fabs(level) < 5 * GpuGridT::hx);
  if (!pointIsGhost)
  {
    return;
  }

  const CudaFloat2T<ElemT> surfacePoint{ i * GpuGridT::hx - nx * level,  j * GpuGridT::hy - ny * level };
  pSurfacePoints[globalIdx] = surfacePoint;

  const EBoundaryCondition boundaryCondition = ShapeT::getBoundaryCondition(surfacePoint.x ,surfacePoint.y);
  if (boundaryCondition == EBoundaryCondition::eWall && level > GpuGridT::hx / 4)
  {
    const ElemT iMirror = i - 2 * nx * level * GpuGridT::hxReciprocal;
    const ElemT jMirror = j - 2 * ny * level * GpuGridT::hxReciprocal;

    const int iMirrorInt = std::round(iMirror);
    const int jMirrorInt = std::round(jMirror);

    const ElemT sum = std::fabs(iMirror - iMirrorInt) + std::fabs(jMirror - jMirrorInt);
    if (sum < sqr(GpuGridT::hx))
    {
      const unsigned mirrorGlobalIdx = jMirrorInt * GpuGridT::nx + iMirrorInt;
      pClosestIndices[globalIdx]     = thrust::make_pair(globalIdx, mirrorGlobalIdx);;
      pBoundaryConditions[globalIdx] = EBoundaryCondition::eMirror;
      return;
    }
  }

  const unsigned closestGlobalIdx = getClosestIndex<GpuGridT>(pCurrPhi, i, j, nx, ny);
  pClosestIndices[globalIdx]      = thrust::make_pair(globalIdx, closestGlobalIdx);
  pBoundaryConditions[globalIdx]  = boundaryCondition;
  pStencilIndices[globalIdx]      = getStencilIndices<GpuGridT, order>(pCurrPhi, surfacePoint, { nx, ny });
}

template <class GpuGridT, class ShapeT, unsigned order, class ElemT>
void calculateGhostPointDataWrapper(thrust::device_ptr<const ElemT>                        pCurrPhi,
                                    thrust::device_ptr<thrust::pair<unsigned, unsigned>>   pClosestIndices,
                                    thrust::device_ptr<EBoundaryCondition>                 pBoundaryConditions,
                                    thrust::device_ptr<CudaFloat2T<ElemT>>                 pNormals,
                                    thrust::device_ptr<CudaFloat2T<ElemT>>                 pSurfacePoints,
                                    thrust::device_ptr<kae::Matrix<unsigned, order, order>> pStencilIndices)
{
  calculateGhostPointData<GpuGridT, ShapeT, order><<<GpuGridT::gridSize, GpuGridT::blockSize>>>
    (pCurrPhi.get(), pClosestIndices.get(), pBoundaryConditions.get(), 
     pNormals.get(), pSurfacePoints.get(), pStencilIndices.get());
  cudaDeviceSynchronize();
}

} // namespace detail

} // namespace kae
