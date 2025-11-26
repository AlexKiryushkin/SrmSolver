#pragma once

#include "cuda_includes.h"

#include "boundary_condition.h"
#include "cuda_float_types.h"
#include "get_closest_index.h"
#include "get_coordinates_matrix.h"
#include "get_extrapolated_ghost_value.h"
#include "level_set_derivatives.h"
#include "math_utilities.h"
#include "matrix/matrix.h"
#include "matrix/matrix_operations.h"

namespace kae {

namespace detail {

template <class ShapeT, unsigned order, class ElemT>
__global__ void calculateGhostPointData(const ElemT *                         pCurrPhi,
                                        thrust::pair<unsigned, unsigned> *    pClosestIndices,
                                        EBoundaryCondition *                  pBoundaryConditions,
                                        CudaFloat2T<ElemT> *                  pNormals,
                                        CudaFloat2T<ElemT> *                  pSurfacePoints,
                                        kae::Matrix<unsigned, order, order> * pStencilIndices,
                                        unsigned nx, unsigned ny, ElemT hx, ElemT hy)
{
  const unsigned i         = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned j         = threadIdx.y + blockDim.y * blockIdx.y;
  const unsigned globalIdx = j * nx + i;
  if ((i >= nx) || (j >= ny))
  {
    return;
  }

  if ((i < 10U) || (j < 10U) || (i >= nx - 10) || (j >= ny - 10))
  {
    return;
  }

  ElemT hxReciprocal = 1 / hx;

  ElemT normalX = getLevelSetDerivative(pCurrPhi, globalIdx, 1U, hx, hxReciprocal, true);
  ElemT normalY = getLevelSetDerivative(pCurrPhi, globalIdx, nx, hx, hxReciprocal, true);
  const ElemT length = std::hypot(normalX, normalY);
  normalX /= length;
  normalY /= length;
  pNormals[globalIdx] = { normalX, normalY };

  const ElemT level = pCurrPhi[globalIdx];
  const bool pointIsGhost = (level >= 0) && (std::fabs(level) < 5 * hx);
  if (!pointIsGhost)
  {
    return;
  }

  const CudaFloat2T<ElemT> surfacePoint{ i * hx - normalX * level,  j * hy - normalY * level };
  pSurfacePoints[globalIdx] = surfacePoint;

  const EBoundaryCondition boundaryCondition = ShapeT::getBoundaryCondition(surfacePoint.x ,surfacePoint.y);
  if (boundaryCondition == EBoundaryCondition::eWall && level > hx / 4)
  {
    const ElemT iMirror = i - 2 * normalX * level * hxReciprocal;
    const ElemT jMirror = j - 2 * normalY * level * hxReciprocal;

    const int iMirrorInt = std::round(iMirror);
    const int jMirrorInt = std::round(jMirror);

    const ElemT sum = std::fabs(iMirror - iMirrorInt) + std::fabs(jMirror - jMirrorInt);
    if (sum < static_cast<ElemT>(0.005) * hx)
    {
      const unsigned mirrorGlobalIdx = jMirrorInt * nx + iMirrorInt;
      pClosestIndices[globalIdx]     = thrust::make_pair(globalIdx, mirrorGlobalIdx);;
      pBoundaryConditions[globalIdx] = EBoundaryCondition::eMirror;
      return;
    }
  }

  const unsigned closestGlobalIdx = getClosestIndex(pCurrPhi, i, j, normalX, normalY, nx, ny, hx, hy);
  pClosestIndices[globalIdx]      = thrust::make_pair(globalIdx, closestGlobalIdx);
  pBoundaryConditions[globalIdx]  = boundaryCondition;
  pStencilIndices[globalIdx]      = getStencilIndices<order>(pCurrPhi, surfacePoint, { normalX, normalY }, nx, hx, hy);
}

template <class ShapeT, unsigned order, class ElemT>
void calculateGhostPointDataWrapper(thrust::device_ptr<const ElemT>                        pCurrPhi,
    thrust::device_ptr<thrust::pair<unsigned, unsigned>>   pClosestIndices,
    thrust::device_ptr<EBoundaryCondition>                 pBoundaryConditions,
    thrust::device_ptr<CudaFloat2T<ElemT>>                 pNormals,
    thrust::device_ptr<CudaFloat2T<ElemT>>                 pSurfacePoints,
    thrust::device_ptr<kae::Matrix<unsigned, order, order>> pStencilIndices,
    GpuGridT<ElemT> grid)
{
    calculateGhostPointData<ShapeT, order> << <grid.gridSize, grid.blockSize >> >
        (pCurrPhi.get(), pClosestIndices.get(), pBoundaryConditions.get(),
            pNormals.get(), pSurfacePoints.get(), pStencilIndices.get(), grid.nx, grid.ny, grid.hx, grid.hy);
    cudaDeviceSynchronize();
}

} // namespace detail

} // namespace kae
