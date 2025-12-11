#pragma once

#include "cuda_includes.h"

#include "level_set_derivatives.h"
#include "shape/shape.h"

namespace kae {

namespace detail {

template <class ElemT>
__global__ void reinitializeTVDSubStep(thrust::device_ptr<const ElemT> pPrevValue,
                                       thrust::device_ptr<const ElemT> pFirstValue,
                                       thrust::device_ptr<ElemT>       pCurrValue,
                                       Shape<ElemT> shape,
                                       ElemT dt, ElemT prevWeight, unsigned nx, unsigned ny, unsigned smExtension, unsigned smx, ElemT h)
{
  const unsigned ti        = threadIdx.x;
  const unsigned ai        = ti + smExtension;

  const unsigned tj        = threadIdx.y;
  const unsigned aj        = tj + smExtension;

  const unsigned i         = ti + blockDim.x * blockIdx.x;
  const unsigned j         = tj + blockDim.y * blockIdx.y;
  if ((i >= nx) || (j >= ny))
  {
    return;
  }

  const unsigned sharedIdx = aj * smx + ai;
  const unsigned globalIdx = j * nx + i;

  const ElemT hReciprocal = static_cast<ElemT>(1) / h;

  extern __shared__ ElemT prevMatrix[];

  if ((ti < smExtension) && (i >= smExtension))
  {
    prevMatrix[sharedIdx - smExtension] = pPrevValue[globalIdx - smExtension];
  }

  if ((tj < smExtension) && (j >= smExtension))
  {
    prevMatrix[(aj - smExtension) * smx + ai] = pPrevValue[(j - smExtension) * nx + i];
  }

  prevMatrix[sharedIdx] = pPrevValue[globalIdx];

  if ((ti >= blockDim.x - smExtension) && (i + smExtension < nx))
  {
    prevMatrix[sharedIdx + smExtension] = pPrevValue[globalIdx + smExtension];
  }

  if ((tj >= blockDim.y - smExtension) && (j + smExtension < ny))
  {
    prevMatrix[(aj + smExtension) * smx + ai] = pPrevValue[(j + smExtension) * nx + i];
  }

  __syncthreads();

  const bool schemeShouldBeApplied = (i > smExtension + 1) && 
                                     (i < nx - smExtension - 2) && 
                                     (j > smExtension + 1) && 
                                     (j < ny - smExtension - 2) && 
      shape.shouldApplyScheme(i, j, h);

  if (schemeShouldBeApplied)
  {
    const ElemT sgdValue = prevMatrix[sharedIdx];
    const ElemT grad     = getLevelSetAbsGradient(prevMatrix, sharedIdx, smx, h, hReciprocal, (sgdValue > 0));
    const ElemT sgn      = sgdValue / std::hypot(sgdValue, grad * h);
    const ElemT val      = sgdValue - dt * sgn * (grad - static_cast<ElemT>(1.0));

    if (prevWeight != static_cast<ElemT>(1.0))
    {
      pCurrValue[globalIdx] = (1 - prevWeight) * pFirstValue[globalIdx] + prevWeight * val;
    }
    else
    {
      pCurrValue[globalIdx] = val;
    }
  }
}

template <class ElemT>
void reinitializeTVDSubStepWrapper(thrust::device_ptr<const ElemT> pPrevValue,
    thrust::device_ptr<const ElemT> pFirstValue,
    thrust::device_ptr<ElemT>       pCurrValue, Shape<ElemT> shape,
    GpuGridT<ElemT> grid,
    ElemT dt, ElemT prevWeight)
{
    reinitializeTVDSubStep<< <grid.gridSize, grid.blockSize, grid.smSize * sizeof(ElemT) >> >
        (pPrevValue, pFirstValue, pCurrValue, shape, dt, prevWeight, grid.nx, grid.ny, grid.smExtension, grid.sharedMemory.x, grid.hx);
    cudaDeviceSynchronize();
}

} // namespace detail

} // namespace kae
