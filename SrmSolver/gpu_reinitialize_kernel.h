#pragma once

#include <thrust/device_ptr.h>

#include <device_launch_parameters.h>

#include "level_set_derivatives.h"

namespace kae {

namespace detail {

template <class GpuGridT, class ShapeT>
__global__ void reinitializeTVDSubStep(thrust::device_ptr<const float> pPrevValue,
                                       thrust::device_ptr<const float> pFirstValue,
                                       thrust::device_ptr<float>       pCurrValue,
                                       float dt, float prevWeight)
{
  const unsigned ti        = threadIdx.x;
  const unsigned ai        = ti + GpuGridT::smExtension;

  const unsigned tj        = threadIdx.y;
  const unsigned aj        = tj + GpuGridT::smExtension;

  const unsigned i         = ti + blockDim.x * blockIdx.x;
  const unsigned j         = tj + blockDim.y * blockIdx.y;
  if ((i >= GpuGridT::nx) || (j >= GpuGridT::ny))
  {
    return;
  }

  const unsigned sharedIdx = aj * GpuGridT::sharedMemory.x + ai;
  const unsigned globalIdx = j * GpuGridT::nx + i;

  __shared__ float prevMatrix[GpuGridT::smSize];

  if ((ti < GpuGridT::smExtension) && (i >= GpuGridT::smExtension))
  {
    prevMatrix[sharedIdx - GpuGridT::smExtension] = pPrevValue[globalIdx - GpuGridT::smExtension];
  }

  if ((tj < GpuGridT::smExtension) && (j >= GpuGridT::smExtension))
  {
    prevMatrix[(aj - GpuGridT::smExtension) * GpuGridT::sharedMemory.x + ai] = pPrevValue[(j - GpuGridT::smExtension) * GpuGridT::nx + i];
  }

  prevMatrix[sharedIdx] = pPrevValue[globalIdx];

  if ((ti >= blockDim.x - GpuGridT::smExtension) && (i + GpuGridT::smExtension < GpuGridT::nx))
  {
    prevMatrix[sharedIdx + GpuGridT::smExtension] = pPrevValue[globalIdx + GpuGridT::smExtension];
  }

  if ((tj >= blockDim.y - GpuGridT::smExtension) && (j + GpuGridT::smExtension < GpuGridT::ny))
  {
    prevMatrix[(aj + GpuGridT::smExtension) * GpuGridT::sharedMemory.x + ai] = pPrevValue[(j + GpuGridT::smExtension) * GpuGridT::nx + i];
  }

  __syncthreads();

  const bool schemeShouldBeApplied = (i > GpuGridT::smExtension + 1) && 
                                     (i < GpuGridT::nx - GpuGridT::smExtension - 2) && 
                                     (j > GpuGridT::smExtension + 1) && 
                                     (j < GpuGridT::ny - GpuGridT::smExtension - 2) && 
                                      ShapeT::shouldApplyScheme(i, j);

  if (schemeShouldBeApplied)
  {
    const float sgdValue = prevMatrix[sharedIdx];
    const float grad     = getLevelSetGradient<GpuGridT, GpuGridT::sharedMemory.x>(prevMatrix, sharedIdx, (sgdValue > 0.0f));
    const float sgn      = sgdValue / std::hypot(sgdValue, grad * GpuGridT::hx);
    const float val      = sgdValue - dt * sgn * (grad - 1.0f);

    if (prevWeight != 1.0f)
    {
      pCurrValue[globalIdx] = (1.0f - prevWeight) * pFirstValue[globalIdx] + prevWeight * val;
    }
    else
    {
      pCurrValue[globalIdx] = val;
    }
  }
}

template <class GpuGridT, class ShapeT>
void reinitializeTVDSubStepWrapper(thrust::device_ptr<const float> pPrevValue,
                                   thrust::device_ptr<const float> pFirstValue,
                                   thrust::device_ptr<float>       pCurrValue,
                                   float dt, float prevWeight)
{
  reinitializeTVDSubStep<GpuGridT, ShapeT><<<GpuGridT::gridSize, GpuGridT::blockSize>>>
  (pPrevValue, pFirstValue, pCurrValue, dt, prevWeight);
  cudaDeviceSynchronize();
}

} // namespace detail

} // namespace kae
