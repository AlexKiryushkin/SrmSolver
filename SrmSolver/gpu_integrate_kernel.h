#pragma once

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include <device_launch_parameters.h>

#include "level_set_derivatives.h"
#include "propellant_properties.h"

namespace kae {

namespace detail {

template <class GpuGridT, class ShapeT, class PropellantPropertiesT, class GasStateT>
__global__ void integrateEqTvdSubStep(thrust::device_ptr<const float>    pPrevValue,
                                      thrust::device_ptr<const float>    pFirstValue,
                                      thrust::device_ptr<float>          pCurrValue,
                                      thrust::device_ptr<GasStateT>      pGasStates,
                                      thrust::device_ptr<const unsigned> pClosestIndices,
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

  const bool schemeShouldBeApplied = (std::fabs(prevMatrix[sharedIdx]) < 10 * GpuGridT::hx);
  if (schemeShouldBeApplied)
  {
    const float sgdValue = prevMatrix[sharedIdx];

    const float normalX  = getLevelSetDerivative<GpuGridT, 1U>(prevMatrix, sharedIdx, true);
    const float normalY  = getLevelSetDerivative<GpuGridT, GpuGridT::sharedMemory.x>(prevMatrix, sharedIdx, true);
    const float xSurface = i * GpuGridT::hx - normalX * sgdValue;
    const float ySurface = j * GpuGridT::hy - normalY * sgdValue;

    const bool pointIsOnGrain       = ShapeT::isPointOnGrain(xSurface, ySurface);
    const unsigned closestGlobalIdx = ((sgdValue < 0.0f) ? globalIdx : pClosestIndices[globalIdx]);
    const float un                  = (pointIsOnGrain ? 
                                         kae::BurningRate<PropellantPropertiesT>{}(pGasStates.get()[closestGlobalIdx]) :
                                         0.0f);
    const float grad                = ((un != 0.0f) ? std::hypot(normalX, normalY) : 0.0f);
    const float val                 = sgdValue - dt * un * grad;
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

template <class GpuGridT, class ShapeT, class PropellantPropertiesT, class GasStateT>
void integrateEqTvdSubStepWrapper(thrust::device_ptr<const float>    pPrevValue,
                                  thrust::device_ptr<const float>    pFirstValue,
                                  thrust::device_ptr<float>          pCurrValue,
                                  thrust::device_ptr<GasStateT>      pGasStates,
                                  thrust::device_ptr<const unsigned> pClosestIndices,
                                  float dt, float prevWeight)
{
  integrateEqTvdSubStep<GpuGridT, ShapeT, PropellantPropertiesT><<<GpuGridT::gridSize, GpuGridT::blockSize>>>
  (pPrevValue, pFirstValue, pCurrValue, pGasStates, pClosestIndices, dt, prevWeight);
  cudaDeviceSynchronize();
}

} // namespace detail

} // namespace kae
