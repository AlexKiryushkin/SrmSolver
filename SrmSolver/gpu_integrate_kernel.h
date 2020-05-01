#pragma once

#include "cuda_includes.h"

#include "level_set_derivatives.h"
#include "propellant_properties.h"

namespace kae {

namespace detail {

template <class GpuGridT,
          class ElemT>
__global__ void integrateEqTvdSubStep(thrust::device_ptr<const ElemT>  pPrevValue,
                                      thrust::device_ptr<const ElemT>  pFirstValue,
                                      thrust::device_ptr<ElemT>        pCurrValue,
                                      thrust::device_ptr<const  ElemT> pVelocities,
                                      ElemT dt, ElemT prevWeight)
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

  constexpr auto smx = GpuGridT::sharedMemory.x;
  const unsigned sharedIdx = aj * smx + ai;
  const unsigned globalIdx = j * GpuGridT::nx + i;

  __shared__ ElemT prevMatrix[GpuGridT::smSize];

  if ((ti < GpuGridT::smExtension) && (i >= GpuGridT::smExtension))
  {
    prevMatrix[sharedIdx - GpuGridT::smExtension] = pPrevValue[globalIdx - GpuGridT::smExtension];
  }

  if ((tj < GpuGridT::smExtension) && (j >= GpuGridT::smExtension))
  {
    prevMatrix[(aj - GpuGridT::smExtension) * smx + ai] = pPrevValue[(j - GpuGridT::smExtension) * GpuGridT::nx + i];
  }

  prevMatrix[sharedIdx] = pPrevValue[globalIdx];

  if ((ti >= blockDim.x - GpuGridT::smExtension) && (i + GpuGridT::smExtension < GpuGridT::nx))
  {
    prevMatrix[sharedIdx + GpuGridT::smExtension] = pPrevValue[globalIdx + GpuGridT::smExtension];
  }

  if ((tj >= blockDim.y - GpuGridT::smExtension) && (j + GpuGridT::smExtension < GpuGridT::ny))
  {
    prevMatrix[(aj + GpuGridT::smExtension) * smx + ai] = pPrevValue[(j + GpuGridT::smExtension) * GpuGridT::nx + i];
  }

  __syncthreads();

  const bool schemeShouldBeApplied = (std::fabs(prevMatrix[sharedIdx]) < 10 * GpuGridT::hx);
  if (schemeShouldBeApplied)
  {
    const ElemT sgdValue = prevMatrix[sharedIdx];

    const ElemT normalX  = getLevelSetDerivative<GpuGridT, 1U>(prevMatrix, sharedIdx, true);
    const ElemT normalY  = getLevelSetDerivative<GpuGridT, smx>(prevMatrix, sharedIdx, true);

    const ElemT un                  = pVelocities[globalIdx];
    const ElemT grad                = ((un != 0) ? std::hypot(normalX, normalY) : 0);
    const ElemT val                 = sgdValue - dt * un * grad;
    if (prevWeight != 1)
    {
      pCurrValue[globalIdx] = (1 - prevWeight) * pFirstValue[globalIdx] + prevWeight * val;
    }
    else
    {
      pCurrValue[globalIdx] = val;
    }
  }
}

template <class GpuGridT,
          class ElemT>
void integrateEqTvdSubStepWrapper(thrust::device_ptr<const ElemT> pPrevValue,
                                  thrust::device_ptr<const ElemT> pFirstValue,
                                  thrust::device_ptr<ElemT>       pCurrValue,
                                  thrust::device_ptr<const ElemT> pVelocities,
                                  ElemT dt, ElemT prevWeight)
{
  integrateEqTvdSubStep<GpuGridT><<<GpuGridT::gridSize, GpuGridT::blockSize>>>
  (pPrevValue, pFirstValue, pCurrValue, pVelocities, dt, prevWeight);
  cudaDeviceSynchronize();
}

} // namespace detail

} // namespace kae
