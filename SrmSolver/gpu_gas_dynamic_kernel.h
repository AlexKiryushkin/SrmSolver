#pragma once

#include <thrust/device_ptr.h>

#include <device_launch_parameters.h>

#include "cuda_float_types.h"
#include "float4_arithmetics.h"
#include "gas_dynamic_flux.h"
#include "gas_state.h"

namespace kae {

namespace detail {

template <class GpuGridT, class ShapeT, class GasStateT, class ElemT = typename GasStateT::ElemType>
__global__ void gasDynamicIntegrateTVDSubStep(thrust::device_ptr<const GasStateT> pPrevValue,
                                              thrust::device_ptr<const GasStateT> pFirstValue,
                                              thrust::device_ptr<GasStateT>       pCurrValue,
                                              thrust::device_ptr<const ElemT>     pCurrPhi,
                                              ElemT dt, CudaFloatT<2U, ElemT> lambda, ElemT prevWeight)
{
  const unsigned ti = threadIdx.x;
  const unsigned ai = ti + GpuGridT::smExtension;

  const unsigned tj = threadIdx.y;
  const unsigned aj = tj + GpuGridT::smExtension;

  const unsigned i = ti + blockDim.x * blockIdx.x;
  const unsigned j = tj + blockDim.y * blockIdx.y;
  if (i >= GpuGridT::nx || j >= GpuGridT::ny)
  {
    return;
  }

  constexpr unsigned smx   = GpuGridT::sharedMemory.x;
  const unsigned sharedIdx = aj * smx + ai;
  const unsigned globalIdx = j * GpuGridT::nx + i;

  __shared__ GasStateT prevMatrix[GpuGridT::smSize];
  __shared__ ElemT prevSgdMatrix[GpuGridT::smSize];
  __shared__ CudaFloatT<4U, ElemT> xFluxes[GpuGridT::smSize];
  __shared__ CudaFloatT<4U, ElemT> yFluxes[GpuGridT::smSize];

  if ((ti < GpuGridT::smExtension) && (i >= GpuGridT::smExtension))
  {
    prevSgdMatrix[sharedIdx - GpuGridT::smExtension] = pCurrPhi[globalIdx - GpuGridT::smExtension];
    if (prevSgdMatrix[sharedIdx - GpuGridT::smExtension] < 5 * GpuGridT::hx)
    {
      prevMatrix[sharedIdx - GpuGridT::smExtension] = pPrevValue[globalIdx - GpuGridT::smExtension];
    }
  }

  if ((tj < GpuGridT::smExtension) && (j >= GpuGridT::smExtension))
  {
    prevSgdMatrix[(aj - GpuGridT::smExtension) * smx + ai] = pCurrPhi[(j - GpuGridT::smExtension) * GpuGridT::nx + i];
    if (prevSgdMatrix[(aj - GpuGridT::smExtension) * smx + ai] < 5 * GpuGridT::hx)
    {
      prevMatrix[(aj - GpuGridT::smExtension) * smx + ai] = pPrevValue[(j - GpuGridT::smExtension) * GpuGridT::nx + i];
    }
  }

  prevSgdMatrix[sharedIdx] = pCurrPhi[globalIdx];
  if (prevSgdMatrix[sharedIdx] < 5 * GpuGridT::hx)
  {
    prevMatrix[sharedIdx] = pPrevValue[globalIdx];
  }

  if ((ti >= blockDim.x - GpuGridT::smExtension) && (i + GpuGridT::smExtension < GpuGridT::nx))
  {
    prevSgdMatrix[sharedIdx + GpuGridT::smExtension] = pCurrPhi[globalIdx + GpuGridT::smExtension];
    if (prevSgdMatrix[sharedIdx + GpuGridT::smExtension] < 5 * GpuGridT::hx)
    {
      prevMatrix[sharedIdx + GpuGridT::smExtension] = pPrevValue[globalIdx + GpuGridT::smExtension];
    }
  }

  if ((tj >= blockDim.y - GpuGridT::smExtension) && (j + GpuGridT::smExtension < GpuGridT::ny))
  {
    prevSgdMatrix[(aj + GpuGridT::smExtension) * smx + ai] = pCurrPhi[(j + GpuGridT::smExtension) * GpuGridT::nx + i];
    if (prevSgdMatrix[(aj + GpuGridT::smExtension) * smx + ai] < 5 * GpuGridT::hx)
    {
      prevMatrix[(aj + GpuGridT::smExtension) * smx + ai] = pPrevValue[(j + GpuGridT::smExtension) * GpuGridT::nx + i];
    }
  }

  __syncthreads();

  ElemT levelValue = prevSgdMatrix[sharedIdx];
  bool fluxShouldBeCalculated = levelValue <= GpuGridT::hx + static_cast<ElemT>(1e-6);
  if (fluxShouldBeCalculated)
  {
    if (ti == 0U)
    {
      xFluxes[sharedIdx - 1U] = getXFluxes<1U>(prevMatrix, sharedIdx - 1U, lambda.x);
    }

    if (tj == 0U)
    {
      yFluxes[sharedIdx - smx] = getYFluxes<smx>(prevMatrix, sharedIdx - smx, lambda.y);
    }

    xFluxes[sharedIdx] = getXFluxes<1U>(prevMatrix, sharedIdx, lambda.x);
    yFluxes[sharedIdx] = getYFluxes<smx>(prevMatrix, sharedIdx, lambda.y);
  }

  __syncthreads();

  bool schemeShouldBeApplied = levelValue < 0;
  if (schemeShouldBeApplied)
  {
    const ElemT rReciprocal = 1 / ShapeT::getRadius(i, j);
    const auto newConservativeVariables = ConservativeVariables::get(prevMatrix[sharedIdx]) -
                                          dt * GpuGridT::hxReciprocal * (xFluxes[sharedIdx] - xFluxes[sharedIdx - 1U]) -
                                          dt * GpuGridT::hyReciprocal * (yFluxes[sharedIdx] - yFluxes[sharedIdx - smx]) -
                                          dt * rReciprocal * SourceTerm::get(prevMatrix[sharedIdx]);
    const GasStateT newGasState = ConservativeToGasState::get<GasStateT>(newConservativeVariables);

    if (prevWeight != 1)
    {
      GasStateT firstGasState = pFirstValue[globalIdx];
      pCurrValue[globalIdx] = GasStateT{ (1 - prevWeight) * firstGasState.rho + prevWeight * newGasState.rho,
                                         (1 - prevWeight) * firstGasState.ux  + prevWeight * newGasState.ux,
                                         (1 - prevWeight) * firstGasState.uy  + prevWeight * newGasState.uy,
                                         (1 - prevWeight) * firstGasState.p   + prevWeight * newGasState.p };
    }
    else
    {
      pCurrValue[globalIdx] = newGasState;
    }
  }
}

template <class GpuGridT, class ShapeT, class GasStateT, class ElemT>
void gasDynamicIntegrateTVDSubStepWrapper(thrust::device_ptr<const GasStateT> pPrevValue,
                                          thrust::device_ptr<const GasStateT> pFirstValue,
                                          thrust::device_ptr<GasStateT> pCurrValue,
                                          thrust::device_ptr<const ElemT> pCurrPhi,
                                          ElemT dt, CudaFloatT<2U, ElemT> lambda, ElemT pPrevWeight)
{
  gasDynamicIntegrateTVDSubStep<GpuGridT, ShapeT, GasStateT><<<GpuGridT::gridSize, GpuGridT::blockSize>>>
  (pPrevValue, pFirstValue, pCurrValue, pCurrPhi, dt, lambda, pPrevWeight);
  cudaDeviceSynchronize();
}

} // namespace detail

} // namespace kae
