#pragma once

#include "cuda_includes.h"

#include "cuda_float_types.h"
#include "float4_arithmetics.h"
#include "gas_dynamic_flux.h"
#include "gas_state.h"

namespace kae {

namespace detail {

template <class GpuGridT, class ShapeT, class GasStateT, class ElemT = typename GasStateT::ElemType>
__global__ void gasDynamicIntegrateTVDSubStep(const GasStateT * __restrict__ pPrevValue,
                                              const GasStateT * __restrict__ pFirstValue,
                                              GasStateT *       __restrict__ pCurrValue,
                                              const ElemT *     __restrict__ pCurrPhi,
                                              ElemT dt, CudaFloatT<2U, ElemT> lambda, ElemT prevWeight)
{
  constexpr auto     hx             = GpuGridT::hx;
  constexpr auto     levelThreshold = 5 * hx;
  constexpr unsigned smx            = GpuGridT::sharedMemory.x;
  constexpr unsigned nx             = GpuGridT::nx;
  constexpr unsigned ny             = GpuGridT::ny;
  constexpr unsigned smExtension    = GpuGridT::smExtension;

  const unsigned ti = threadIdx.x;
  const unsigned ai = ti + smExtension;

  const unsigned tj = threadIdx.y;
  const unsigned aj = tj + smExtension;

  const unsigned i = ti + blockDim.x * blockIdx.x;
  const unsigned j = tj + blockDim.y * blockIdx.y;
  if ((i >= nx) || (j >= ny))
  {
    return;
  }

  const unsigned sharedIdx = aj * smx + ai;
  const unsigned globalIdx = j * nx + i;

  __shared__ GasStateT prevMatrix[GpuGridT::smSize];
  __shared__ ElemT prevSgdMatrix[GpuGridT::smSize];
  __shared__ ElemT xFlux1[GpuGridT::smSize];
  __shared__ ElemT xFlux2[GpuGridT::smSize];
  __shared__ ElemT xFlux3[GpuGridT::smSize];
  __shared__ ElemT xFlux4[GpuGridT::smSize];
  __shared__ ElemT yFlux1[GpuGridT::smSize];
  __shared__ ElemT yFlux2[GpuGridT::smSize];
  __shared__ ElemT yFlux3[GpuGridT::smSize];
  __shared__ ElemT yFlux4[GpuGridT::smSize];

  const bool loadLeftHalo   = (ti < smExtension) && (i >= smExtension);
  const bool loadBottomHalo = (tj < smExtension) && (j >= smExtension);
  const bool loadRightHalo = (ti >= blockDim.x - smExtension) && (i + smExtension < nx);
  const bool loadTopHalo = (tj >= blockDim.y - smExtension) && (j + smExtension < ny);

  if (loadLeftHalo)
  {
    prevSgdMatrix[sharedIdx - smExtension] = pCurrPhi[globalIdx - smExtension];
  }

  if (loadBottomHalo)
  {
    prevSgdMatrix[sharedIdx - smExtension * smx] = pCurrPhi[globalIdx - smExtension * nx];
  }

  prevSgdMatrix[sharedIdx] = pCurrPhi[globalIdx];

  if (loadRightHalo)
  {
    prevSgdMatrix[sharedIdx + smExtension] = pCurrPhi[globalIdx + smExtension];
  }

  if (loadTopHalo)
  {
    prevSgdMatrix[sharedIdx + smExtension * smx] = pCurrPhi[globalIdx + smExtension * nx];
  }


  if (loadLeftHalo && (prevSgdMatrix[sharedIdx - smExtension] < levelThreshold))
  {
    prevMatrix[sharedIdx - smExtension] = pPrevValue[globalIdx - smExtension];
  }

  if (loadBottomHalo && (prevSgdMatrix[sharedIdx - smExtension * smx] < levelThreshold))
  {
    prevMatrix[sharedIdx - smExtension * smx] = pPrevValue[globalIdx - smExtension * nx];
  }

  if (prevSgdMatrix[sharedIdx] < levelThreshold)
  {
    prevMatrix[sharedIdx] = pPrevValue[globalIdx];
  }

  if (loadRightHalo && (prevSgdMatrix[sharedIdx + smExtension] < levelThreshold))
  {
    prevMatrix[sharedIdx + smExtension] = pPrevValue[globalIdx + smExtension];
  }

  if (loadTopHalo && (prevSgdMatrix[sharedIdx + smExtension * smx] < levelThreshold))
  {
    prevMatrix[sharedIdx + smExtension * smx] = pPrevValue[globalIdx + smExtension * nx];
  }

  __syncthreads();

  const ElemT levelValue = prevSgdMatrix[sharedIdx];
  const bool fluxShouldBeCalculated = (levelValue <= hx + static_cast<ElemT>(1e-6));
  if (fluxShouldBeCalculated)
  {
    if (tj == 0U)
    {
      yFlux1[sharedIdx - smx] = getFlux<Rho, MassFluxY, smx>(prevMatrix, sharedIdx - smx, lambda.y);
      yFlux2[sharedIdx - smx] = getFlux<MassFluxX, MomentumFluxXy, smx>(prevMatrix, sharedIdx - smx, lambda.y);
      yFlux3[sharedIdx - smx] = getFlux<MassFluxY, MomentumFluxYy, smx>(prevMatrix, sharedIdx - smx, lambda.y);
      yFlux4[sharedIdx - smx] = getFlux<RhoEnergy, EnthalpyFluxY, smx>(prevMatrix, sharedIdx - smx, lambda.y);
    }

    yFlux1[sharedIdx] = getFlux<Rho, MassFluxY, smx>(prevMatrix, sharedIdx, lambda.y);
    yFlux2[sharedIdx] = getFlux<MassFluxX, MomentumFluxXy, smx>(prevMatrix, sharedIdx, lambda.y);
    yFlux3[sharedIdx] = getFlux<MassFluxY, MomentumFluxYy, smx>(prevMatrix, sharedIdx, lambda.y);
    yFlux4[sharedIdx] = getFlux<RhoEnergy, EnthalpyFluxY, smx>(prevMatrix, sharedIdx, lambda.y);

    xFlux1[sharedIdx] = getFlux<Rho, MassFluxX, 1U>(prevMatrix, sharedIdx, lambda.x);
    xFlux2[sharedIdx] = getFlux<MassFluxX, MomentumFluxXx, 1U>(prevMatrix, sharedIdx, lambda.x);
    xFlux3[sharedIdx] = getFlux<MassFluxY, MomentumFluxXy, 1U>(prevMatrix, sharedIdx, lambda.x);
    xFlux4[sharedIdx] = getFlux<RhoEnergy, EnthalpyFluxX, 1U>(prevMatrix, sharedIdx, lambda.x);
  }

  if ((tj == 1U) && (ti < GpuGridT::blockSize.y))
  {
    const auto transposedSharedIdx = ai * smx + aj - 1U;
    if (prevSgdMatrix[transposedSharedIdx] <= hx + static_cast<ElemT>(1e-6))
    {

      xFlux1[transposedSharedIdx - 1U] = getFlux<Rho, MassFluxX, 1U>(prevMatrix, transposedSharedIdx - 1U, lambda.x);
      xFlux2[transposedSharedIdx - 1U] = getFlux<MassFluxX, MomentumFluxXx, 1U>(prevMatrix, transposedSharedIdx - 1U, lambda.x);
      xFlux3[transposedSharedIdx - 1U] = getFlux<MassFluxY, MomentumFluxXy, 1U>(prevMatrix, transposedSharedIdx - 1U, lambda.x);
      xFlux4[transposedSharedIdx - 1U] = getFlux<RhoEnergy, EnthalpyFluxX, 1U>(prevMatrix, transposedSharedIdx - 1U, lambda.x);
    }
  }

  __syncthreads();

  const bool schemeShouldBeApplied = (levelValue < 0);
  if (schemeShouldBeApplied)
  {
    const ElemT rReciprocal = 1 / ShapeT::getRadius(i, j);

    GasStateT calculatedGasState = prevMatrix[sharedIdx];
    const CudaFloatT<4U, ElemT> newConservativeVariables =
      {
        Rho::get(calculatedGasState) -
          dt * GpuGridT::hxReciprocal * (xFlux1[sharedIdx] - xFlux1[sharedIdx - 1U]) - 
          dt * GpuGridT::hyReciprocal * (yFlux1[sharedIdx] - yFlux1[sharedIdx - smx]) -
          dt * rReciprocal * MassFluxY::get(calculatedGasState),
        MassFluxX::get(calculatedGasState) -
          dt * GpuGridT::hxReciprocal * (xFlux2[sharedIdx] - xFlux2[sharedIdx - 1U]) -
          dt * GpuGridT::hyReciprocal * (yFlux2[sharedIdx] - yFlux2[sharedIdx - smx]) -
          dt * rReciprocal * MomentumFluxXy::get(calculatedGasState),
        MassFluxY::get(calculatedGasState) -
          dt * GpuGridT::hxReciprocal * (xFlux3[sharedIdx] - xFlux3[sharedIdx - 1U]) -
          dt * GpuGridT::hyReciprocal * (yFlux3[sharedIdx] - yFlux3[sharedIdx - smx]) -
          dt * rReciprocal * MassFluxY::get(calculatedGasState) * calculatedGasState.uy,
        RhoEnergy::get(calculatedGasState) -
          dt * GpuGridT::hxReciprocal * (xFlux4[sharedIdx] - xFlux4[sharedIdx - 1U]) -
          dt * GpuGridT::hyReciprocal * (yFlux4[sharedIdx] - yFlux4[sharedIdx - smx]) -
          dt * rReciprocal * EnthalpyFluxY::get(calculatedGasState)
      };

    calculatedGasState = ConservativeToGasState::get<GasStateT>(newConservativeVariables);
    if (prevWeight != 1)
    {
      const GasStateT firstGasState = pFirstValue[globalIdx];
      pCurrValue[globalIdx] = GasStateT{ (1 - prevWeight) * firstGasState.rho + prevWeight * calculatedGasState.rho,
                                         (1 - prevWeight) * firstGasState.ux  + prevWeight * calculatedGasState.ux,
                                         (1 - prevWeight) * firstGasState.uy  + prevWeight * calculatedGasState.uy,
                                         (1 - prevWeight) * firstGasState.p   + prevWeight * calculatedGasState.p };
    }
    else
    {
      pCurrValue[globalIdx] = calculatedGasState;
    }
  }
}

template <class GpuGridT, class ShapeT, std::size_t mult, class GasStateT, class ElemT = typename GasStateT::ElemType>
__global__ void gasDynamicIntegrateTVDSubStep(const GasStateT * __restrict__ pPrevValue,
                                              const GasStateT * __restrict__ pFirstValue,
                                              GasStateT *       __restrict__ pCurrValue,
                                              const ElemT *     __restrict__ pCurrPhi,
                                              ElemT dt, CudaFloatT<2U, ElemT> lambda, ElemT prevWeight)
{
  constexpr auto     hx             = GpuGridT::hx;
  constexpr auto     levelThreshold = 5 * hx;
  constexpr unsigned smExtension    = GpuGridT::smExtension;
  constexpr unsigned smx            = GpuGridT::blockSize.x * mult + 2 * smExtension;
  constexpr unsigned smy            = GpuGridT::blockSize.y * mult + 2 * smExtension;
  constexpr unsigned nx             = GpuGridT::nx;
  constexpr unsigned ny             = GpuGridT::ny;
  constexpr auto     smSize         = smx * smy;

  const unsigned ti = threadIdx.x * mult;
  const unsigned ai = smExtension + ti;

  const unsigned tj = threadIdx.y * mult;
  const unsigned aj = smExtension + tj;

  const unsigned i = ti + blockDim.x * blockIdx.x * mult;
  const unsigned j = tj + blockDim.y * blockIdx.y * mult;
  if ((i + mult >= nx) || (j + mult >= ny))
  {
    return;
  }

  const unsigned sharedIdx = aj * smx + ai;
  const unsigned globalIdx = j * nx + i;

  __shared__ GasStateT prevMatrix[smSize];
  __shared__ ElemT prevSgdMatrix[smSize];

  const bool loadLeft   = (ti == 0) && (i >= smExtension);
  const bool loadBottom = (tj == 0) && (j >= smExtension);
  const bool loadRight  = (ti == blockDim.x - 1U) && (i + smExtension + mult < nx);
  const bool loadTop    = (tj == blockDim.y - 1U) && (j + smExtension + mult < ny);

  /**
   * Load Level Set Function
   */
  if (loadLeft)
  {
#pragma unroll
    for (int row = 0; row < mult; ++row)
#pragma unroll
      for (int halo = 1; halo <= smExtension; ++halo)
      {
        prevSgdMatrix[sharedIdx + row * smx - halo] = pCurrPhi[globalIdx + row * nx - halo];
      }
  }

  if (loadBottom)
  {
#pragma unroll
    for (int halo = 1; halo <= smExtension; ++halo)
#pragma unroll
      for (int col = 0; col < mult; ++col)
      {
        prevSgdMatrix[sharedIdx + col - halo * smx] = pCurrPhi[globalIdx + col - halo * nx];
      }
  }

#pragma unroll
  for (int col = 0; col < mult; ++col)
#pragma unroll
    for (int row = 0; row < mult; ++row)
    {
      prevSgdMatrix[sharedIdx + col * smx + row] = pCurrPhi[globalIdx + col * nx + row];
    }

  if (loadRight)
  {
#pragma unroll
    for (int row = 0; row < mult; ++row)
#pragma unroll
      for (int halo = 1; halo <= smExtension; ++halo)
      {
        prevSgdMatrix[sharedIdx + row * smx + halo] = pCurrPhi[globalIdx + row * nx + halo];
      }
  }

  if (loadTop)
  {
#pragma unroll
    for (int halo = 1; halo <= smExtension; ++halo)
#pragma unroll
      for (int col = 0; col < mult; ++col)
      {
        prevSgdMatrix[sharedIdx + col + halo * smx] = pCurrPhi[globalIdx + col + halo * nx];
      }
  }

  /**
   * Load Gas State Values
   */
  if (loadLeft)
  {
#pragma unroll
    for (int row = 0; row < mult; ++row)
#pragma unroll
      for (int halo = 1; halo <= smExtension; ++halo)
      {
        if (prevSgdMatrix[sharedIdx + row * smx - halo] < levelThreshold)
        {
          prevMatrix[sharedIdx + row * smx - halo] = pPrevValue[globalIdx + row * nx - halo];
        }
      }
    
  }

  if (loadBottom)
  {

#pragma unroll
    for (int halo = 1; halo <= smExtension; ++halo)
#pragma unroll
      for (int col = 0; col < mult; ++col)
      {
        if (prevSgdMatrix[sharedIdx + col - halo * smx] < levelThreshold)
        {
          prevMatrix[sharedIdx + col - halo * smx] = pPrevValue[globalIdx + col - halo * nx];
        }
      }
  }

#pragma unroll
  for (int col = 0; col < mult; ++col)
#pragma unroll
    for (int row = 0; row < mult; ++row)
    {
      if (prevSgdMatrix[sharedIdx + col * smx + row] < levelThreshold)
      {
        prevMatrix[sharedIdx + col * smx + row] = pPrevValue[globalIdx + col * nx + row];
      }
    }

  if (loadRight)
  {
#pragma unroll
    for (int row = 0; row < mult; ++row)
#pragma unroll
      for (int halo = 1; halo <= smExtension; ++halo)
      {
        if (prevSgdMatrix[sharedIdx + row * smx + halo] < levelThreshold)
        {
          prevMatrix[sharedIdx + row * smx + halo] = pPrevValue[globalIdx + row * nx + halo];
        }
      }
  }

  if (loadTop)
  {
#pragma unroll
    for (int halo = 1; halo <= smExtension; ++halo)
#pragma unroll
      for (int col = 0; col < mult; ++col)
      {
        if (prevSgdMatrix[sharedIdx + col + halo * smx] < levelThreshold)
        {
          prevMatrix[sharedIdx + col + halo * smx] = pPrevValue[globalIdx + col + halo * nx];
        }
      }
  }

  __syncthreads();
  
#pragma unroll
  for (int col = 0; col < mult; ++col)
#pragma unroll
    for (int row = 0; row < mult; ++row)
    {
      if (prevSgdMatrix[sharedIdx + col * smx + row] < 0)
      {
        pCurrValue[globalIdx + col * nx + row] = prevMatrix[sharedIdx + col * smx + row];
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
  gasDynamicIntegrateTVDSubStep<GpuGridT, ShapeT, 2U, GasStateT><<<{GpuGridT::gridSize.x / 2, GpuGridT::gridSize.y / 2}, GpuGridT::blockSize>>>
  (pPrevValue.get(), pFirstValue.get(), pCurrValue.get(), pCurrPhi.get(), dt, lambda, pPrevWeight);
  cudaDeviceSynchronize();
  gasDynamicIntegrateTVDSubStep<GpuGridT, ShapeT, GasStateT><<<GpuGridT::gridSize, GpuGridT::blockSize>>>
  (pPrevValue.get(), pFirstValue.get(), pCurrValue.get(), pCurrPhi.get(), dt, lambda, pPrevWeight);
  cudaDeviceSynchronize();
}

} // namespace detail

} // namespace kae
