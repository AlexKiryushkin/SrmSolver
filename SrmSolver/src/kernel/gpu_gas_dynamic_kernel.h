#pragma once

#include "cuda_includes.h"

#include "cuda_float_types.h"
#include "gas_dynamic_flux.h"
#include "gas_state.h"
#include "shape/shape.h"

constexpr unsigned maxSizeX = 120U;
constexpr unsigned maxSizeY = 200U;
__constant__ int8_t calculateBlockMatrix[maxSizeX * maxSizeY];

namespace kae {

namespace detail {

template <class GasStateT, class ElemT = typename GasStateT::ElemType>
__global__ void
/*__launch_bounds__ (256, 5)*/
gasDynamicIntegrateTVDSubStep(const GasStateT * __restrict__ pPrevValue,
                                              const GasStateT * __restrict__ pFirstValue,
                                              GasStateT *       __restrict__ pCurrValue,
                                              const ElemT *     __restrict__ pCurrPhi, Shape<ElemT> shape, GasParameters<ElemT> gasParameters,
                                              ElemT dt, CudaFloat2T<ElemT> lambda, ElemT prevWeight, unsigned nx, unsigned ny, ElemT hx, ElemT hy, unsigned smx, unsigned smy, unsigned smExtension)
{
  const auto levelThreshold = 4 * hx;
  const auto hxReciprocal = 1 / hx;
  const auto hyReciprocal = 1 / hy;

  const unsigned smSize = smx * smy;

  const unsigned fluxSmx        = blockDim.x + 1U;
  const unsigned fluxSmy        = blockDim.y + 1U;
  const unsigned fluxSmSize     = fluxSmx * fluxSmy;

  const unsigned ti = threadIdx.x;
  const unsigned ai = ti + smExtension;

  const unsigned tj = threadIdx.y;
  const unsigned aj = tj + smExtension;

  const unsigned i = ti + blockDim.x * blockIdx.x;
  const unsigned j = tj + blockDim.y * blockIdx.y;
  if ((i >= nx) || (j >= ny) || (calculateBlockMatrix[blockIdx.y * maxSizeX + blockIdx.x] == 0))
  {
    return;
  }

  const unsigned sharedIdx = aj * smx + ai;
  const unsigned globalIdx = j * nx + i;

  extern __shared__ float sharedData[];

  GasStateT *prevMatrix = (GasStateT*)sharedData;

  ElemT* xFlux1 = (ElemT *)(prevMatrix + smSize);
  ElemT* xFlux2 = xFlux1 + fluxSmSize;
  ElemT* xFlux3 = xFlux2 + fluxSmSize;
  ElemT* xFlux4 = xFlux3 + fluxSmSize;
  ElemT* yFlux1 = xFlux4 + fluxSmSize;
  ElemT* yFlux2 = yFlux1 + fluxSmSize;
  ElemT* yFlux3 = yFlux2 + fluxSmSize;
  ElemT* yFlux4 = yFlux3 + fluxSmSize;

  const auto levelValue = __ldg(&pCurrPhi[globalIdx]);

  if ((ti < smExtension) && (i >= smExtension))
  {
    prevMatrix[sharedIdx - smExtension] = pPrevValue[globalIdx - smExtension];
  }

  if ((tj < smExtension) && (j >= smExtension))
  {
    prevMatrix[sharedIdx - smExtension * smx] = pPrevValue[globalIdx - smExtension * nx];
  }

  if (levelValue < levelThreshold)
  {
    prevMatrix[sharedIdx] = pPrevValue[globalIdx];
  }

  if ((ti >= blockDim.x - smExtension) && (i + smExtension < nx))
  {
    prevMatrix[sharedIdx + smExtension] = pPrevValue[globalIdx + smExtension];
  }

  if ((tj >= blockDim.y - smExtension) && (j + smExtension < ny))
  {
    prevMatrix[sharedIdx + smExtension * smx] = pPrevValue[globalIdx + smExtension * nx];
  }

  __syncthreads();

  if ((tj == 1U) && (ti < blockDim.y))
  {
    const auto transposedSharedIdx = ai * smx + aj - 1U;
    {
      const auto transFluxSharedIdx = (ti + 1U) * fluxSmx + tj;
      xFlux1[transFluxSharedIdx - 1U] = getFlux<Rho, MassFluxX>(prevMatrix, gasParameters, transposedSharedIdx - 1U, lambda.x, hx, 1U);
      xFlux2[transFluxSharedIdx - 1U] = getFlux<MassFluxX, MomentumFluxXx>(prevMatrix, gasParameters, transposedSharedIdx - 1U, lambda.x, hx, 1U);
      xFlux3[transFluxSharedIdx - 1U] = getFlux<MassFluxY, MomentumFluxXy>(prevMatrix, gasParameters, transposedSharedIdx - 1U, lambda.x, hx, 1U);
      xFlux4[transFluxSharedIdx - 1U] = getFlux<RhoEnergy, EnthalpyFluxX>(prevMatrix, gasParameters, transposedSharedIdx - 1U, lambda.x, hx, 1U);
    }
  }

  const unsigned fluxSharedIdx = (tj + 1U) * fluxSmx + ti + 1U;
  const bool fluxShouldBeCalculated = (levelValue <= hx + static_cast<ElemT>(1e-6));
  if (fluxShouldBeCalculated)
  {
    if (tj == 0U)
    {
      yFlux1[fluxSharedIdx - fluxSmx] = getFlux<Rho, MassFluxY>(prevMatrix, gasParameters, sharedIdx - smx, lambda.y, hy, smx);
      yFlux2[fluxSharedIdx - fluxSmx] = getFlux<MassFluxX, MomentumFluxXy>(prevMatrix, gasParameters, sharedIdx - smx, lambda.y, hy, smx);
      yFlux3[fluxSharedIdx - fluxSmx] = getFlux<MassFluxY, MomentumFluxYy>(prevMatrix, gasParameters, sharedIdx - smx, lambda.y, hy, smx);
      yFlux4[fluxSharedIdx - fluxSmx] = getFlux<RhoEnergy, EnthalpyFluxY>(prevMatrix, gasParameters, sharedIdx - smx, lambda.y, hy, smx);
    }

    yFlux1[fluxSharedIdx] = getFlux<Rho, MassFluxY>(prevMatrix, gasParameters, sharedIdx, lambda.y, hy, smx);
    yFlux2[fluxSharedIdx] = getFlux<MassFluxX, MomentumFluxXy>(prevMatrix, gasParameters, sharedIdx, lambda.y, hy, smx);
    yFlux3[fluxSharedIdx] = getFlux<MassFluxY, MomentumFluxYy>(prevMatrix, gasParameters, sharedIdx, lambda.y, hy, smx);
    yFlux4[fluxSharedIdx] = getFlux<RhoEnergy, EnthalpyFluxY>(prevMatrix, gasParameters, sharedIdx, lambda.y, hy, smx);

    xFlux1[fluxSharedIdx] = getFlux<Rho, MassFluxX>(prevMatrix, gasParameters, sharedIdx, lambda.x, hx, 1U);
    xFlux2[fluxSharedIdx] = getFlux<MassFluxX, MomentumFluxXx>(prevMatrix, gasParameters, sharedIdx, lambda.x, hx, 1U);
    xFlux3[fluxSharedIdx] = getFlux<MassFluxY, MomentumFluxXy>(prevMatrix, gasParameters, sharedIdx, lambda.x, hx, 1U);
    xFlux4[fluxSharedIdx] = getFlux<RhoEnergy, EnthalpyFluxX>(prevMatrix, gasParameters, sharedIdx, lambda.x, hx, 1U);
  }

  __syncthreads();

  const bool schemeShouldBeApplied = (levelValue < 0);
  if (schemeShouldBeApplied)
  {
    const ElemT rReciprocal = 1 / shape.getRadius(i * hx, j * hy);

    GasStateT calculatedGasState = prevMatrix[sharedIdx];
    CudaFloat4T<ElemT> newConservativeVariables =
      {
        Rho::get(calculatedGasState, gasParameters) -
          dt * hxReciprocal * (xFlux1[fluxSharedIdx] - xFlux1[fluxSharedIdx - 1U]) -
          dt * hyReciprocal * (yFlux1[fluxSharedIdx] - yFlux1[fluxSharedIdx - fluxSmx]) -
          dt * rReciprocal * MassFluxY::get(calculatedGasState, gasParameters),
        MassFluxX::get(calculatedGasState, gasParameters) -
          dt * hxReciprocal * (xFlux2[fluxSharedIdx] - xFlux2[fluxSharedIdx - 1U]) -
          dt * hyReciprocal * (yFlux2[fluxSharedIdx] - yFlux2[fluxSharedIdx - fluxSmx]) -
          dt * rReciprocal * MomentumFluxXy::get(calculatedGasState, gasParameters),
        MassFluxY::get(calculatedGasState, gasParameters) -
          dt * hxReciprocal * (xFlux3[fluxSharedIdx] - xFlux3[fluxSharedIdx - 1U]) -
          dt * hyReciprocal * (yFlux3[fluxSharedIdx] - yFlux3[fluxSharedIdx - fluxSmx]) -
          dt * rReciprocal * MassFluxY::get(calculatedGasState, gasParameters) * calculatedGasState.uy,
        RhoEnergy::get(calculatedGasState, gasParameters) -
          dt * hxReciprocal * (xFlux4[fluxSharedIdx] - xFlux4[fluxSharedIdx - 1U]) -
          dt * hyReciprocal * (yFlux4[fluxSharedIdx] - yFlux4[fluxSharedIdx - fluxSmx]) -
          dt * rReciprocal * EnthalpyFluxY::get(calculatedGasState, gasParameters)
      };
    if (prevWeight != 1)
    {
      const GasStateT firstGasState = pFirstValue[globalIdx];
      newConservativeVariables.x = prevWeight * newConservativeVariables.x + (1 - prevWeight) * Rho::get(firstGasState, gasParameters);
      newConservativeVariables.y = prevWeight * newConservativeVariables.y + (1 - prevWeight) * MassFluxX::get(firstGasState, gasParameters);
      newConservativeVariables.z = prevWeight * newConservativeVariables.z + (1 - prevWeight) * MassFluxY::get(firstGasState, gasParameters);
      newConservativeVariables.w = prevWeight * newConservativeVariables.w + (1 - prevWeight) * RhoEnergy::get(firstGasState, gasParameters);
    }
    calculatedGasState = ConservativeToGasState::get<GasStateT>(newConservativeVariables, gasParameters);
    pCurrValue[globalIdx] = calculatedGasState;
  }
}

template <class GasStateT, class ElemT>
void gasDynamicIntegrateTVDSubStepWrapper(thrust::device_ptr<const GasStateT> pPrevValue,
                                          thrust::device_ptr<const GasStateT> pFirstValue,
                                          thrust::device_ptr<GasStateT> pCurrValue,
                                          thrust::device_ptr<const ElemT> pCurrPhi, Shape<ElemT> shape, GasParameters<ElemT> gasParameters, GpuGridT<ElemT> grid,
                                          ElemT dt, CudaFloat2T<ElemT> lambda, ElemT pPrevWeight)
{
    const unsigned fluxSmx = grid.blockSize.x + 1U;
    const unsigned fluxSmy = grid.blockSize.y + 1U;
    const unsigned fluxSmSize = fluxSmx * fluxSmy;

    const auto smSize = grid.smSize * sizeof(GasStateT) + 8U * fluxSmSize * sizeof(ElemT);
    gasDynamicIntegrateTVDSubStep<GasStateT> << <grid.gridSize, grid.blockSize, smSize >> >
        (pPrevValue.get(), pFirstValue.get(), pCurrValue.get(), pCurrPhi.get(), shape, gasParameters, dt, lambda, pPrevWeight, grid.nx, grid.ny, grid.hx, grid.hy, grid.sharedMemory.x, grid.sharedMemory.y, grid.smExtension);
}

} // namespace detail

} // namespace kae
