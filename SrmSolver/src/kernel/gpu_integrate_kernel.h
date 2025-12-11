#pragma once

#include "cuda_includes.h"

#include "level_set_derivatives.h"
#include "physical_properties.h"

namespace kae {

namespace detail {

template <class ElemT>
__global__ void integrateEqTvdSubStep(thrust::device_ptr<const ElemT>  pPrevValue,
    thrust::device_ptr<const ElemT>  pFirstValue,
    thrust::device_ptr<ElemT>        pCurrValue,
    thrust::device_ptr<const  ElemT> pVelocities,
    ElemT dt, ElemT prevWeight, unsigned nx, unsigned ny, unsigned smExtension, unsigned smx, ElemT h)
{
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

    const ElemT hReciprocal =  static_cast<ElemT>(1) / h;

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

    const bool schemeShouldBeApplied = (std::fabs(prevMatrix[sharedIdx]) < 10 * h);
    if (schemeShouldBeApplied)
    {
        const ElemT sgdValue = prevMatrix[sharedIdx];

        const ElemT un = pVelocities[globalIdx];
        const ElemT normalX = getLevelSetDerivative(prevMatrix, sharedIdx, 1U, h, hReciprocal, (un > 0));
        const ElemT normalY = getLevelSetDerivative(prevMatrix, sharedIdx, smx, h, hReciprocal, (un > 0));

        const ElemT grad = ((un != 0) ? std::hypot(normalX, normalY) : 0);
        const ElemT val = sgdValue - dt * un * grad;
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

template <class ElemT>
void integrateEqTvdSubStepWrapper(thrust::device_ptr<const ElemT> pPrevValue,
    thrust::device_ptr<const ElemT> pFirstValue,
    thrust::device_ptr<ElemT>       pCurrValue,
    thrust::device_ptr<const ElemT> pVelocities,
    GpuGridT<ElemT> grid,
    ElemT dt, ElemT prevWeight)
{
    integrateEqTvdSubStep << <grid.gridSize, grid.blockSize, grid.smSize * sizeof(ElemT) >> >
        (pPrevValue, pFirstValue, pCurrValue, pVelocities, dt, prevWeight, grid.nx, grid.ny, grid.smExtension, grid.sharedMemory.x, grid.hx);
    cudaDeviceSynchronize();
}

} // namespace detail

} // namespace kae
