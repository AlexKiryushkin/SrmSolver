#pragma once

#include "cuda_includes.h"
#include "to_float.h"

namespace kae
{

template <unsigned Nx, unsigned Ny, class LxToType, class LyToType, unsigned SmExtension, class ElemT,
          unsigned blockSizeX = 32U * sizeof(float) / sizeof(ElemT),
          unsigned blockSizeY = 8U  * sizeof(float) / sizeof(ElemT)>
struct GpuGrid
{
  using ElemType = ElemT;

  constexpr static unsigned nx{ Nx };
  constexpr static unsigned ny{ Ny };
  constexpr static unsigned n{ Nx * Ny };
  constexpr static ElemT lx{ detail::ToFloatV<LxToType, ElemT> };
  constexpr static ElemT ly{ detail::ToFloatV<LyToType, ElemT> };
  constexpr static ElemT hx = lx / (nx - 1);
  constexpr static ElemT hy = ly / (ny - 1);
  constexpr static ElemT hxReciprocal = 1 / hx;
  constexpr static ElemT hyReciprocal = 1 / hy;
  constexpr static dim3 blockSize{ blockSizeX, blockSizeY };
  constexpr static dim3 gridSize{ (Nx + blockSize.x - 1) / blockSize.x, (Ny + blockSize.y - 1) / blockSize.y };
  constexpr static unsigned smExtension = SmExtension;
  constexpr static dim3 sharedMemory{ blockSize.x + 2 * smExtension, blockSize.y + 2 * smExtension };
  constexpr static unsigned smSize = sharedMemory.x * sharedMemory.y;
  constexpr static unsigned smSizeBytes = smSize * sizeof(ElemT);
};

template <class ElemT>
struct GpuGridT
{
    GpuGridT(unsigned nx_, unsigned ny_, ElemT lx_, ElemT ly_, unsigned smExtension_,
        unsigned blockSizeX_ = 32U * sizeof(float) / sizeof(ElemT),
        unsigned blockSizeY_ = 8U * sizeof(float) / sizeof(ElemT))
        : nx{ nx_ }, ny{ ny_ }, n{ nx * ny }, lx{ lx_ }, ly{ ly_ },
          hx{ lx / (nx - 1) }, hy{ ly / (ny - 1) }, hxReciprocal{ 1 / hx }, hyReciprocal{ 1 / hy },
          blockSize{ blockSizeX_, blockSizeY_ }, gridSize{ (nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y },
        smExtension{smExtension_}, sharedMemory{ blockSize.x + 2 * smExtension, blockSize.y + 2 * smExtension },
        smSize{ sharedMemory.x * sharedMemory.y }, smSizeBytes{smSize * sizeof(ElemT)}
    {
    }

    unsigned nx;
    unsigned ny;
    unsigned n;
    ElemT lx;
    ElemT ly;
    ElemT hx;
    ElemT hy;
    ElemT hxReciprocal;
    ElemT hyReciprocal;
    dim3 blockSize;
    dim3 gridSize;
    unsigned smExtension;
    dim3 sharedMemory;
    unsigned smSize;
    unsigned smSizeBytes;
};

} // namespace kae
