#pragma once

#include "to_float.h"

namespace kae
{

template <unsigned Nx, unsigned Ny, class LxToType, class LyToType, unsigned SmExtension, class ElemT>
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
  constexpr static dim3 blockSize{ 32U * sizeof(float) / sizeof(ElemT), 16U * sizeof(float) / sizeof(ElemT) };
  constexpr static dim3 gridSize{ (Nx + blockSize.x - 1) / blockSize.x, (Ny + blockSize.y - 1) / blockSize.y };
  constexpr static unsigned smExtension = SmExtension;
  constexpr static dim3 sharedMemory{ blockSize.x + 2 * smExtension, blockSize.y + 2 * smExtension };
  constexpr static unsigned smSize = sharedMemory.x * sharedMemory.y;
  constexpr static unsigned smSizeBytes = smSize * sizeof(ElemT);
};

} // namespace kae
