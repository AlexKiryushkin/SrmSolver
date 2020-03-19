#pragma once

#include <cstddef>

#include "to_float.h"

namespace kae
{

template <unsigned Nx, unsigned Ny, class LxToType, class LyToType, unsigned SmExtension = 3U>
struct GpuGrid
{
  constexpr static unsigned nx{ Nx };
  constexpr static unsigned ny{ Ny };
  constexpr static unsigned n{ Nx * Ny };
  constexpr static float lx{ detail::ToFloatV<LxToType> };
  constexpr static float ly{ detail::ToFloatV<LyToType> };
  constexpr static float hx = lx / (nx - 1);
  constexpr static float hy = ly / (ny - 1);
  constexpr static float hxReciprocal = 1.0f / hx;
  constexpr static float hyReciprocal = 1.0f / hy;
  constexpr static dim3 blockSize{ 32U, 16U };
  constexpr static dim3 gridSize{ (Nx + blockSize.x - 1) / blockSize.x, (Ny + blockSize.y - 1) / blockSize.y };
  constexpr static unsigned smExtension = SmExtension;
  constexpr static dim3 sharedMemory{ blockSize.x + 2 * smExtension, blockSize.y + 2 * smExtension };
  constexpr static unsigned smSize = sharedMemory.x * sharedMemory.y;
  constexpr static unsigned smSizeBytes = smSize * sizeof(float);
};

} // namespace kae
