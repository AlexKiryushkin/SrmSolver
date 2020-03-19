#pragma once

#include "math_utilities.h"

namespace kae {

namespace detail {

template <class GpuGridT, class Shape>
__host__ __device__ unsigned getClosestRotatedStateIndex(const float * pCurrPhi,
                                                         unsigned      i,
                                                         unsigned      j,
                                                         float         nx,
                                                         float         ny)
{
  const unsigned globalIdx = j * GpuGridT::nx + i;
  const float level        = pCurrPhi[globalIdx];

  const float xSurface    = i * GpuGridT::hx - nx * level;
  const float ySurface    = j * GpuGridT::hy - ny * level;

  const unsigned iSurface = std::round(xSurface * GpuGridT::hxReciprocal);
  const unsigned jSurface = std::round(ySurface * GpuGridT::hyReciprocal);

  float minDistanceSquared = 100.0f * GpuGridT::hx * GpuGridT::hx;
  unsigned iClosest   = 0U;
  unsigned jClosest   = 0U;
  for (unsigned iCl = iSurface - 3; iCl <= iSurface + 3; ++iCl)
    for (unsigned jCl = jSurface - 3; jCl <= jSurface + 3; ++jCl)
    {
      if (pCurrPhi[jCl * GpuGridT::nx + iCl] >= 0)
      {
        continue;
      }

      float distanceSquared = sqr(iCl * GpuGridT::hx - xSurface) + sqr(jCl * GpuGridT::hy - ySurface);
      if (minDistanceSquared < distanceSquared)
      {
        continue;
      }

      iClosest = iCl;
      jClosest = jCl;
      minDistanceSquared = distanceSquared;
    }

  return jClosest * GpuGridT::nx + iClosest;
}

} // namespace detail

} // namespace kae