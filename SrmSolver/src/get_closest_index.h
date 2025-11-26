#pragma once

#include "math_utilities.h"

namespace kae {

namespace detail {

template <class ElemT>
HOST_DEVICE unsigned getClosestIndex(const ElemT * pCurrPhi,
                                     unsigned      i,
                                     unsigned      j,
                                     ElemT         normalX,
                                     ElemT         normalY,
                                     unsigned nx, unsigned ny, ElemT hx, ElemT hy)
{
  const unsigned globalIdx = j * nx + i;
  const ElemT level        = pCurrPhi[globalIdx];

  const ElemT xSurface    = i * hx - normalX * level;
  const ElemT ySurface    = j * hy - normalY * level;

  const unsigned iSurface = std::round(xSurface  / hx);
  const unsigned jSurface = std::round(ySurface  / hy);

  ElemT minDistanceSquared = std::numeric_limits<ElemT>::max();
  unsigned iClosest   = nx * ny;
  unsigned jClosest   = nx * ny;
  for (unsigned iCl = iSurface - 3; iCl <= iSurface + 3; ++iCl)
  {
    for (unsigned jCl = jSurface - 3; jCl <= jSurface + 3; ++jCl)
    {
      if (pCurrPhi[jCl * nx + iCl] >= 0)
      {
        continue;
      }

      ElemT distanceSquared = sqr(iCl * hx - xSurface) + sqr(jCl * hy - ySurface);
      if (minDistanceSquared < distanceSquared)
      {
        continue;
      }

      iClosest = iCl;
      jClosest = jCl;
      minDistanceSquared = distanceSquared;
    }
  }
  return jClosest * nx + iClosest;
}

} // namespace detail

} // namespace kae