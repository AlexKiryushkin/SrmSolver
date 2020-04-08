#pragma once

#include <Eigen/Core>

#include "cuda_float_types.h"
#include "math_utilities.h"

namespace kae {

namespace detail {

template <class GpuGridT, class ElemT = typename GpuGridT::ElemType>
Eigen::Matrix<ElemT, 9U, 3U> getCoordinatesMatrix(const ElemT * pCurrPhi, unsigned closestIdx, ElemT xSurface, ElemT ySurface)
{
  unsigned counter{ 0U };
  Eigen::Matrix<ElemT, 9U, 3U> coordinateMatrix{};

  const unsigned i = closestIdx % GpuGridT::nx;
  const unsigned j = closestIdx / GpuGridT::nx;
  for (unsigned iIdx{ i - 1 }; iIdx <= i + 1; ++iIdx)
  {
    for (unsigned jIdx{ j - 1 }; jIdx <= j + 1; ++jIdx)
    {
      const auto currentGlobalIdx{ jIdx * GpuGridT::nx + iIdx };
      if (pCurrPhi[currentGlobalIdx] < 0)
      {
        const CudaFloatT<2U, ElemT> globalDeltas{ iIdx * GpuGridT::hx - xSurface, jIdx * GpuGridT::hy - ySurface };
        const CudaFloatT<2U, ElemT> localDeltas{ TransformCoordinates{}(globalDeltas, { nx, ny }) };
        coordinateMatrix(counter, 0) = static_cast<ElemT>(1.0);
        coordinateMatrix(counter, 1) = localDeltas.x;
        coordinateMatrix(counter, 2) = localDeltas.y;
        ++counter;
      }
    }
  }

  return coordinateMatrix;
}

} // detail

} // namespace kae
