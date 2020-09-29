#pragma once

#include "cuda_float_types.h"
#include "float4_arithmetics.h"

namespace kae {

namespace detail {

namespace impl {

template <class GpuGridT, unsigned order, class ElemT, class ReturnT = kae::Matrix<unsigned, order, order>>
HOST_DEVICE ReturnT getStencilIndicesImpl(const ElemT *      pCurrPhi,
                                          CudaFloat2T<ElemT> scaledSurfacePoint,
                                          CudaFloat2T<ElemT> scaledNormal)
{
  ReturnT indicesMatrix{};
  const auto alongX = std::fabs(scaledNormal.x) > std::fabs(scaledNormal.y);
  if (alongX)
  {
    const auto xRounded = (scaledNormal.x > 0) ? std::floor(scaledSurfacePoint.x) : std::ceil(scaledSurfacePoint.x);
    const auto delta    = std::fabs(scaledSurfacePoint.x - xRounded);
    const auto startI   = static_cast<unsigned>(xRounded);
    const auto startJ   = scaledSurfacePoint.y - delta * scaledNormal.y;
    for (unsigned i{ 0U }; i < order; ++i)
    {
      const auto idxX        = (scaledNormal.x > 0) ? (startI - i) : (startI + i);
      const auto yCoordinate = startJ - i * scaledNormal.y;
      const auto startIdxY   = static_cast<unsigned>(std::round(yCoordinate));
      const auto roundDown   = yCoordinate - startIdxY > 0;
      auto externalPointMet = false;
      auto moveUp           = false;
      unsigned nodesAdded{};
      for (unsigned j{ 0U }; (j < 2U * order) && (nodesAdded < order); ++j)
      {
        const auto isOdd    = j % 2 == 1;
        const auto sumUpIdx = (isOdd && roundDown) || (!isOdd && !roundDown) || (externalPointMet && moveUp);
        const auto idxY     = sumUpIdx ? (startIdxY + (j + 1) / 2U) : (startIdxY - (j + 1) / 2U);
        const auto index    = idxY * GpuGridT::nx + idxX;
        if (pCurrPhi[index] < 0)
        {
          indicesMatrix(i, nodesAdded++) = index;
        }
        else
        {
          externalPointMet = true;
          moveUp = !sumUpIdx;
          continue;
        }

        if (externalPointMet)
        {
          ++j;
        }
      }
    }
  }
  else
  {
    const auto yRounded = (scaledNormal.y > 0) ? std::floor(scaledSurfacePoint.y) : std::ceil(scaledSurfacePoint.y);
    const auto delta = std::fabs(scaledSurfacePoint.y - yRounded);
    const auto startJ = static_cast<unsigned>(yRounded);
    const auto startI = scaledSurfacePoint.x - delta * scaledNormal.x;
    for (unsigned j{ 0U }; j < order; ++j)
    {
      const auto idxY = (scaledNormal.y > 0) ? (startJ - j) : (startJ + j);
      const auto xCoordinate = startI - j * scaledNormal.x;
      const auto startIdxX = static_cast<unsigned>(std::round(xCoordinate));
      const auto roundDown = xCoordinate - startIdxX > 0;
      auto externalPointMet = false;
      auto moveUp = false;
      unsigned nodesAdded{};
      for (unsigned i{ 0U }; (i < 2U * order) && (nodesAdded < order); ++i)
      {
        const auto isOdd = i % 2 == 1;
        const auto sumUpIdx = (isOdd && roundDown) || (!isOdd && !roundDown) || (externalPointMet && moveUp);
        const auto idxX = sumUpIdx ? (startIdxX + (i + 1) / 2U) : (startIdxX - (i + 1) / 2U);
        const auto index = idxY * GpuGridT::nx + idxX;
        if (pCurrPhi[index] < 0)
        {
          indicesMatrix(j, nodesAdded++) = index;
        }
        else
        {
          externalPointMet = true;
          moveUp = !sumUpIdx;
          continue;
        }

        if (externalPointMet)
        {
          ++i;
        }
      }
    }
  }

  return indicesMatrix;
}

} // namespace impl

template <class GpuGridT, unsigned order, class ElemT, class ReturnT = kae::Matrix<unsigned, order, order>>
HOST_DEVICE ReturnT getStencilIndices(const ElemT *      pCurrPhi,
                                      CudaFloat2T<ElemT> surfacePoint,
                                      CudaFloat2T<ElemT> normal)
{
  constexpr auto hxRec = GpuGridT::hxReciprocal;
  constexpr auto hyRec = GpuGridT::hyReciprocal;

  const CudaFloat2T<ElemT> scaledSurfacePoint{ surfacePoint.x * hxRec, surfacePoint.y * hyRec };
  const auto maxNormalElement = thrust::max(std::fabs(normal.x), std::fabs(normal.y));
  normal = { normal.x / maxNormalElement, normal.y / maxNormalElement };
  return impl::getStencilIndicesImpl<GpuGridT, order>(pCurrPhi, scaledSurfacePoint, normal);
}

} // namespace detail 

} // namespace kae
