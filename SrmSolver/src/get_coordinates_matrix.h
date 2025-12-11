#pragma once

#include "cuda_float_types.h"
#include "gas_state.h"
#include "get_stencil_indices.h"
#include "math_utilities.h"
#include "matrix/matrix.h"
#include "matrix/matrix_operations.h"

namespace kae {

namespace detail {

namespace impl {

template <unsigned order>
class CoordinateRow;

template <>
class CoordinateRow<1U>
{
public:

  template <class ElemT>
  using ReturnT = kae::Matrix<ElemT, 1U, 1U>;

  template <class GpuGridT, class ElemT>
  static HOST_DEVICE ReturnT<ElemT> get(CudaFloat2T<ElemT>)
  {
    return ReturnT<ElemT>{ static_cast<ElemT>(1) };
  }
};

template <>
class CoordinateRow<2U>
{
public:

  template <class ElemT>
  using ReturnT = kae::Matrix<ElemT, 1U, 3U>;

  template <class GpuGridT, class ElemT>
  static HOST_DEVICE ReturnT<ElemT> get(const CudaFloat2T<ElemT> deltas)
  {
    return ReturnT<ElemT>{ static_cast<ElemT>(1), deltas.x, deltas.y };
  }
};

template <>
class CoordinateRow<3U>
{
public:

  template <class ElemT>
  using ReturnT = kae::Matrix<ElemT, 1U, 6U>;

  template <class GpuGridT, class ElemT>
  static HOST_DEVICE ReturnT<ElemT> get(const CudaFloat2T<ElemT> deltas)
  {
    const auto dn   = deltas.x;
    const auto dtau = deltas.y;
    return ReturnT<ElemT>{ static_cast<ElemT>(1), dn, dtau, dn * dn, dn * dtau, dtau * dtau };
  }
};

} // namespace impl

template <class GpuGridT, class InputMatrixT, class ElemT = typename GpuGridT::ElemType,
          unsigned order = InputMatrixT::rows,
          unsigned degreesOfFreedom = order * (order + 1U) / 2,
          class ReturnT             = kae::Matrix<ElemT, order * order, degreesOfFreedom>>
HOST_DEVICE ReturnT getCoordinatesMatrix(const CudaFloat2T<ElemT> surfacePoint,
                                         const CudaFloat2T<ElemT> normal,
                                         const InputMatrixT &     indexMatrix)
{
  static_assert(InputMatrixT::rows == InputMatrixT::cols, "");

  ReturnT coordinateMatrix;
  for (unsigned i{}; i < order; ++i)
  {
    for(unsigned j{}; j < order; ++j)
    {
      const auto rowIndex = i * order + j;
      const auto gridIndex = indexMatrix(i, j);
      const auto idxX = gridIndex % GpuGridT::nx;
      const auto idxY = gridIndex / GpuGridT::nx;
      const CudaFloat2T<ElemT> globalDeltas{ idxX * GpuGridT::hx - surfacePoint.x, idxY * GpuGridT::hy - surfacePoint.y };
      const CudaFloat2T<ElemT> localDeltas{ TransformCoordinates{}(globalDeltas, normal) };
      setRow(coordinateMatrix, rowIndex, impl::CoordinateRow<order>::template get<GpuGridT, ElemT>(localDeltas));
    }
  }
  return coordinateMatrix;
}

template <class GpuGridT, class InputMatrixT, class GasState,
          class ElemT    = typename GpuGridT::ElemType,
          unsigned order = InputMatrixT::rows,
          class ReturnT  = kae::Matrix<ElemT, order * order, 4U>>
HOST_DEVICE ReturnT getRightHandSideMatrix(const CudaFloat2T<ElemT> normal,
                                           const GasState *         pGasStates,
                                           const InputMatrixT &     indexMatrix)
{
  static_assert(InputMatrixT::rows == InputMatrixT::cols, "");

  ReturnT rhsMatrix;
  for (unsigned i{}; i < order; ++i)
  {
    for (unsigned j{}; j < order; ++j)
    {
      const auto rowIndex = i * order + j;
      const auto gridIndex = indexMatrix(i, j);
      const auto rotatedState = Rotate::get(pGasStates[gridIndex], normal.x, normal.y);
      setRow(rhsMatrix, rowIndex, 
            kae::Matrix<ElemT, 4U, 1U>{rotatedState.rho, rotatedState.ux, rotatedState.uy, rotatedState.p});
    }
  }

  return rhsMatrix;
}

template <class GpuGridT, class InputMatrixT, class GasState,
          class ElemT = typename GpuGridT::ElemType,
          unsigned order = InputMatrixT::rows,
          class ReturnT      = kae::Matrix<ElemT, order * order, 4U>>
HOST_DEVICE ReturnT getRightHandSideMatrix(const CudaFloat2T<ElemT> normal,
                                           const GasState *         pGasStates,
                                           const GasState &         rotatedClosestState,
                                           const InputMatrixT &     indexMatrix)
{
  static_assert(InputMatrixT::rows == InputMatrixT::cols, "");

  ReturnT rhsMatrix;
  const auto leftEigenVectors = DispatchedLeftPrimitiveEigenVectorsX::get(rotatedClosestState);
  for (unsigned i{}; i < order; ++i)
  {
    for (unsigned j{}; j < order; ++j)
    {
      const auto rowIndex = i * order + j;
      const auto gridIndex = indexMatrix(i, j);
      const auto rotatedState = Rotate::get(pGasStates[gridIndex], normal.x, normal.y);
      const auto characteristicsVariables = PrimitiveCharacteristicVariables::get(leftEigenVectors, rotatedState);
      setRow(rhsMatrix, rowIndex, characteristicsVariables);
    }
  }

  return rhsMatrix;
}

} // detail

} // namespace kae
