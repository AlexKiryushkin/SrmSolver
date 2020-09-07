#pragma once

#include "cuda_includes.h"

#include "cuda_float_types.h"
#include "matrix.h"
#include "matrix_operations.h"

namespace kae {

namespace detail {


template <class GpuGridT, unsigned order, class GasStateT, class ElemT = typename GpuGridT::ElemType,
          class InputMatrixT = kae::Matrix<unsigned, order, order>,
          unsigned degreesOfFreedom = order * (order + 1U) / 2>
HOST_DEVICE auto getPolynomial(const CudaFloat2T<ElemT> surfacePoint,
                               const CudaFloat2T<ElemT> normal,
                               const GasStateT*         pGasValues,
                               const InputMatrixT&      indexMatrix)
{
  const auto lhsMatrix = getCoordinatesMatrix<GpuGridT, order>(surfacePoint, normal, indexMatrix);
  const auto rhsMatrix = getRightHandSideMatrix<GpuGridT, order>(normal, pGasValues, indexMatrix);
  const auto A = transpose(lhsMatrix) * lhsMatrix;
  const auto b = transpose(lhsMatrix) * rhsMatrix;
  return choleskySolve(A, b);
}

template <class GpuGridT, unsigned order, class GasStateT, class ElemT = typename GpuGridT::ElemType,
          class InputMatrixT = kae::Matrix<unsigned, order, order>,
          unsigned degreesOfFreedom = order * (order + 1U) / 2>
HOST_DEVICE auto getWenoPolynomial(const CudaFloat2T<ElemT> surfacePoint,
                                   const CudaFloat2T<ElemT> normal,
                                   const GasStateT*         pGasValues,
                                   const InputMatrixT&      indexMatrix)
{
  //const auto p1 = getPolynomial<GpuGridT, 1U>(surfacePoint, normal, pGasValues, indexMatrix.template block<1U, 1U>());
  //const auto p2 = getPolynomial<GpuGridT, 2U>(surfacePoint, normal, pGasValues, indexMatrix.template block<2U, 2U>());
  auto p3 = getPolynomial<GpuGridT, 3U>(surfacePoint, normal, pGasValues, indexMatrix);

  //const auto w1 = kae::Matrix<ElemT, 1U, 4U>::Constant(GpuGridT::hx * GpuGridT::hx);
  //const auto w2 = GpuGridT::hx * GpuGridT::hx;

  return p3;
}

} // namespace detail

} // namespace kae
