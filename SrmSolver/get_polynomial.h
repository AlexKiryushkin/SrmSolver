#pragma once

#include "cuda_includes.h"

#include "cuda_float_types.h"
#include "matrix.h"
#include "matrix_operations.h"
#include "submatrix.h"

namespace kae {

namespace detail {

template <class GpuGridT, class MatrixT, class ElemT = typename MatrixT::ElemType,
          class = std::enable_if_t<MatrixT::rows == 1U || MatrixT::rows == 3U || MatrixT::rows == 6U>>
kae::Matrix<ElemT, MatrixT::cols, 1U> getPolynomialWeight(const MatrixT & matrix)
{
  kae::Matrix<ElemT, MatrixT::cols, 1U> weightMatrix{};
  constexpr auto hSqr = sqr(GpuGridT::hx);
  for (unsigned i{}; i < MatrixT::cols; ++i)
  {
    switch (MatrixT::rows)
    {
      case 1U:
      {
        weightMatrix(i) = 2 * hSqr;
        break;
      }
      case 3U:
      {
        weightMatrix(i) = (sqr(matrix(1U, i)) + sqr(matrix(2U, i))) * hSqr;
        break;
      }
      case 6U:
      {
        weightMatrix(i) = hSqr * (sqr(matrix(1U, i)) + sqr(matrix(2U, i)) + 
          7 * hSqr * sqr(matrix(3U, i)) / 3 + 4 * hSqr * sqr(matrix(4U, i)) / 3 + 7 * hSqr * sqr(matrix(5U, i)) / 3);
        break;
      }
      default:
      {
        break;
      }
    }
  }
  return weightMatrix;
}

template <class GpuGridT, class MatrixT, class ElemT = typename MatrixT::ElemType,
          class = std::enable_if_t<MatrixT::cols == 1U>>
kae::Matrix<ElemT, MatrixT::rows, 3U> getWenoCoefficients(const MatrixT & bettas0, 
                                                          const MatrixT & bettas1, 
                                                          const MatrixT & bettas2)
{
  kae::Matrix<ElemT, MatrixT::rows, 3U> wenoCoefficients{};

  constexpr auto d0 = 2 * sqr(GpuGridT::hx);
  constexpr auto d1 = 2 * GpuGridT::hx;
  constexpr auto d2 = static_cast<ElemT>(1) - d0 - d1;
  constexpr auto epsilon = std::numeric_limits<ElemT>::epsilon();

  for (unsigned i{}; i < MatrixT::rows; ++i)
  {
    const auto alpha0 = d0 / sqr(bettas0(i) + epsilon);
    const auto alpha1 = d1 / sqr(bettas1(i) + epsilon);
    const auto alpha2 = d2 / sqr(bettas2(i) + epsilon);
    const auto sum = alpha0 + alpha1 + alpha2;
    wenoCoefficients(i, 0U) = alpha0 / sum;
    wenoCoefficients(i, 1U) = alpha1 / sum;
    wenoCoefficients(i, 2U) = alpha2 / sum;
  }
  return wenoCoefficients;
}

template <class GpuGridT, class MatrixT, class ElemT = typename MatrixT::ElemType,
          class = std::enable_if_t<MatrixT::cols == 1U>>
kae::Matrix<ElemT, MatrixT::rows, 2U> getWenoCoefficients(const MatrixT & bettas0, 
                                                          const MatrixT & bettas1)
{
  kae::Matrix<ElemT, MatrixT::rows, 2U> wenoCoefficients{};

  constexpr auto d0 = 2 * sqr(GpuGridT::hx);
  constexpr auto d1 = static_cast<ElemT>(1) - d0;
  constexpr auto epsilon = std::numeric_limits<ElemT>::epsilon();

  for (unsigned i{}; i < MatrixT::rows; ++i)
  {
    const auto alpha0 = d0 / sqr(bettas0(i) + epsilon);
    const auto alpha1 = d1 / sqr(bettas1(i) + epsilon);
    const auto sum = alpha0 + alpha1;
    wenoCoefficients(i, 0U) = alpha0 / sum;
    wenoCoefficients(i, 1U) = alpha1 / sum;
  }
  return wenoCoefficients;
}

template <class GpuGridT, class GasStateT, class ElemT = typename GpuGridT::ElemType, class InputMatrixT>
HOST_DEVICE auto getPolynomial(const CudaFloat2T<ElemT> surfacePoint,
                               const CudaFloat2T<ElemT> normal,
                               const GasStateT*         pGasValues,
                               const InputMatrixT&      indexMatrix)
{
  static_assert(InputMatrixT::rows == InputMatrixT::cols, "");

  const auto lhsMatrix = getCoordinatesMatrix<GpuGridT>(surfacePoint, normal, indexMatrix);
  const auto rhsMatrix = getRightHandSideMatrix<GpuGridT>(normal, pGasValues, indexMatrix);
  const auto A = transpose(lhsMatrix) * lhsMatrix;
  const auto b = transpose(lhsMatrix) * rhsMatrix;
  return choleskySolve(A, b);
}

template <class GpuGridT, class GasStateT, class ElemT = typename GpuGridT::ElemType,
          class InputMatrixT,
          unsigned order = InputMatrixT::rows,
          unsigned degreesOfFreedom = order * (order + 1U) / 2>
HOST_DEVICE auto getWenoPolynomial(const CudaFloat2T<ElemT> surfacePoint,
                                   const CudaFloat2T<ElemT> normal,
                                   const GasStateT*         pGasValues,
                                   const InputMatrixT&      indexMatrix)
{
  static_assert(InputMatrixT::rows == InputMatrixT::cols, "");

  const auto subMatrix1 = Submatrix<InputMatrixT, 1U, 1U>{ indexMatrix };
  const auto subMatrix2 = Submatrix<InputMatrixT, 2U, 2U>{ indexMatrix };
  const auto p0 = getPolynomial<GpuGridT>(surfacePoint, normal, pGasValues, subMatrix1);
  const auto p1 = getPolynomial<GpuGridT>(surfacePoint, normal, pGasValues, subMatrix2);
  //const auto p2 = getPolynomial<GpuGridT>(surfacePoint, normal, pGasValues, indexMatrix);

  const auto betta0 = getPolynomialWeight<GpuGridT>(p0);
  const auto betta1 = getPolynomialWeight<GpuGridT>(p1);
  //const auto betta2 = getPolynomialWeight<GpuGridT>(p2);

  const auto wenoCoefficients = getWenoCoefficients<GpuGridT>(betta0, betta1/*, betta2*/);
  kae::Matrix<ElemT, degreesOfFreedom, 4U> wenoPolynomial{};
  for (unsigned i{}; i < degreesOfFreedom; ++i)
  {
    for (unsigned j{}; j < 4U; ++j)
    {
      constexpr auto zero = static_cast<ElemT>(0);
      const auto coefficient0 = (i < decltype(p0)::rows) ? p0(i, j) : zero;
      const auto coefficient1 = (i < decltype(p1)::rows) ? p1(i, j) : zero;
      //const auto coefficient2 = p2(i, j);

      wenoPolynomial(i, j) = coefficient0 * wenoCoefficients(j, 0U) +
                             coefficient1 * wenoCoefficients(j, 1U);/* +
                             coefficient2 * wenoCoefficients(j, 2U);*/
    }
  }
  return wenoPolynomial;
}

} // namespace detail

} // namespace kae
