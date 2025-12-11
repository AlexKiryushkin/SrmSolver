#pragma once

#include "cuda_includes.h"

#include "math_utilities.h"
#include "matrix/matrix.h"

namespace kae {

namespace detail {

template <class LhsMatrixT,
          class ElemT = typename LhsMatrixT::ElemType,
          class ReturnT = kae::Matrix<ElemT, LhsMatrixT::rows, LhsMatrixT::cols>>
  HOST_DEVICE ReturnT choleskyDecompositionL(const LhsMatrixT& lhsMatrix)
{
  static_assert(LhsMatrixT::rows == LhsMatrixT::cols, "");

  ReturnT lCholeskyMatrix{};
  for (unsigned i{}; i < LhsMatrixT::rows; ++i)
  {
    ElemT diagonalSubtractor{};
    for (unsigned p{}; p < i; ++p)
    {
      diagonalSubtractor += sqr(lCholeskyMatrix(i, p));
    }
    lCholeskyMatrix(i, i) = std::sqrt(lhsMatrix(i, i) - diagonalSubtractor);
    for (unsigned j{ i + 1 }; j < LhsMatrixT::rows; ++j)
    {
      ElemT elemSubtractor{};
      for (unsigned p{}; p < i; ++p)
      {
        elemSubtractor += lCholeskyMatrix(i, p) * lCholeskyMatrix(j, p);
      }
      lCholeskyMatrix(j, i) = (lhsMatrix(j, i) - elemSubtractor) / lCholeskyMatrix(i, i);
    }
  }
  return lCholeskyMatrix;
}

template <class LhsMatrixT,
          class RhsMatrixT,
          class ElemT = typename LhsMatrixT::ElemType,
          class ReturnT = Matrix<ElemT, RhsMatrixT::rows, RhsMatrixT::cols>>
  HOST_DEVICE ReturnT choleskySolve(const LhsMatrixT& lhsMatrix, const RhsMatrixT & rhsMatrix)
{
  static_assert(LhsMatrixT::rows == LhsMatrixT::cols, "");
  static_assert(LhsMatrixT::rows == RhsMatrixT::rows, "");

  const auto lCholeskyMatrix = choleskyDecompositionL(lhsMatrix);
  ReturnT solution = rhsMatrix;
  for (int i{}; i < LhsMatrixT::rows; ++i)
  {
    for (int j{}; j < i; ++j)
    {
      for (int k{ 0 }; k < RhsMatrixT::cols; ++k)
      {
        const auto value = lCholeskyMatrix(i, j) * solution(j, k);
        solution(i, k) -= value;
      }
    }
    for (int k{ 0 }; k < RhsMatrixT::cols; ++k)
    {
      solution(i, k) /= lCholeskyMatrix(i, i);
    }
  }

  for (int i{ LhsMatrixT::rows - 1U }; i >= 0; --i)
  {
    for (int j{ i + 1 }; j < LhsMatrixT::rows; ++j)
    {
      for (int k{ 0 }; k < RhsMatrixT::cols; ++k)
      {
        solution(i, k) -= lCholeskyMatrix(j, i) * solution(j, k);
      }
    }
    for (int k{ 0 }; k < RhsMatrixT::cols; ++k)
    {
      solution(i, k) /= lCholeskyMatrix(i, i);
    }
  }

  return solution;
}

} // namespace detail

} // namespace kae
