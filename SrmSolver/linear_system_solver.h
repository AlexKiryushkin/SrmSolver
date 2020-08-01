#pragma once

#include "cuda_includes.h"
#include "eigen_includes.h"

#include "math_utilities.h"

namespace kae {

namespace detail {

template <class LhsMatrixT,
          class ElemT = typename LhsMatrixT::Scalar,
          class ReturnT = Eigen::Matrix<ElemT, LhsMatrixT::RowsAtCompileTime, LhsMatrixT::ColsAtCompileTime>>
  __host__ __device__ ReturnT choleskyDecompositionL(const LhsMatrixT& lhsMatrix)
{
  static_assert(LhsMatrixT::RowsAtCompileTime == LhsMatrixT::ColsAtCompileTime, "");

  ReturnT lCholeskyMatrix = ReturnT::Zero();
  for (unsigned i{}; i < LhsMatrixT::RowsAtCompileTime; ++i)
  {
    ElemT diagonalSubtractor{};
    for (unsigned p{}; p < i; ++p)
    {
      diagonalSubtractor += sqr(lCholeskyMatrix(i, p));
    }
    lCholeskyMatrix(i, i) = std::sqrt(lhsMatrix(i, i) - diagonalSubtractor);
    for (unsigned j{ i + 1 }; j < LhsMatrixT::RowsAtCompileTime; ++j)
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
          class ElemT = typename LhsMatrixT::Scalar,
          class ReturnT = Eigen::Matrix<ElemT, RhsMatrixT::RowsAtCompileTime, RhsMatrixT::ColsAtCompileTime>>
  __host__ __device__ ReturnT choleskySolve(const LhsMatrixT& lhsMatrix, const RhsMatrixT & rhsMatrix)
{
  static_assert(LhsMatrixT::RowsAtCompileTime == LhsMatrixT::ColsAtCompileTime, "");
  static_assert(LhsMatrixT::RowsAtCompileTime == RhsMatrixT::RowsAtCompileTime, "");

  const auto lCholeskyMatrix = choleskyDecompositionL(lhsMatrix);
  ReturnT solution = rhsMatrix;

  for (int i{}; i < LhsMatrixT::RowsAtCompileTime; ++i)
  {
    for (int j{}; j < i; ++j)
    {
      for (int k{ 0 }; k < RhsMatrixT::ColsAtCompileTime; ++k)
      {
        const auto value = lCholeskyMatrix(i, j) * solution(j, k);
        solution(i, k) -= value;
      }
    }
    for (int k{ 0 }; k < RhsMatrixT::ColsAtCompileTime; ++k)
    {
      solution(i, k) /= lCholeskyMatrix(i, i);
    }
  }

  for (int i{ LhsMatrixT::RowsAtCompileTime - 1U }; i >= 0; --i)
  {
    for (int j{ i + 1 }; j < LhsMatrixT::RowsAtCompileTime; ++j)
    {
      for (int k{ 0 }; k < RhsMatrixT::ColsAtCompileTime; ++k)
      {
        solution(i, k) -= lCholeskyMatrix(j, i) * solution(j, k);
      }
    }
    for (int k{ 0 }; k < RhsMatrixT::ColsAtCompileTime; ++k)
    {
      solution(i, k) /= lCholeskyMatrix(i, i);
    }
  }

  return solution;
}

} // namespace detail

} // namespace kae
