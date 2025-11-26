#pragma once

#include "cuda_includes.h"
#include "std_includes.h"

#include "elemwise_result.h"
#include "math_utilities.h"
#include "multiply_result.h"
#include "transpose_view.h"

namespace kae {

template <class MatrixT, typename = std::enable_if_t<IsMatrixV<MatrixT>>>
HOST_DEVICE TransposeView<MatrixT> transpose(const MatrixT & matrix)
{
  return TransposeView<MatrixT>{ matrix };
}

template <class LhsMatrixT, class RhsMatrixT,
          typename = std::enable_if_t<IsMatrixV<LhsMatrixT> && IsMatrixV<RhsMatrixT>>>
HOST_DEVICE MultiplyResult<LhsMatrixT, RhsMatrixT> operator*(const LhsMatrixT& lhs, const RhsMatrixT& rhs)
{
  return MultiplyResult<LhsMatrixT, RhsMatrixT>{ lhs, rhs };
}

template <class LhsMatrixT, class RhsMatrixT,
          typename = std::enable_if_t<IsMatrixV<LhsMatrixT> && IsMatrixV<RhsMatrixT>>>
HOST_DEVICE auto operator+(const LhsMatrixT& lhs, const RhsMatrixT& rhs)
{
  return BinaryElemwiseResult<LhsMatrixT, RhsMatrixT, thrust::plus<typename LhsMatrixT::ElemType>>{ lhs, rhs };
}

template <class LhsMatrixT, class RhsMatrixT,
          typename = std::enable_if_t<IsMatrixV<LhsMatrixT> && IsMatrixV<RhsMatrixT>>>
HOST_DEVICE auto operator-(const LhsMatrixT& lhs, const RhsMatrixT& rhs)
{
  return BinaryElemwiseResult<LhsMatrixT, RhsMatrixT, thrust::minus<typename LhsMatrixT::ElemType>>{ lhs, rhs };
}

template <class MatrixT, typename = std::enable_if_t<IsMatrixV<MatrixT>>>
HOST_DEVICE auto cwiseAbs(const MatrixT & matrix)
{
  return UnaryElemwiseResult<MatrixT, detail::AbsFunctor>(matrix);
}

template <class MatrixT, typename = std::enable_if_t<IsMatrixV<MatrixT>>>
HOST_DEVICE auto maxCoeff(const MatrixT & matrix)
{
  using ElemType = typename MatrixT::ElemType;
  ElemType result = std::numeric_limits<ElemType>::lowest();
  for (unsigned i{}; i < MatrixT::rows; ++i)
  {
    for (unsigned j{}; j < MatrixT::cols; ++j)
    {
      result = thrust::max(result, matrix(i, j));
    }
  }
  return result;
}

template <class LhsMatrixT, class RhsMatrixT>
HOST_DEVICE void setRow(LhsMatrixT & matrix, unsigned row, const RhsMatrixT & rhsMatrix)
{
  for (unsigned col{}; col < LhsMatrixT::cols; ++col)
  {
    matrix(row, col) = rhsMatrix(col);
  }
}

template <class LhsMatrixT, class RhsMatrixT>
HOST_DEVICE void setCol(LhsMatrixT& matrix, unsigned col, const RhsMatrixT& rhsMatrix)
{
  for (unsigned row{}; row < LhsMatrixT::rows; ++row)
  {
    matrix(row, col) = rhsMatrix(row);
  }
}

} // namespace kae
