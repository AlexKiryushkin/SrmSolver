#pragma once

#include "multiply_result.h"
#include "transpose_view.h"

namespace kae {

template <class MatrixT>
HOST_DEVICE TransposeView<MatrixT> transpose(const MatrixT & matrix)
{
  return TransposeView<MatrixT>{ matrix };
}

template <class LhsMatrixT, class RhsMatrixT>
HOST_DEVICE MultiplyResult<LhsMatrixT, RhsMatrixT> operator*(const LhsMatrixT& lhs, const RhsMatrixT& rhs)
{
  return MultiplyResult<LhsMatrixT, RhsMatrixT>{ lhs, rhs };
}
} // namespace kae
