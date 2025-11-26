#pragma once

#include "matrix.h"
#include "wrapper_base.h"

namespace kae {

template <class MatrixT, unsigned nRows, unsigned nCols, unsigned startRow = 0U, unsigned startCol = 0U>
class Submatrix : public WrapperBase<Submatrix<MatrixT, nRows, nCols, startRow, startCol>>
{
public:

  using ElemType                 = typename MatrixT::ElemType;
  constexpr static unsigned rows = nRows;
  constexpr static unsigned cols = nCols;
  constexpr static unsigned size = rows * cols;
  using CastMatrixType           = Matrix<ElemType, rows, cols>;
  using BaseType                 = WrapperBase<Submatrix<MatrixT, nRows, nCols, startRow, startCol>>;

  static_assert(MatrixT::rows >= nRows + startRow, "");
  static_assert(MatrixT::cols >= nCols + startCol, "");

  explicit HOST_DEVICE Submatrix(const MatrixT & matrix) : m_matrix(matrix) {}

  HOST_DEVICE ElemType operator()(unsigned i, unsigned j) const { return m_matrix(startRow + i, startCol + j); }
  HOST_DEVICE operator CastMatrixType() const noexcept { return BaseType::template cast<CastMatrixType>(); }

private:
  const MatrixT& m_matrix;
};

} // namespace kae
