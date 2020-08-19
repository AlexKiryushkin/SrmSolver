#pragma once

#include "matrix.h"
#include "wrapper_base.h"

namespace kae {

template <class MatrixT>
class TransposeView : public WrapperBase<TransposeView<MatrixT>>
{
public:

  constexpr static unsigned rows = MatrixT::cols;
  constexpr static unsigned cols = MatrixT::rows;
  constexpr static unsigned size = rows * cols;

  using ElemType       = typename MatrixT::ElemType;
  using CastMatrixType = Matrix<ElemType, rows, cols>;
  using BaseType       = WrapperBase<TransposeView<MatrixT>>;

  explicit TransposeView(const MatrixT & matrix) noexcept : m_matrix{ matrix } {}

  const ElemType& operator()(unsigned i, unsigned j) const noexcept { return m_matrix(j, i); }

  operator CastMatrixType() const noexcept { return BaseType::template cast<CastMatrixType>(); }

private:
  const MatrixT& m_matrix;
};

} // namespace kae
