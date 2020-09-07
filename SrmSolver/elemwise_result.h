#pragma once

#include "std_includes.h"

#include "matrix.h"
#include "wrapper_base.h"

namespace kae {

template <class LhsMatrixT, class RhsMatrixT, class FunctorT>
class BinaryElemwiseResult : public WrapperBase<BinaryElemwiseResult<LhsMatrixT, RhsMatrixT, FunctorT>>
{
public:

  static_assert(LhsMatrixT::rows == RhsMatrixT::rows, "");
  static_assert(LhsMatrixT::cols == RhsMatrixT::cols, "");
  static_assert(std::is_same<typename LhsMatrixT::ElemType, typename RhsMatrixT::ElemType>::value, "");

  using ElemType                 = typename LhsMatrixT::ElemType;
  constexpr static unsigned rows = LhsMatrixT::rows;
  constexpr static unsigned cols = LhsMatrixT::cols;
  constexpr static unsigned size = rows * cols;
  using CastMatrixType           = Matrix<ElemType, rows, cols>;
  using BaseType                 = WrapperBase<BinaryElemwiseResult<LhsMatrixT, RhsMatrixT, FunctorT>>;

  HOST_DEVICE BinaryElemwiseResult(const LhsMatrixT& lhsMatrix, const RhsMatrixT& rhsMatrix)
    : m_lhsMatrix{ lhsMatrix }, m_rhsMatrix{ rhsMatrix } {}

  HOST_DEVICE ElemType operator()(unsigned i, unsigned j) const { return FunctorT{}(m_lhsMatrix(i, j), m_rhsMatrix(i, j)); }
  HOST_DEVICE operator CastMatrixType() const noexcept { return BaseType::template cast<CastMatrixType>(); }

private:
  const LhsMatrixT& m_lhsMatrix;
  const RhsMatrixT& m_rhsMatrix;
};

template <class MatrixT, class FunctorT>
class UnaryElemwiseResult : public WrapperBase<UnaryElemwiseResult<MatrixT, FunctorT>>
{
public:

  using ElemType                 = typename MatrixT::ElemType;
  constexpr static unsigned rows = MatrixT::rows;
  constexpr static unsigned cols = MatrixT::cols;
  constexpr static unsigned size = rows * cols;
  using CastMatrixType           = Matrix<ElemType, rows, cols>;
  using BaseType                 = WrapperBase<UnaryElemwiseResult<MatrixT, FunctorT>>;

  HOST_DEVICE UnaryElemwiseResult(const MatrixT& matrix)
    : m_matrix{ matrix } {}

  HOST_DEVICE ElemType operator()(unsigned i, unsigned j) const { return FunctorT{}(m_matrix(i, j)); }
  HOST_DEVICE operator CastMatrixType() const noexcept { return BaseType::template cast<CastMatrixType>(); }

private:
  const MatrixT& m_matrix;
};

} // namespace kae
