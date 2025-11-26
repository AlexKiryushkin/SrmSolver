#pragma once

#include <type_traits>

#include "matrix.h"
#include "wrapper_base.h"

namespace kae {

template <class LhsMatrixT, class RhsMatrixT>
class MultiplyResult : public WrapperBase<MultiplyResult<LhsMatrixT, RhsMatrixT>>
{
public:

  static_assert(std::is_same<typename LhsMatrixT::ElemType, typename RhsMatrixT::ElemType>::value, "");
  static_assert(LhsMatrixT::cols == RhsMatrixT::rows, "");

  using ElemType                   = typename LhsMatrixT::ElemType;
  constexpr static unsigned rows   = LhsMatrixT::rows;
  constexpr static unsigned cols   = RhsMatrixT::cols;
  constexpr static unsigned common = LhsMatrixT::cols;
  constexpr static unsigned size   = rows * cols;
  using CastMatrixType             = Matrix<ElemType, rows, cols>;
  using BaseType                   = WrapperBase<MultiplyResult<LhsMatrixT, RhsMatrixT>>;

  HOST_DEVICE MultiplyResult(const LhsMatrixT & lhsMatrix, const RhsMatrixT & rhsMatrix)
    : m_lhsMatrix{ lhsMatrix }, m_rhsMatrix{ rhsMatrix } {}

  HOST_DEVICE ElemType operator()(unsigned i, unsigned j) const;
  HOST_DEVICE operator CastMatrixType() const noexcept { return BaseType::template cast<CastMatrixType>(); }

private:
  const LhsMatrixT& m_lhsMatrix;
  const RhsMatrixT& m_rhsMatrix;
};

template <class LhsMatrixT, class RhsMatrixT>
HOST_DEVICE auto MultiplyResult<LhsMatrixT, RhsMatrixT>::operator()(unsigned i, unsigned j) const -> ElemType
{
  ElemType elem{};
  for (unsigned k{}; k < common; ++k)
  {
    elem += m_lhsMatrix(i, k) * m_rhsMatrix(k, j);
  }
  return elem;
}

} // namespace kae
