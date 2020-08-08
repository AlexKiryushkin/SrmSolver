#pragma once

#include "cuda_includes.h"
#include "std_includes.h"

namespace kae {

template <class ElemT, unsigned nRows, unsigned nCols>
struct Matrix
{
  using ElemType = ElemT;
  constexpr static unsigned rows = nRows;
  constexpr static unsigned cols = nCols;
  constexpr static unsigned size = rows * cols;

  Matrix() = default;
  template <class... Ts>
  constexpr HOST_DEVICE Matrix(Ts... values) noexcept : m_data{ values... } { static_assert(sizeof...(Ts) == size, ""); }
  explicit constexpr HOST_DEVICE Matrix(const std::initializer_list<std::initializer_list<ElemT>>& elems) noexcept;

  constexpr HOST_DEVICE const ElemT& operator()(unsigned i, unsigned j) const noexcept { return m_data[i * nCols + j]; }
  constexpr HOST_DEVICE ElemT& operator()(unsigned i, unsigned j) noexcept { return m_data[i * nCols + j]; }

  constexpr HOST_DEVICE const ElemType* data() const noexcept { return m_data; }
  constexpr HOST_DEVICE ElemType* data() noexcept { return m_data; }

private:
  ElemT m_data[nRows * nCols];
};

template <class ElemT, unsigned nRows, unsigned nCommon, unsigned nCols>
Matrix<ElemT, nRows, nCols> operator*(const Matrix<ElemT, nRows, nCommon>& lhs, 
                                      const Matrix<ElemT, nCommon, nCols>& rhs);

} // namespace kae

#include  "matrix_def.h"
