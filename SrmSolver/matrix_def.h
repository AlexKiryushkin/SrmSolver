#pragma once

namespace kae {

template <class ElemT, unsigned nRows, unsigned nCols>
constexpr Matrix<ElemT, nRows, nCols>::Matrix(const std::initializer_list<std::initializer_list<ElemT>>& elems) noexcept
  : m_data{}
{
  unsigned i{};
  for (const auto & row : elems)
  {
    if (i >= nRows)
    {
      continue;
    }

    unsigned j{};
    for (const auto & elem : row)
    {
      if (j >= nCols)
      {
        continue;
      }

      (*this)(i, j) = elem;
      ++j;
    }
    ++i;
  }
}

template<class ElemT, unsigned nRows, unsigned nCommon, unsigned nCols>
Matrix<ElemT, nRows, nCols> operator*(const Matrix<ElemT, nRows, nCommon>& lhs, 
                                      const Matrix<ElemT, nCommon, nCols>& rhs)
{
  Matrix<ElemT, nRows, nCols> result;
  for (unsigned i{}; i < nRows; ++i)
  {
    for (unsigned j{}; j < nCols; ++j)
    {
      ElemT elem{};
      for (unsigned k{}; k < nCommon; ++k)
      {
        elem += lhs(i, k) * rhs(k, j);
      }
      result(i, j) = elem;
    }
  }
  return result;
}

} // namespace kae
