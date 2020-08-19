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

} // namespace kae
