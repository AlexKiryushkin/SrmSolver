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


template <class T>
struct IsMatrix
{
private:

  template <class U, unsigned i = U::rows + U::cols + U::size, class = typename U::ElemType>
  static std::true_type test(void*);

  template <class U>
  static std::false_type test(...);

public:

  static constexpr bool value = decltype(test<T>(nullptr))::value;
};

template <class MatrixT, typename>
std::ostream& operator<<(std::ostream& os, const MatrixT& matrix)
{
  for (unsigned i{}; i < MatrixT::rows; ++i)
  {
    for (unsigned j{}; j < MatrixT::cols; ++j)
    {
      os << matrix(i, j) << " ";
    }
    os << "\n";
  }
  return os;
}

} // namespace kae
