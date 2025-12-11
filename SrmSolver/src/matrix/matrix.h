#pragma once

#include "cuda_includes.h"
#include "std_includes.h"

namespace kae {

template <class ElemT, unsigned nRows, unsigned nCols>
class Matrix
{
public:
  using ElemType = ElemT;
  constexpr static unsigned rows = nRows;
  constexpr static unsigned cols = nCols;
  constexpr static unsigned size = rows * cols;
  using MatrixType = Matrix<ElemType, rows, cols>;

  Matrix() = default;
  template <class... Ts, class = std::enable_if_t<sizeof...(Ts) == size>>
  explicit constexpr HOST_DEVICE Matrix(Ts... values) noexcept : m_data{ values... } {}
  explicit constexpr HOST_DEVICE Matrix(const std::initializer_list<std::initializer_list<ElemT>>& elems) noexcept;
  explicit constexpr Matrix(const ElemT* pData) noexcept { std::copy(pData, std::next(pData, size), m_data); }

  constexpr HOST_DEVICE const ElemT& operator()(unsigned i, unsigned j) const noexcept { return m_data[i * nCols + j]; }
  constexpr HOST_DEVICE ElemT& operator()(unsigned i, unsigned j) noexcept { return m_data[i * nCols + j]; }

  template <class MatrixT = MatrixType, class = std::enable_if_t<MatrixT::rows == 1U || MatrixT::cols == 1U>>
  constexpr HOST_DEVICE const ElemT& operator()(unsigned i) const noexcept { return m_data[i]; }
  template <class MatrixT = MatrixType, class = std::enable_if_t<MatrixT::rows == 1U || MatrixT::cols == 1U>>
  constexpr HOST_DEVICE ElemT& operator()(unsigned i) noexcept { return m_data[i]; }

  constexpr HOST_DEVICE const ElemType* data() const noexcept { return m_data; }
  constexpr HOST_DEVICE ElemType* data() noexcept { return m_data; }

public:

  static MatrixType identity()
  {
    MatrixType matrix{};
    constexpr auto minDim = std::min(rows, cols);
    for (unsigned i{}; i < rows; ++i)
    {
      for (unsigned j{}; j < cols; ++j)
      {
        matrix(i, j) = (i == j) ? static_cast<ElemType>(1.0) : static_cast<ElemType>(0.0);
      }
    }
    return matrix;
  }

  static MatrixType random()
  {
    static thread_local std::mt19937_64 engine{ 777U };
    static thread_local std::uniform_real_distribution<ElemType> generator{ static_cast<ElemType>(-1.0) };

    MatrixType matrix{};
    auto* pData = matrix.data();
    std::generate(pData, std::next(pData, size), [&]() { return generator(engine); });
    return matrix;
  }

private:
  ElemT m_data[nRows * nCols];
};

template<class T>
struct IsMatrix;

template <class T>
constexpr bool IsMatrixV = IsMatrix<T>::value;

template <class MatrixT, typename = std::enable_if_t<IsMatrixV<MatrixT>>>
std::ostream& operator<<(std::ostream& os, const MatrixT& matrix);

} // namespace kae

#include  "matrix_def.h"
