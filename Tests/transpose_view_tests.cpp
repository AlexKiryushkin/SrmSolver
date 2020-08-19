
#include <algorithm>
#include <random>

#include <gtest/gtest.h>

#include <SrmSolver/matrix_operations.h>
#include <SrmSolver/transpose_view.h>

namespace kae_tests {

template <class T>
class transpose_view_tests : public ::testing::Test
{
public:
  template <class ElemT>
  void generate(ElemT * data, unsigned size)
  {
    static thread_local std::mt19937_64 engine{ std::random_device{}() };
    static thread_local std::uniform_real_distribution<ElemT> generator{ static_cast<ElemT>(-1.0), static_cast<ElemT>(1.0) };
    std::generate(data, std::next(data, size), [&]() { return generator(engine); });
  }
};

using TypeParams = ::testing::Types<float, double>;
TYPED_TEST_SUITE(transpose_view_tests, TypeParams);

TYPED_TEST(transpose_view_tests, transpose_view_tests_operator_braces)
{
  constexpr unsigned rows{ 10U };
  constexpr unsigned cols{ 12U };
  using ElemType             = TypeParam;
  using MatrixType           = kae::Matrix<ElemType, rows, cols>;
  using TransposedMatrixType = kae::Matrix<ElemType, cols ,rows>;
  using TransposeViewType    = kae::TransposeView<MatrixType>;

  static_assert(TransposeViewType::cols == MatrixType::rows);
  static_assert(TransposeViewType::rows == MatrixType::cols);
  static_assert(TransposeViewType::size == MatrixType::size);
  static_assert(std::is_same<typename TransposeViewType::ElemType, ElemType>::value);
  static_assert(std::is_same<typename TransposeViewType::CastMatrixType, TransposedMatrixType>::value);
  static_assert(sizeof(TransposeViewType) == sizeof(MatrixType*));

  MatrixType matrix;
  this->generate(matrix.data(), MatrixType::size);

  TransposeViewType transposedView{ matrix };
  for (unsigned i{}; i < rows; ++i)
  {
    for (unsigned j{}; j < cols; ++j)
    {
      EXPECT_EQ(matrix(i, j), transposedView(j, i));
    }
  }
}

TYPED_TEST(transpose_view_tests, transpose_view_tests_cast_operator)
{
  constexpr unsigned rows{ 10U };
  constexpr unsigned cols{ 12U };
  using ElemType             = TypeParam;
  using MatrixType           = kae::Matrix<ElemType, rows, cols>;
  using TransposedMatrixType = kae::Matrix<ElemType, cols, rows>;
  using TransposeViewType    = kae::TransposeView<MatrixType>;

  MatrixType matrix;
  this->generate(matrix.data(), MatrixType::size);
  TransposedMatrixType transposedView = TransposeViewType{ matrix };
  for (unsigned i{}; i < rows; ++i)
  {
    for (unsigned j{}; j < cols; ++j)
    {
      EXPECT_EQ(matrix(i, j), transposedView(j, i));
    }
  }
}

} // namespace kae_tests
