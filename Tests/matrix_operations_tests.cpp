
#include <random>

#include <gtest/gtest.h>

#include <SrmSolver/matrix_operations.h>

namespace kae_tests {

template <class T>
class matrix_operations_tests : public ::testing::Test
{
public:
  template <class ElemT>
  void generate(ElemT* data, unsigned size)
  {
    static thread_local std::mt19937_64 engine{ std::random_device{}() };
    static thread_local std::uniform_real_distribution<ElemT> generator{ static_cast<ElemT>(-1.0), static_cast<ElemT>(1.0) };
    std::generate(data, std::next(data, size), [&]() { return generator(engine); });
  }
};

using TypeParams = ::testing::Types<float, double>;
TYPED_TEST_SUITE(matrix_operations_tests, TypeParams);

TYPED_TEST(matrix_operations_tests, matrix_operations_tests_transpose)
{
  constexpr unsigned rows{ 9U };
  constexpr unsigned cols{ 7U };
  using ElemType = TypeParam;
  using MatrixType = kae::Matrix<ElemType, rows, cols>;

  MatrixType matrix;
  this->generate(matrix.data(), MatrixType::size);
  const auto transposedView = kae::transpose(matrix);
  static_assert(sizeof(transposedView) == sizeof(void*), "");
  for (unsigned i{}; i < rows; ++i)
  {
    for (unsigned j{}; j < cols; ++j)
    {
      EXPECT_EQ(matrix(i, j), transposedView(j, i));
    }
  }

  const auto doubleTransposedView = kae::transpose(transposedView);
  static_assert(sizeof(doubleTransposedView) == sizeof(void*), "");
  for (unsigned i{}; i < rows; ++i)
  {
    for (unsigned j{}; j < cols; ++j)
    {
      EXPECT_EQ(matrix(i, j), doubleTransposedView(i, j));
    }
  }
}

TYPED_TEST(matrix_operations_tests, matrix_operations_tests_multiply)
{
  constexpr unsigned rows{ 3U };
  constexpr unsigned common{ 4U };
  constexpr unsigned cols{ 2U };
  using ElemType         = TypeParam;
  using LhsMatrixType    = kae::Matrix<ElemType, rows, common>;
  using RhsMatrixType    = kae::Matrix<ElemType, common, cols>;
  using ResultMatrixType = kae::Matrix<ElemType, rows, cols>;
  LhsMatrixType lhsMatrix{
    static_cast<ElemType>(1),  static_cast<ElemType>(2),   static_cast<ElemType>(3),  static_cast<ElemType>(4),
    static_cast<ElemType>(5),  static_cast<ElemType>(6),   static_cast<ElemType>(7),  static_cast<ElemType>(8),
    static_cast<ElemType>(9),  static_cast<ElemType>(10),  static_cast<ElemType>(11), static_cast<ElemType>(12) };

  RhsMatrixType rhsMatrix{ static_cast<ElemType>(1),  static_cast<ElemType>(-2),
                           static_cast<ElemType>(-3), static_cast<ElemType>(4),
                           static_cast<ElemType>(5),  static_cast<ElemType>(-6),
                           static_cast<ElemType>(7), static_cast<ElemType>(-8)
  };

  const auto result = lhsMatrix * rhsMatrix;
  static_assert(sizeof(result) == 2 * sizeof(void*), "");

  constexpr ResultMatrixType goldResult{ static_cast<ElemType>(38), static_cast<ElemType>(-44),
                                         static_cast<ElemType>(78), static_cast<ElemType>(-92),
                                         static_cast<ElemType>(118), static_cast<ElemType>(-140), };

  constexpr auto threshold = std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                    static_cast<ElemType>(1e-15);

  for (unsigned i{}; i < rows; ++i)
  {
    for (unsigned j{}; j < cols; ++j)
    {
      EXPECT_NEAR(result(i, j), goldResult(i, j), threshold);
    }
  }
}

} // namespace kae_tests
