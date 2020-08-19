
#include <gtest/gtest.h>

#include <SrmSolver/matrix.h>

namespace kae_tests {

template <class T>
class matrix_tests : public ::testing::Test
{
public:
  using ElemType = T;

  template <unsigned nRows, unsigned nCols>
  using MatrixType = kae::Matrix<ElemType, nRows, nCols>;
};

using TypeParams = ::testing::Types<float, double>;
TYPED_TEST_SUITE(matrix_tests, TypeParams);

TYPED_TEST(matrix_tests, matrix_tests_default_constructor)
{
  using tf = TestFixture;
  using MatrixType = typename tf::template MatrixType<6U, 6U>;
  static_assert(std::is_trivially_default_constructible<MatrixType>::value, 
                "Matrix is not trivially default constructible");

  MatrixType matrix{};
  const auto allAreZero = std::all_of(matrix.data(), 
                                      std::next(matrix.data(), MatrixType::size), 
                                      [](const auto elem) { return elem == 0; });
  EXPECT_TRUE(allAreZero);
}

TYPED_TEST(matrix_tests, matrix_tests_variadic_constructor)
{
  using tf           = TestFixture;
  using ElemType     = typename tf::ElemType;
  constexpr auto dim = 3U;
  using MatrixType   = typename tf::template MatrixType<dim, dim>;

  MatrixType matrix{ static_cast<ElemType>(1), static_cast<ElemType>(2), static_cast<ElemType>(3),
                     static_cast<ElemType>(4), static_cast<ElemType>(5), static_cast<ElemType>(6),
                     static_cast<ElemType>(7), static_cast<ElemType>(8), static_cast<ElemType>(9) };
  for (unsigned i{}; i < dim; ++i)
  {
    for (unsigned j{}; j < dim; ++j)
    {
      const auto value = static_cast<ElemType>(i * dim + j + 1U);
      EXPECT_EQ(matrix(i, j), value);
    }
  }
}

TYPED_TEST(matrix_tests, matrix_tests_initializer_list_constructor)
{
  using tf            = TestFixture;
  using ElemType      = typename tf::ElemType;
  constexpr auto rows = 4U;
  constexpr auto cols = 3U;
  using MatrixType   = typename tf::template MatrixType<rows, cols>;

  MatrixType matrix{ {static_cast<ElemType>(1), static_cast<ElemType>(2), static_cast<ElemType>(3)},
                     {static_cast<ElemType>(4), static_cast<ElemType>(5), static_cast<ElemType>(6)},
                     {static_cast<ElemType>(7), static_cast<ElemType>(8), static_cast<ElemType>(9)} ,
                     {static_cast<ElemType>(10), static_cast<ElemType>(11), static_cast<ElemType>(12)} };
  for (unsigned i{}; i < rows; ++i)
  {
    for (unsigned j{}; j < cols; ++j)
    {
      const auto value = static_cast<ElemType>(i * cols + j + 1U);
      EXPECT_EQ(matrix(i, j), value);
    }
  }
}

TYPED_TEST(matrix_tests, matrix_tests_operator_round_braces)
{
  using tf = TestFixture;
  using ElemType = typename tf::ElemType;
  constexpr auto rows = 4U;
  constexpr auto cols = 3U;
  using MatrixType = typename tf::template MatrixType<rows, cols>;

  MatrixType matrix{};
  for (unsigned i{}; i < rows; ++i)
  {
    for (unsigned j{}; j < cols; ++j)
    {
      const auto value = static_cast<ElemType>(i * cols + j + 1U);
      matrix(i, j) = value;
      EXPECT_EQ(matrix(i, j), value);
    }
  }
}

} // namespace kae_tests
