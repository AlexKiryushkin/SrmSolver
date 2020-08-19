
#include <gtest/gtest.h>

#include <SrmSolver/multiply_result.h>

namespace kae_tests {

template <class T>
class multiply_result_tests : public ::testing::Test
{
  
};

using TypeParams = ::testing::Types<float, double>;
TYPED_TEST_SUITE(multiply_result_tests, TypeParams);

TYPED_TEST(multiply_result_tests, multiply_result_tests_constructor)
{
  constexpr unsigned rows{ 3U };
  constexpr unsigned common{ 4U };
  constexpr unsigned cols{ 2U };
  using ElemType         = TypeParam;
  using LhsMatrixType    = kae::Matrix<ElemType, rows, common>;
  using RhsMatrixType    = kae::Matrix<ElemType, common, cols>;
  using ResultMatrixType = kae::Matrix<ElemType, rows, cols>;
  using MultiplyResultType = kae::MultiplyResult<LhsMatrixType, RhsMatrixType>;

  static_assert(MultiplyResultType::rows   == rows, "");
  static_assert(MultiplyResultType::common == common, "");
  static_assert(MultiplyResultType::cols   == cols, "");
  static_assert(MultiplyResultType::size   == rows * cols, "");
  static_assert(std::is_same<typename MultiplyResultType::CastMatrixType, ResultMatrixType>::value, "");

  LhsMatrixType lhsMatrix{
    static_cast<ElemType>(1),  static_cast<ElemType>(2),   static_cast<ElemType>(3),  static_cast<ElemType>(4),
    static_cast<ElemType>(5),  static_cast<ElemType>(6),   static_cast<ElemType>(7),  static_cast<ElemType>(8),
    static_cast<ElemType>(9),  static_cast<ElemType>(10),  static_cast<ElemType>(11), static_cast<ElemType>(12) };

  RhsMatrixType rhsMatrix{ static_cast<ElemType>(1),  static_cast<ElemType>(-2),
                           static_cast<ElemType>(-3), static_cast<ElemType>(4),
                           static_cast<ElemType>(5),  static_cast<ElemType>(-6),
                           static_cast<ElemType>(7),  static_cast<ElemType>(-8) };

  const MultiplyResultType result{ lhsMatrix, rhsMatrix };
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

TYPED_TEST(multiply_result_tests, multiply_result_tests_cast_operator)
{
  constexpr unsigned rows{ 3U };
  constexpr unsigned common{ 4U };
  constexpr unsigned cols{ 2U };
  using ElemType = TypeParam;
  using LhsMatrixType = kae::Matrix<ElemType, rows, common>;
  using RhsMatrixType = kae::Matrix<ElemType, common, cols>;
  using ResultMatrixType = kae::Matrix<ElemType, rows, cols>;
  using MultiplyResultType = kae::MultiplyResult<LhsMatrixType, RhsMatrixType>;

  static_assert(MultiplyResultType::rows == rows, "");
  static_assert(MultiplyResultType::common == common, "");
  static_assert(MultiplyResultType::cols == cols, "");
  static_assert(MultiplyResultType::size == rows * cols, "");
  static_assert(std::is_same<typename MultiplyResultType::CastMatrixType, ResultMatrixType>::value, "");

  LhsMatrixType lhsMatrix{
    static_cast<ElemType>(1),  static_cast<ElemType>(2),   static_cast<ElemType>(3),  static_cast<ElemType>(4),
    static_cast<ElemType>(5),  static_cast<ElemType>(6),   static_cast<ElemType>(7),  static_cast<ElemType>(8),
    static_cast<ElemType>(9),  static_cast<ElemType>(10),  static_cast<ElemType>(11), static_cast<ElemType>(12) };

  RhsMatrixType rhsMatrix{ static_cast<ElemType>(1),  static_cast<ElemType>(-2),
                           static_cast<ElemType>(-3), static_cast<ElemType>(4),
                           static_cast<ElemType>(5),  static_cast<ElemType>(-6),
                           static_cast<ElemType>(7),  static_cast<ElemType>(-8) };

  const ResultMatrixType result = MultiplyResultType{ lhsMatrix, rhsMatrix };

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
