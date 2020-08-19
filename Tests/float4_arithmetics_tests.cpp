
#include <gtest/gtest.h>

#include <SrmSolver/float4_arithmetics.h>

#include "comparators.h"

namespace kae_tests {

template <class T>
class float4_arithmetics : public ::testing::Test {};

using TypeParams = ::testing::Types<float, double>;
TYPED_TEST_SUITE(float4_arithmetics, TypeParams);

TYPED_TEST(float4_arithmetics, float4_arithmetics_operator_plus)
{
  using ElemType = std::conditional_t<std::is_same<TypeParam, float>::value, float4, double4>;

  constexpr ElemType lhs{ 1.0, 2.0, 3.0, 4.0 };
  constexpr ElemType rhs{ 5.0, -6.0, 8.0, -7.0 };
  const ElemType res{ lhs + rhs };

  constexpr ElemType goldRes{ 6.0, -4.0, 11.0, -3.0 };
  constexpr TypeParam threshold{ std::is_same<TypeParam, float>::value ? static_cast<TypeParam>(1e-6) :
                                                                         static_cast<TypeParam>(1e-14) };
  EXPECT_FLOAT4_NEAR(res, goldRes, threshold);
}

TYPED_TEST(float4_arithmetics, float4_arithmetics_operator_minus)
{
  using ElemType = std::conditional_t<std::is_same<TypeParam, float>::value, float4, double4>;

  constexpr ElemType lhs{ 1.0, 2.0, 3.0, 4.0 };
  constexpr ElemType rhs{ 5.0, -6.0, 8.0, -7.0 };
  const ElemType res{ lhs - rhs };

  constexpr ElemType goldRes{ -4.0, 8.0, -5.0, 11.0 };
  constexpr TypeParam threshold{ std::is_same<TypeParam, float>::value ? static_cast<TypeParam>(1e-6) :
                                                                         static_cast<TypeParam>(1e-14) };
  EXPECT_FLOAT4_NEAR(res, goldRes, threshold);
}

TYPED_TEST(float4_arithmetics, float4_arithmetics_operator_multiply)
{
  using ElemType = std::conditional_t<std::is_same<TypeParam, float>::value, float4, double4>;

  constexpr TypeParam lhs{ 2.0 };
  constexpr ElemType rhs{ 5.0, -6.0, 8.0, -7.0 };
  const ElemType res{ lhs * rhs };

  constexpr ElemType goldRes{ 10.0, -12.0, 16.0, -14.0 };
  constexpr TypeParam threshold{ std::is_same<TypeParam, float>::value ? static_cast<TypeParam>(1e-6) :
                                                                         static_cast<TypeParam>(1e-14) };
  EXPECT_FLOAT4_NEAR(res, goldRes, threshold);
}

} // namespace kae_tests
