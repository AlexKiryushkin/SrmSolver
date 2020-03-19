
#include <gtest/gtest.h>

#include <SrmSolver/math_utilities.h>

#include "comparators.h"

namespace tests {

TEST(absmin, absmin_functional_1)
{
  constexpr float value1{ 1.0f };
  constexpr float value2{ -2.0f };

  const auto res1 = kae::absmin(value1, value2);
  const auto res2 = kae::absmin(value2, value1);

  EXPECT_EQ(res1, value1);
  EXPECT_EQ(res2, value1);
}

TEST(absmin, absmin_functional_2)
{
  constexpr float value1{ -1.0f };
  constexpr float value2{ 2.0f };

  const auto res1 = kae::absmin(value1, value2);
  const auto res2 = kae::absmin(value2, value1);

  EXPECT_EQ(res1, value1);
  EXPECT_EQ(res2, value1);
}

TEST(absmin, absmin_functional_3)
{
  constexpr float value1{ -1.0f };
  constexpr float value2{ 2.0f };
  constexpr float value3{ -5.0f };

  const auto res1 = kae::absmin(value1, value2, value3);
  const auto res2 = kae::absmin(value1, value3, value2);
  const auto res3 = kae::absmin(value2, value1, value3);
  const auto res4 = kae::absmin(value2, value3, value1);
  const auto res5 = kae::absmin(value3, value1, value2);
  const auto res6 = kae::absmin(value3, value2, value1);

  EXPECT_EQ(res1, value1);
  EXPECT_EQ(res2, value1);
  EXPECT_EQ(res3, value1);
  EXPECT_EQ(res4, value1);
  EXPECT_EQ(res5, value1);
  EXPECT_EQ(res6, value1);
}

TEST(absmax, absmax_functional_1)
{
  constexpr float value1{ 1.0f };
  constexpr float value2{ -2.0f };

  const auto res1 = kae::absmax(value1, value2);
  const auto res2 = kae::absmax(value2, value1);

  EXPECT_EQ(res1, value2);
  EXPECT_EQ(res2, value2);
}

TEST(absmax, absmax_functional_2)
{
  constexpr float value1{ -1.0f };
  constexpr float value2{ 2.0f };

  const auto res1 = kae::absmax(value1, value2);
  const auto res2 = kae::absmax(value2, value1);

  EXPECT_EQ(res1, value2);
  EXPECT_EQ(res2, value2);
}

TEST(absmax, absmax_functional_3)
{
  constexpr float value1{ -1.0f };
  constexpr float value2{ 2.0f };
  constexpr float value3{ -5.0f };

  const auto res1 = kae::absmax(value1, value2, value3);
  const auto res2 = kae::absmax(value1, value3, value2);
  const auto res3 = kae::absmax(value2, value1, value3);
  const auto res4 = kae::absmax(value2, value3, value1);
  const auto res5 = kae::absmax(value3, value1, value2);
  const auto res6 = kae::absmax(value3, value2, value1);

  EXPECT_EQ(res1, value3);
  EXPECT_EQ(res2, value3);
  EXPECT_EQ(res3, value3);
  EXPECT_EQ(res4, value3);
  EXPECT_EQ(res5, value3);
  EXPECT_EQ(res6, value3);
}

TEST(sqr, sqr_functional_1)
{
  constexpr float value{ 2.0f };
  const auto res = kae::sqr(value);

  constexpr float goldRes{ 4.0f };
  constexpr float threshold{ 1e-6f };
  EXPECT_NEAR(res, goldRes, threshold);
}

TEST(sqr, sqr_functional_2)
{
  constexpr float value{ -2.0f };
  const auto res = kae::sqr(value);

  constexpr float goldRes{ 4.0f };
  constexpr float threshold{ 1e-6f };
  EXPECT_NEAR(res, goldRes, threshold);
}

TEST(elemwise_max, elemwise_max_functional_1)
{
  constexpr float2 value1{ 2.0f, 3.0f };
  constexpr float2 value2{ 1.0f, 5.0f };
  const auto res = kae::ElemwiseMax{}(value1, value2);

  constexpr float2 goldRes{ 2.0f, 5.0f };
  constexpr float threshold{ 1e-6f };
  EXPECT_FLOAT2_NEAR(res, goldRes, threshold);
}

TEST(elemwise_max, elemwise_abs_max_functional_1)
{
  constexpr float4 value1{ 2.0f, 3.0f, -7.0f, 9.0f };
  constexpr float4 value2{ 1.0f, 5.0f, 4.0f, -10.0f };
  const auto res = kae::ElemwiseAbsMax{}(value1, value2);

  constexpr float4 goldRes{ 2.0f, 5.0f, -7.0f, -10.0f };
  constexpr float threshold{ 1e-6f };
  EXPECT_FLOAT4_NEAR(res, goldRes, threshold);
}

} // namespace tests
