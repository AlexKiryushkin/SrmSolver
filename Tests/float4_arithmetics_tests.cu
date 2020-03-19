
#include <gtest/gtest.h>

#include <SrmSolver/float4_arithmetics.h>

#include "comparators.h"

namespace tests {

TEST(float4_arithmetics, float4_arithmetics_operator_plus)
{
  constexpr float4 lhs{ 1.0f, 2.0f, 3.0f, 4.0f };
  constexpr float4 rhs{ 5.0f, -6.0f, 8.0f, -7.0f };
  const float4 res{ lhs + rhs };

  constexpr float4 goldRes{ 6.0f, -4.0f, 11.0f, -3.0f };
  constexpr float threshold{ 1e-6f };
  EXPECT_FLOAT4_NEAR(res, goldRes, threshold);
}

TEST(float4_arithmetics, float4_arithmetics_operator_minus)
{
  constexpr float4 lhs{ 1.0f, 2.0f, 3.0f, 4.0f };
  constexpr float4 rhs{ 5.0f, -6.0f, 8.0f, -7.0f };
  const float4 res{ lhs - rhs };

  constexpr float4 goldRes{ -4.0f, 8.0f, -5.0f, 11.0f };
  constexpr float threshold{ 1e-6f };
  EXPECT_FLOAT4_NEAR(res, goldRes, threshold);
}

TEST(float4_arithmetics, float4_arithmetics_operator_multiply)
{
  constexpr float lhs{ 2.0f };
  constexpr float4 rhs{ 5.0f, -6.0f, 8.0f, -7.0f };
  const float4 res{ lhs * rhs };

  constexpr float4 goldRes{ 10.0f, -12.0f, 16.0f, -14.0f };
  constexpr float threshold{ 1e-6f };
  EXPECT_FLOAT4_NEAR(res, goldRes, threshold);
}

} // namespace tests
