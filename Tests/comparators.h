#pragma once

#include <gtest/gtest.h>

#define EXPECT_DIM3_EQ(lhs, rhs)  \
  EXPECT_EQ(lhs.x, rhs.x);        \
  EXPECT_EQ(lhs.y, rhs.y);        \
  EXPECT_EQ(lhs.z, rhs.z);        \

#define EXPECT_FLOAT2_NEAR(lhs, rhs, threshold)  \
  EXPECT_NEAR(lhs.x, rhs.x, threshold);          \
  EXPECT_NEAR(lhs.y, rhs.y, threshold);          \

#define EXPECT_FLOAT4_NEAR(lhs, rhs, threshold)  \
  EXPECT_NEAR(lhs.x, rhs.x, threshold);          \
  EXPECT_NEAR(lhs.y, rhs.y, threshold);          \
  EXPECT_NEAR(lhs.z, rhs.z, threshold);          \
  EXPECT_NEAR(lhs.w, rhs.w, threshold);          \

#define EXPECT_GAS_STATE_NEAR(lhs, rhs, threshold)  \
  EXPECT_NEAR(lhs.rho, rhs.rho, threshold);         \
  EXPECT_NEAR(lhs.ux,  rhs.ux,  threshold);         \
  EXPECT_NEAR(lhs.uy,  rhs.uy,  threshold);         \
  EXPECT_NEAR(lhs.p,   rhs.p,   threshold);         \
