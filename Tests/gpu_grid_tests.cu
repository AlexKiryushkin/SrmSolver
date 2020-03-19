
#include <ratio>

#include <gtest/gtest.h>

#include <SrmSolver/gpu_grid.h>

#include "comparators.h"

namespace tests {

TEST(gpu_grid, gpu_grid_fields)
{
  constexpr unsigned nx{ 70U };
  constexpr unsigned ny{ 30U };
  using LxToType = std::ratio<35, 10>;
  using LyToType = std::ratio<26, 100>;
  using GpuGridType = kae::GpuGrid<nx, ny, LxToType, LyToType>;

  constexpr unsigned goldNx{ 70U };
  constexpr unsigned goldNy{ 30U };
  constexpr unsigned goldN{ 2100U };
  constexpr float goldLx = 3.5f;
  constexpr float goldLy = 0.26f;
  constexpr float goldHx = goldLx / (goldNx - 1);
  constexpr float goldHy = goldLy / (goldNy - 1);
  constexpr dim3 goldBlockSize{ 32U, 16U };
  constexpr dim3 goldGridSize{ 3, 2 };
  constexpr dim3 goldSharedMemory{ 38U, 22U };
  constexpr unsigned goldSmExtension = 3U;
  constexpr unsigned goldSmSize = 836U;
  constexpr unsigned goldSmSizeBytes = 3344U;

  EXPECT_EQ(GpuGridType::nx, goldNx);
  EXPECT_EQ(GpuGridType::ny, goldNy);
  EXPECT_EQ(GpuGridType::n, goldN);
  EXPECT_EQ(GpuGridType::lx, goldLx);
  EXPECT_EQ(GpuGridType::ly, goldLy);
  EXPECT_EQ(GpuGridType::hx, goldHx);
  EXPECT_EQ(GpuGridType::hy, goldHy);
  EXPECT_EQ(GpuGridType::hxReciprocal, 1.0f / goldHx);
  EXPECT_EQ(GpuGridType::hyReciprocal, 1.0f / goldHy);
  EXPECT_DIM3_EQ(GpuGridType::blockSize, goldBlockSize);
  EXPECT_DIM3_EQ(GpuGridType::gridSize, goldGridSize);
  EXPECT_DIM3_EQ(GpuGridType::sharedMemory, goldSharedMemory);
  EXPECT_EQ(GpuGridType::smExtension, goldSmExtension);
  EXPECT_EQ(GpuGridType::smSize, goldSmSize);
  EXPECT_EQ(GpuGridType::smSizeBytes, goldSmSizeBytes);
}

} // namespace tests
