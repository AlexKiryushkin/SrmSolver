
#include <ratio>

#include <gtest/gtest.h>

#include <SrmSolver/gpu_grid.h>

#include "comparators.h"

namespace kae_tests {

template <class T>
class gpu_grid : public ::testing::Test {};

using TypeParams = ::testing::Types<float, double>;
TYPED_TEST_SUITE(gpu_grid, TypeParams);

TYPED_TEST(gpu_grid, gpu_grid_fields)
{
  using ElemType = TypeParam;
  constexpr unsigned nx{ 70U };
  constexpr unsigned ny{ 30U };
  constexpr unsigned smExtension{ 3U };
  constexpr unsigned blockSizeX{ 32U * sizeof(float) / sizeof(ElemType) };
  constexpr unsigned blockSizeY{ 8U * sizeof(float) / sizeof(ElemType) };
  using LxToType = std::ratio<35, 10>;
  using LyToType = std::ratio<26, 100>;
  using GpuGridType = kae::GpuGrid<nx, ny, LxToType, LyToType, smExtension, ElemType, blockSizeX, blockSizeY>;

  constexpr unsigned goldNx{ 70U };
  constexpr unsigned goldNy{ 30U };
  constexpr unsigned goldN{ 2100U };
  constexpr ElemType goldLx{ static_cast<ElemType>(3.5) };
  constexpr ElemType goldLy{ static_cast<ElemType>(0.26) };
  constexpr ElemType goldHx{ goldLx / (goldNx - 1) };
  constexpr ElemType goldHy{ goldLy / (goldNy - 1) };
  constexpr dim3 goldBlockSize{ blockSizeX, blockSizeY };
  constexpr dim3 goldGridSize{ (goldNx + goldBlockSize.x - 1) / goldBlockSize.x, (goldNy + goldBlockSize.y - 1) / goldBlockSize.y };
  constexpr dim3 goldSharedMemory{ goldBlockSize.x + 2 * smExtension, goldBlockSize.y + 2 * smExtension };
  constexpr unsigned goldSmExtension = 3U;
  constexpr unsigned goldSmSize = goldSharedMemory.x * goldSharedMemory.y;
  constexpr unsigned goldSmSizeBytes = goldSmSize * sizeof(ElemType);

  EXPECT_EQ(GpuGridType::nx, goldNx);
  EXPECT_EQ(GpuGridType::ny, goldNy);
  EXPECT_EQ(GpuGridType::n, goldN);
  EXPECT_EQ(GpuGridType::lx, goldLx);
  EXPECT_EQ(GpuGridType::ly, goldLy);
  EXPECT_EQ(GpuGridType::hx, goldHx);
  EXPECT_EQ(GpuGridType::hy, goldHy);
  EXPECT_EQ(GpuGridType::hxReciprocal, 1 / goldHx);
  EXPECT_EQ(GpuGridType::hyReciprocal, 1 / goldHy);
  EXPECT_DIM3_EQ(GpuGridType::blockSize, goldBlockSize);
  EXPECT_DIM3_EQ(GpuGridType::gridSize, goldGridSize);
  EXPECT_DIM3_EQ(GpuGridType::sharedMemory, goldSharedMemory);
  EXPECT_EQ(GpuGridType::smExtension, goldSmExtension);
  EXPECT_EQ(GpuGridType::smSize, goldSmSize);
  EXPECT_EQ(GpuGridType::smSizeBytes, goldSmSizeBytes);
}

} // namespace kae_tests
