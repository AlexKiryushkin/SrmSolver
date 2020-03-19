
#include <gtest/gtest.h>

#include <vector>

#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/logical.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <SrmSolver/gpu_matrix.h>
#include <SrmSolver/to_float.h>
#include <SrmSolver/gpu_grid.h>

namespace tests {

template <class ValueT>
struct EqualToValue
{
  __host__ __device__ bool operator()(float value) const
  {
    return value == kae::detail::ToFloatV<ValueT>;
  }
};

TEST(gpu_matrix, gpu_matrix_constructor_a)
{
  constexpr unsigned nx{ 70U };
  constexpr unsigned ny{ 30U };
  using LxToType = std::ratio<35, 10>;
  using LyToType = std::ratio<26, 100>;
  using GpuGridType = kae::GpuGrid<nx, ny, LxToType, LyToType>;
  using ValueType = std::ratio<1, 1>;
  kae::GpuMatrix<GpuGridType, float> matrix{ kae::detail::ToFloatV<ValueType> };

  auto && deviceValues = matrix.values();
  auto allZeros = thrust::all_of(std::begin(deviceValues), std::end(deviceValues), EqualToValue<ValueType>{});
  EXPECT_TRUE(allZeros);
}

struct Initializer
{
  __host__ __device__ float operator()(unsigned i, unsigned j) const
  {
    return static_cast<float>(i * i + j * j);
  }
};

TEST(gpu_matrix, gpu_matrix_constructor_b)
{
  constexpr unsigned nx{ 70U };
  constexpr unsigned ny{ 30U };
  using LxToType = std::ratio<35, 10>;
  using LyToType = std::ratio<26, 100>;
  using GpuGridType = kae::GpuGrid<nx, ny, LxToType, LyToType>;
  kae::GpuMatrix<GpuGridType, float> matrix{ Initializer{} };

  auto && deviceValues = matrix.values();
  auto matrixSize = deviceValues.size();

  std::vector<float> hostValues(matrixSize);
  thrust::copy(std::begin(deviceValues), std::end(deviceValues), std::begin(hostValues));

  Initializer initializer;
  for (unsigned i = 0; i < nx; ++i)
  {
    for (unsigned j = 0; j < ny; ++j)
    {
      auto index = j * nx + i;
      EXPECT_EQ(hostValues[index], initializer(i, j));
    }
  }
}

TEST(gpu_matrix, gpu_matrix_values_non_const)
{
  constexpr unsigned nx{ 45U };
  constexpr unsigned ny{ 20U };
  using LxToType = std::ratio<35, 10>;
  using LyToType = std::ratio<26, 100>;
  using GpuGridType = kae::GpuGrid<nx, ny, LxToType, LyToType>;
  kae::GpuMatrix<GpuGridType, float> matrix{ Initializer{} };

  auto && deviceValues = matrix.values();
  auto matrixSize = deviceValues.size();
  EXPECT_EQ(matrixSize, nx * ny);
}

TEST(gpu_matrix, gpu_matrix_values_const)
{
  constexpr unsigned nx{ 45U };
  constexpr unsigned ny{ 20U };
  using LxToType = std::ratio<35, 10>;
  using LyToType = std::ratio<26, 100>;
  using GpuGridType = kae::GpuGrid<nx, ny, LxToType, LyToType>;
  const kae::GpuMatrix<GpuGridType, float> matrix{ Initializer{} };

  const auto & deviceValues = matrix.values();
  auto matrixSize = deviceValues.size();
  EXPECT_EQ(matrixSize, nx * ny);
}

} // namespace tests
