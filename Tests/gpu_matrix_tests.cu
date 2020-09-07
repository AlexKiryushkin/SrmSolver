
#include <gtest/gtest.h>

#include <vector>

#pragma warning(push, 0)
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/logical.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning(pop)

#include <SrmSolver/gpu_matrix.h>
#include <SrmSolver/to_float.h>
#include <SrmSolver/gpu_grid.h>

#ifndef _DEBUG

namespace kae_tests {

template <class ValueT>
struct EqualToValue
{
  template <class ElemT>
  HOST_DEVICE bool operator()(ElemT value) const
  {
    return value == kae::detail::ToFloatV<ValueT, ElemT>;
  }
};

template <class T>
class gpu_matrix : public ::testing::Test {};

using TypeParams = ::testing::Types<float, double>;
TYPED_TEST_SUITE(gpu_matrix, TypeParams);

TYPED_TEST(gpu_matrix, gpu_matrix_constructor_a)
{
  using ElemType = TypeParam;
  constexpr unsigned nx{ 70U };
  constexpr unsigned ny{ 30U };
  constexpr unsigned smExtension{ 3U };
  using LxToType = std::ratio<35, 10>;
  using LyToType = std::ratio<26, 100>;
  using GpuGridType = kae::GpuGrid<nx, ny, LxToType, LyToType, smExtension, ElemType>;
  using ValueType = std::ratio<1, 1>;
  kae::GpuMatrix<GpuGridType, ElemType> matrix{ kae::detail::ToFloatV<ValueType, ElemType> };

  auto && deviceValues = matrix.values();
  auto allZeros = thrust::all_of(std::begin(deviceValues), std::end(deviceValues), EqualToValue<ValueType>{});
  EXPECT_TRUE(allZeros);
}

struct Initializer
{
  HOST_DEVICE float operator()(unsigned i, unsigned j) const
  {
    return static_cast<float>(i * i + j * j);
  }
};

TYPED_TEST(gpu_matrix, gpu_matrix_constructor_b)
{
  using ElemType = TypeParam;
  constexpr unsigned nx{ 70U };
  constexpr unsigned ny{ 30U };
  constexpr unsigned smExtension{ 3U };
  using LxToType = std::ratio<35, 10>;
  using LyToType = std::ratio<26, 100>;
  using GpuGridType = kae::GpuGrid<nx, ny, LxToType, LyToType, smExtension, ElemType>;
  kae::GpuMatrix<GpuGridType, ElemType> matrix{ Initializer{} };

  auto && deviceValues = matrix.values();
  auto matrixSize = deviceValues.size();

  std::vector<ElemType> hostValues(matrixSize);
  thrust::copy(std::begin(deviceValues), std::end(deviceValues), std::begin(hostValues));

  Initializer initializer;
  for (unsigned i = 0; i < nx; ++i)
  {
    for (unsigned j = 0; j < ny; ++j)
    {
      const auto index = j * nx + i;
      EXPECT_EQ(hostValues[index], initializer(i, j));
    }
  }
}

TYPED_TEST(gpu_matrix, gpu_matrix_values_non_const)
{
  using ElemType = TypeParam;
  constexpr unsigned nx{ 45U };
  constexpr unsigned ny{ 20U };
  constexpr unsigned smExtension{ 3U };
  using LxToType = std::ratio<35, 10>;
  using LyToType = std::ratio<26, 100>;
  using GpuGridType = kae::GpuGrid<nx, ny, LxToType, LyToType, smExtension, ElemType>;
  kae::GpuMatrix<GpuGridType, ElemType> matrix{ Initializer{} };

  auto && deviceValues = matrix.values();
  auto matrixSize = deviceValues.size();
  EXPECT_EQ(matrixSize, nx * ny);
}

TYPED_TEST(gpu_matrix, gpu_matrix_values_const)
{
  using ElemType = TypeParam;
  constexpr unsigned nx{ 45U };
  constexpr unsigned ny{ 20U };
  constexpr unsigned smExtension{ 3U };
  using LxToType = std::ratio<35, 10>;
  using LyToType = std::ratio<26, 100>;
  using GpuGridType = kae::GpuGrid<nx, ny, LxToType, LyToType, smExtension, ElemType>;
  const kae::GpuMatrix<GpuGridType, ElemType> matrix{ Initializer{} };

  const auto & deviceValues = matrix.values();
  auto matrixSize = deviceValues.size();
  EXPECT_EQ(matrixSize, nx * ny);
}

} // namespace kae_tests

#endif
