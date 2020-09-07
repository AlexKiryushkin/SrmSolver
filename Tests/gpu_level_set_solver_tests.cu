
#include <gtest/gtest.h>

#include <SrmSolver/gpu_grid.h>
#include <SrmSolver/gpu_level_set_solver.h>

#include "circle_shape.h"

#ifndef _DEBUG

namespace kae_tests {

template <class T>
class gpu_level_set_solver : public ::testing::Test
{
public:

  constexpr static unsigned nx{ std::tuple_element_t<1U, T>::value };
  constexpr static unsigned ny{ std::tuple_element_t<1U, T>::value };
  constexpr static unsigned smExtension{ 3U };
  using ElemType           = std::tuple_element_t<0U, T>;
  using LxToType           = std::ratio<4, 1>;
  using LyToType           = std::ratio<4, 1>;
  using GpuGridType        = kae::GpuGrid<nx, ny, LxToType, LyToType, smExtension, ElemType>;
  using GpuMatrixType      = kae::GpuMatrix<GpuGridType, ElemType>;
  using ShapeType          = CircleShape<GpuGridType>;
  using LevelSetSolverType = kae::GpuLevelSetSolver<GpuGridType, ShapeType>;
};

using TypeParams = ::testing::Types<
  std::tuple<float,  std::integral_constant<unsigned, 100U>  >,
  std::tuple<float,  std::integral_constant<unsigned, 500U>  >,
  std::tuple<float,  std::integral_constant<unsigned, 1000U> >,
  std::tuple<double, std::integral_constant<unsigned, 100U>  >,
  std::tuple<double, std::integral_constant<unsigned, 500U>  >
>;
TYPED_TEST_SUITE(gpu_level_set_solver, TypeParams);

TYPED_TEST(gpu_level_set_solver, gpu_level_set_solver_constructor_simple)
{
  using tf              = TestFixture;
  using ElemT           = typename tf::ElemType;
  using ShapeT          = typename tf::ShapeType;
  using LevelSetSolverT = typename tf::LevelSetSolverType;

  LevelSetSolverT solver{};
  auto && deviceValues = solver.currState().values();
  const auto matrixSize = deviceValues.size();

  std::vector<ElemT> hostValues(matrixSize);
  thrust::copy(std::begin(deviceValues), std::end(deviceValues), std::begin(hostValues));

  const ShapeT shape;
  for (unsigned i = 0; i < tf::nx; ++i)
  {
    for (unsigned j = 0; j < tf::ny; ++j)
    {
      const auto index = j * tf::nx + i;
      const auto value = shape(i, j);
      const auto threshold = 10 * std::max(static_cast<ElemT>(1.0), value) * std::numeric_limits<ElemT>::epsilon();
      EXPECT_NEAR(hostValues[index], value, threshold);
    }
  }
}

TYPED_TEST(gpu_level_set_solver, gpu_level_set_solver_constructor_reinitialize)
{
  using tf              = TestFixture;
  using ElemT           = typename tf::ElemType;
  using GpuGridT        = typename tf::GpuGridType;
  using ShapeT          = typename tf::ShapeType;
  using LevelSetSolverT = typename tf::LevelSetSolverType;

  LevelSetSolverT solver{ShapeT{}, tf::nx };
  auto&& deviceValues = solver.currState().values();
  const auto matrixSize = deviceValues.size();

  std::vector<ElemT> hostValues(matrixSize);
  thrust::copy(std::begin(deviceValues), std::end(deviceValues), std::begin(hostValues));

  for (unsigned i = 0U; i < tf::nx; ++i)
  {
    for (unsigned j = 0U; j < tf::ny; ++j)
    {
      const auto index = j * tf::nx + i;
      if (std::fabs(hostValues[index]) < 10 * GpuGridT::hx)
      {
        const auto value = ShapeT::reinitializedValue(i, j);
        const auto threshold = 5 * std::max(static_cast<ElemT>(1.0), value) * GpuGridT::hx * GpuGridT::hx;
        EXPECT_NEAR(hostValues[index], value, threshold);
      }
    }
  }
}

TYPED_TEST(gpu_level_set_solver, gpu_level_set_solver_reinitialize)
{
  using tf              = TestFixture;
  using ElemT           = typename tf::ElemType;
  using GpuGridT        = typename tf::GpuGridType;
  using ShapeT          = typename tf::ShapeType;
  using LevelSetSolverT = typename tf::LevelSetSolverType;

  LevelSetSolverT solver{ ShapeT{} };
  solver.reinitialize(tf::nx);
  auto&& deviceValues = solver.currState().values();
  const auto matrixSize = deviceValues.size();

  std::vector<ElemT> hostValues(matrixSize);
  thrust::copy(std::begin(deviceValues), std::end(deviceValues), std::begin(hostValues));

  for (unsigned i = 0U; i < tf::nx; ++i)
  {
    for (unsigned j = 0U; j < tf::ny; ++j)
    {
      const auto index = j * tf::nx + i;
      if (std::fabs(hostValues[index]) < 10 * GpuGridT::hx)
      {
        const auto value = ShapeT::reinitializedValue(i, j);
        const auto threshold = 5 * std::max(static_cast<ElemT>(1.0), value) * GpuGridT::hx * GpuGridT::hx;
        EXPECT_NEAR(hostValues[index], value, threshold);
      }
    }
  }
}

TYPED_TEST(gpu_level_set_solver, gpu_level_set_solver_integrate_overload_a)
{
  using tf = TestFixture;
  using ElemT = typename tf::ElemType;
  using GpuGridT = typename tf::GpuGridType;
  using GpuMatrixT = typename tf::GpuMatrixType;
  using ShapeT = typename tf::ShapeType;
  using LevelSetSolverT = typename tf::LevelSetSolverType;

  LevelSetSolverT solver{ ShapeT{}, tf::nx };
  GpuMatrixT velocities{ static_cast<ElemT>(1.0) };
  const auto dt = solver.integrateInTime(velocities, tf::nx / 10U);

  auto&& deviceValues = solver.currState().values();
  const auto matrixSize = deviceValues.size();

  std::vector<ElemT> hostValues(matrixSize);
  thrust::copy(std::begin(deviceValues), std::end(deviceValues), std::begin(hostValues));

  for (unsigned i = 0U; i < tf::nx; ++i)
  {
    for (unsigned j = 0U; j < tf::ny; ++j)
    {
      const auto index = j * tf::nx + i;
      if (std::fabs(hostValues[index]) < 10 * GpuGridT::hx)
      {
        const auto value = ShapeT::integratedValue(i, j, dt);
        const auto threshold = 5 * std::max(static_cast<ElemT>(1.0), value) * GpuGridT::hx * GpuGridT::hx;
        EXPECT_NEAR(hostValues[index], value, threshold);
      }
    }
  }
}

TYPED_TEST(gpu_level_set_solver, gpu_level_set_solver_integrate_overload_b)
{
  using tf              = TestFixture;
  using ElemT           = typename tf::ElemType;
  using GpuGridT        = typename tf::GpuGridType;
  using GpuMatrixT      = typename tf::GpuMatrixType;
  using ShapeT          = typename tf::ShapeType;
  using LevelSetSolverT = typename tf::LevelSetSolverType;

  LevelSetSolverT solver{ ShapeT{}, tf::nx };
  GpuMatrixT velocities{ static_cast<ElemT>(1.0) };
  const auto integrateTime{ static_cast<ElemT>(0.1) };
  const auto dt = solver.integrateInTime(velocities, integrateTime);
  EXPECT_EQ(dt, integrateTime);

  auto&& deviceValues = solver.currState().values();
  const auto matrixSize = deviceValues.size();

  std::vector<ElemT> hostValues(matrixSize);
  thrust::copy(std::begin(deviceValues), std::end(deviceValues), std::begin(hostValues));

  for (unsigned i = 0U; i < tf::nx; ++i)
  {
    for (unsigned j = 0U; j < tf::ny; ++j)
    {
      const auto index = j * tf::nx + i;
      if (std::fabs(hostValues[index]) < 10 * GpuGridT::hx)
      {
        const auto value = ShapeT::integratedValue(i, j, dt);
        const auto threshold = 5 * std::max(static_cast<ElemT>(1.0), value) * GpuGridT::hx * GpuGridT::hx;
        EXPECT_NEAR(hostValues[index], value, threshold);
      }
    }
  }
}

} // namespace kae_tests

#endif
