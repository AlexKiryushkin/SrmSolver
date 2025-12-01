#pragma once

#include "std_includes.h"

#include "gas_state.h"
#include "gpu_grid.h"
#include "gpu_level_set_solver.h"
#include "gpu_srm_solver.h"
#include "shapes.h"
#include "shape_types.h"

namespace kae {

template <EShapeType ShapeType, class ElemT>
struct ShapeSolverTypes;

template<class ElemT>
struct ShapeSolverTypes<EShapeType::eDualThrustShape, ElemT>
{
  constexpr static unsigned nx{ 800U + 1U };
  constexpr static unsigned ny{ 200U + 1U };
  using LxToType = std::ratio<400, 1000>;
  using LyToType = std::ratio<100, 1000>;
  using GpuGridType = GpuGrid<nx, ny, LxToType, LyToType, 3U, ElemT>;
  using ShapeType = kae::SrmDualThrust<GpuGridType>;
  constexpr static ElemT hx = GpuGridType::hx;
  constexpr static ElemT hy = GpuGridType::hy;
  constexpr static bool stepsAreSame = ((hx > hy) ? (hx - hy < 1e-8f) : (hy - hx < 1e-8f));
  static_assert(stepsAreSame, "Grid steps are different!");

  using GasStateType = GasState<ElemT>;

  using LevelSetSolverType = GpuLevelSetSolver<ElemT, ShapeType>;
  using SrmSolverType = GpuSrmSolver<ShapeType, GasStateType>;

  GpuGridT<ElemT> grid = GpuGridT<ElemT>(nx, ny, detail::ToFloatV<LxToType, ElemT>, detail::ToFloatV<LyToType, ElemT>, 3U);
  PhysicalPropertiesData<ElemT> physicalProperties{ static_cast<ElemT>(0.1052),
                                                    static_cast<ElemT>(-3.135),
                                                    static_cast<ElemT>(2950.0),
                                                    static_cast<ElemT>(1580),
                                                    static_cast<ElemT>(101325),
                                                    static_cast<ElemT>(1.2),
                                                    static_cast<ElemT>(1629.13),
                                                    ShapeType::getFCritical(),
                                                    ShapeType::getInitialSBurn()};
  GasStateType initialGasState{ static_cast<ElemT>(1.0),
                                static_cast<ElemT>(0.0),
                                static_cast<ElemT>(0.0),
                                physicalProperties.P0 };
};

template<class ElemT>
struct ShapeSolverTypes<EShapeType::eNozzleLessShape, ElemT>
{
  constexpr static unsigned nx{ 1410U + 1U };
  constexpr static unsigned ny{ 190U + 1U };
  using LxToType            = std::ratio<1410, 1000>;
  using LyToType            = std::ratio<190, 1000>;
  using GpuGridType         = GpuGrid<nx, ny, LxToType, LyToType, 3U, ElemT>;
  using ShapeType           = kae::SrmShapeNozzleLess<GpuGridType>;
  constexpr static ElemT hx = GpuGridType::hx;
  constexpr static ElemT hy = GpuGridType::hy;
  constexpr static bool stepsAreSame = ((hx > hy) ? (hx - hy < 1e-8f) : (hy - hx < 1e-8f));
  static_assert(stepsAreSame, "Grid steps are different!");

  using GasStateType = GasState<ElemT>;

  using LevelSetSolverType = GpuLevelSetSolver<ElemT, ShapeType>;
  using SrmSolverType      = GpuSrmSolver<ShapeType, GasStateType>;

  GpuGridT<ElemT> grid = GpuGridT<ElemT>(nx, ny, detail::ToFloatV<LxToType, ElemT>, detail::ToFloatV<LyToType, ElemT>, 3U);
  PhysicalPropertiesData<ElemT> physicalProperties{ static_cast<ElemT>(0.41),
                                                  static_cast<ElemT>(-0.096446),
                                                  static_cast<ElemT>(2950.0),
                                                  static_cast<ElemT>(1700),
                                                  static_cast<ElemT>(101325),
                                                  static_cast<ElemT>(1.23),
                                                  static_cast<ElemT>(1800),
                                                  ShapeType::getFCritical(),
                                                  ShapeType::getInitialSBurn() };

  GasStateType initialGasState{ static_cast<ElemT>(1.0),
                                static_cast<ElemT>(0.0),
                                static_cast<ElemT>(0.0),
                                physicalProperties.P0 };
};

template<class ElemT>
struct ShapeSolverTypes<EShapeType::eWithUmbrellaShape, ElemT>
{
  constexpr static unsigned nx{ 820U + 1U };
  constexpr static unsigned ny{ 300U + 1U };
  using LxToType = std::ratio<3280, 1000>;
  using LyToType = std::ratio<1200, 1000>;
  using GpuGridType = GpuGrid<nx, ny, LxToType, LyToType, 3U, ElemT>;
  using ShapeType = kae::SrmShapeWithUmbrella<GpuGridType>;
  constexpr static ElemT hx = GpuGridType::hx;
  constexpr static ElemT hy = GpuGridType::hy;
  constexpr static bool stepsAreSame = ((hx > hy) ? (hx - hy < 1e-8f) : (hy - hx < 1e-8f));
  static_assert(stepsAreSame, "Grid steps are different!");

  using GasStateType = GasState<ElemT>;

  using LevelSetSolverType = GpuLevelSetSolver<ElemT, ShapeType>;
  using SrmSolverType = GpuSrmSolver<ShapeType, GasStateType>;

  GpuGridT<ElemT> grid = GpuGridT<ElemT>(nx, ny, detail::ToFloatV<LxToType, ElemT>, detail::ToFloatV<LyToType, ElemT>, 3U);
  PhysicalPropertiesData<ElemT> physicalProperties{ static_cast<ElemT>(0.5),
                                                    static_cast<ElemT>(-0.00534),
                                                    static_cast<ElemT>(3900.0),
                                                    static_cast<ElemT>(1700),
                                                    static_cast<ElemT>(101325),
                                                    static_cast<ElemT>(1.18),
                                                    static_cast<ElemT>(2628),
                                                    ShapeType::getFCritical(),
                                                    ShapeType::getInitialSBurn() };

  GasStateType initialGasState{ static_cast<ElemT>(0.5),
                                static_cast<ElemT>(0.0),
                                static_cast<ElemT>(0.0),
                                physicalProperties.P0 };
};

template<class ElemT>
struct ShapeSolverTypes<EShapeType::eFlushMountedNozzle, ElemT>
{
  constexpr static unsigned nx{ 2000U / 2U + 1U };
  constexpr static unsigned ny{ 1000U / 2U + 1U };
  using LxToType = std::ratio<2000, 1000>;
  using LyToType = std::ratio<1000, 1000>;
  using GpuGridType = GpuGrid<nx, ny, LxToType, LyToType, 3U, ElemT>;
  using ShapeType = kae::SrmFlushMountedNozzle<GpuGridType>;
  constexpr static ElemT hx = GpuGridType::hx;
  constexpr static ElemT hy = GpuGridType::hy;
  constexpr static bool stepsAreSame = ((hx > hy) ? (hx - hy < 1e-8f) : (hy - hx < 1e-8f));
  static_assert(stepsAreSame, "Grid steps are different!");

  using GasStateType = GasState<ElemT>;

  using LevelSetSolverType = GpuLevelSetSolver<ElemT, ShapeType>;
  using SrmSolverType = GpuSrmSolver<ShapeType, GasStateType>;

  GpuGridT<ElemT> grid = GpuGridT<ElemT>(nx, ny, detail::ToFloatV<LxToType, ElemT>, detail::ToFloatV<LyToType, ElemT>, 3U);
  PhysicalPropertiesData<ElemT> physicalProperties{ static_cast<ElemT>(0.5),
                                                  static_cast<ElemT>(-0.00534),
                                                  static_cast<ElemT>(3900.0),
                                                  static_cast<ElemT>(1700),
                                                  static_cast<ElemT>(101325),
                                                  static_cast<ElemT>(1.18),
                                                  static_cast<ElemT>(2628),
                                                  ShapeType::getFCritical(),
                                                  ShapeType::getInitialSBurn() };

  GasStateType initialGasState{ static_cast<ElemT>(0.5),
                                static_cast<ElemT>(0.0),
                                static_cast<ElemT>(0.0),
                                physicalProperties.P0 };
};

} // namespace kae
