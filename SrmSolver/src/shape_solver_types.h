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

  using NuType = std::ratio<1052, 10000>;
  using MtType = std::ratio<-31355, 10000>;
  using TBurnType = std::ratio<2950, 1>;
  using RhoPType = std::ratio<1580, 1>;
  using P0Type = std::ratio<101325, 1>;
  using KappaType = std::ratio<12, 10>;
  using CpType = std::ratio<162913, 100>;
  using PhysicalPropertiesType = PhysicalProperties<NuType, MtType, TBurnType, RhoPType, P0Type, KappaType, CpType, ShapeType>;

  using GasStateType = GasState<ElemT>;

  using LevelSetSolverType = GpuLevelSetSolver<ElemT, ShapeType>;
  using SrmSolverType = GpuSrmSolver<ShapeType, GasStateType, PhysicalPropertiesType>;

  GpuGridT<ElemT> grid = GpuGridT<ElemT>(nx, ny, detail::ToFloatV<LxToType, ElemT>, detail::ToFloatV<LyToType, ElemT>, 3U);
  constexpr static GasStateType initialGasState{ static_cast<ElemT>(1.0),
                                                 static_cast<ElemT>(0.0),
                                                 static_cast<ElemT>(0.0),
                                                 PhysicalPropertiesType::P0 };
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

  using NuType                   = std::ratio<41, 100>;
  using MtType                   = std::ratio<-96446, 1000000>;
  using TBurnType                = std::ratio<2950, 1>;
  using RhoPType                 = std::ratio<1700, 1>;
  using P0Type                   = std::ratio<101325, 1>;
  using KappaType                = std::ratio<123, 100>;
  using CpType                   = std::ratio<1800, 1>;
  using PhysicalPropertiesType = PhysicalProperties<NuType, MtType, TBurnType, RhoPType, P0Type, KappaType, CpType, ShapeType>;

  using GasStateType = GasState<ElemT>;

  using LevelSetSolverType = GpuLevelSetSolver<ElemT, ShapeType>;
  using SrmSolverType      = GpuSrmSolver<ShapeType, GasStateType, PhysicalPropertiesType>;

  GpuGridT<ElemT> grid = GpuGridT<ElemT>(nx, ny, detail::ToFloatV<LxToType, ElemT>, detail::ToFloatV<LyToType, ElemT>, 3U);

  constexpr static GasStateType initialGasState{ static_cast<ElemT>(1.0),
                                                 static_cast<ElemT>(0.0),
                                                 static_cast<ElemT>(0.0),
                                                 PhysicalPropertiesType::P0 };
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

  using NuType    = std::ratio<5, 10>;
  using MtType    = std::ratio<-534, 100000>;
  using TBurnType = std::ratio<3900, 1>;
  using RhoPType  = std::ratio<1700, 1>;
  using P0Type    = std::ratio<101325, 1>;
  using KappaType = std::ratio<118, 100>;
  using CpType    = std::ratio<2628, 1>;
  using PhysicalPropertiesType = PhysicalProperties<NuType, MtType, TBurnType, RhoPType, P0Type, KappaType, CpType, ShapeType>;

  using GasStateType = GasState<ElemT>;

  using LevelSetSolverType = GpuLevelSetSolver<ElemT, ShapeType>;
  using SrmSolverType = GpuSrmSolver<ShapeType, GasStateType, PhysicalPropertiesType>;

  GpuGridT<ElemT> grid = GpuGridT<ElemT>(nx, ny, detail::ToFloatV<LxToType, ElemT>, detail::ToFloatV<LyToType, ElemT>, 3U);

  constexpr static GasStateType initialGasState{ static_cast<ElemT>(0.5),
                                                 static_cast<ElemT>(0.0),
                                                 static_cast<ElemT>(0.0),
                                                 PhysicalPropertiesType::P0 };
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

  using NuType = std::ratio<5, 10>;
  using MtType = std::ratio<-534, 100000>;
  using TBurnType = std::ratio<3900, 1>;
  using RhoPType = std::ratio<1700, 1>;
  using P0Type = std::ratio<101325, 1>;
  using KappaType = std::ratio<118, 100>;
  using CpType = std::ratio<2628, 1>;
  using PhysicalPropertiesType = PhysicalProperties<NuType, MtType, TBurnType, RhoPType, P0Type, KappaType, CpType, ShapeType>;

  using GasStateType = GasState<ElemT>;

  using LevelSetSolverType = GpuLevelSetSolver<ElemT, ShapeType>;
  using SrmSolverType = GpuSrmSolver<ShapeType, GasStateType, PhysicalPropertiesType>;

  GpuGridT<ElemT> grid = GpuGridT<ElemT>(nx, ny, detail::ToFloatV<LxToType, ElemT>, detail::ToFloatV<LyToType, ElemT>, 3U);

  constexpr static GasStateType initialGasState{ static_cast<ElemT>(0.5),
                                                 static_cast<ElemT>(0.0),
                                                 static_cast<ElemT>(0.0),
                                                 PhysicalPropertiesType::P0 };
};

} // namespace kae
