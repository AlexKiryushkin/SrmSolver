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
struct ShapeSolverTypes<EShapeType::eNozzleLessShape, ElemT>
{
  constexpr static unsigned nx{ 1344U * 2U + 1U };
  constexpr static unsigned ny{ 150U * 2U + 1U };
  using LxToType            = std::ratio<1344, 1000>;
  using LyToType            = std::ratio<150, 1000>;
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

  using GasStateType = GasState<PhysicalPropertiesType, ElemT>;

  using LevelSetSolverType = GpuLevelSetSolver<GpuGridType, ShapeType>;
  using SrmSolverType      = GpuSrmSolver<GpuGridType, ShapeType, GasStateType, PhysicalPropertiesType>;

  constexpr static GasStateType initialGasState{ static_cast<ElemT>(1.0),
                                                 static_cast<ElemT>(0.0),
                                                 static_cast<ElemT>(0.0),
                                                 PhysicalPropertiesType::P0 };
};

template<class ElemT>
struct ShapeSolverTypes<EShapeType::eWithUmbrellaShape, ElemT>
{
  constexpr static unsigned nx{ 820U * 2U + 1U };
  constexpr static unsigned ny{ 300U * 2U + 1U };
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

  using GasStateType = GasState<PhysicalPropertiesType, ElemT>;

  using LevelSetSolverType = GpuLevelSetSolver<GpuGridType, ShapeType>;
  using SrmSolverType = GpuSrmSolver<GpuGridType, ShapeType, GasStateType, PhysicalPropertiesType>;

  constexpr static GasStateType initialGasState{ static_cast<ElemT>(0.5),
                                                 static_cast<ElemT>(0.0),
                                                 static_cast<ElemT>(0.0),
                                                 PhysicalPropertiesType::P0 };
};

} // namespace kae
