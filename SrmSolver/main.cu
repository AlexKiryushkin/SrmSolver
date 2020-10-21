
#include <iostream>

#include "filesystem.h"
#include "shape_solver_types.h"
#include "solver_callbacks.h"

int main()
{
  try
  {
    using ElemType               = float;
    using ShapeSolverType        = kae::ShapeSolverTypes<kae::EShapeType::eWithUmbrellaShape, ElemType>;
    using GpuGridType            = ShapeSolverType::GpuGridType;
    using ShapeType              = ShapeSolverType::ShapeType;
    using SrmSolverType          = ShapeSolverType::SrmSolverType;
    using PhysicalPropertiesType = ShapeSolverType::PhysicalPropertiesType;

    const std::wstring writeFolder{ L"data" };
    const std::wstring currentPath = kae::append(kae::current_path(), writeFolder);
    kae::WriteToFolderCallback<ElemType> callback{ currentPath };

    const auto burnRate = kae::BurningRate<PhysicalPropertiesType>::get(static_cast<ElemType>(1));
    const auto dt = GpuGridType::hx / 2 / burnRate;
    std::cout << dt << '\n';
    SrmSolverType srmSolver{ {}, ShapeSolverType::initialGasState, 100U, static_cast<ElemType>(0.8) };
    srmSolver.dynamicIntegrate(2000U, dt, kae::ETimeDiscretizationOrder::eTwo, callback);
  }
  catch (const std::exception & e)
  {
    std::cout << e.what() << '\n';
  }
  catch (...)
  {
    std::cout << "Unknown exception caught\n";
  }
}
