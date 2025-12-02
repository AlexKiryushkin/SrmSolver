
#include <iostream>

#include "filesystem.h"
#include "shape_solver_types.h"
#include "solver_callbacks.h"

int main()
{
  try
  {
    using ElemType               = float;
    using ShapeSolverType        = kae::ShapeSolverTypes<kae::EShapeType::eFlushMountedNozzle, ElemType>;
    using ShapeType              = ShapeSolverType::ShapeType;
    using SrmSolverType          = ShapeSolverType::SrmSolverType;

    ShapeSolverType shapeSolverType{};
    ShapeType shape{ shapeSolverType.grid.nx, shapeSolverType.grid.ny, shapeSolverType.grid.hx, shapeSolverType.grid.hy };

    const std::wstring writeFolder{ L"data" };
    const std::wstring currentPath = kae::append(kae::current_path(), writeFolder);
    kae::WriteToFolderCallback<ElemType> callback{ currentPath };

    const auto physicalProperties = shapeSolverType.physicalProperties;
    const auto burnRate = kae::BurningRate::get(static_cast<ElemType>(1), physicalProperties.nu, physicalProperties.mt, physicalProperties.rhoP);
    const auto dt = shapeSolverType.grid.hx / 2 / burnRate;
    std::cout << dt << '\n';
    std::cout << physicalProperties;

    kae::GasParameters<ElemType> gasParameters{ physicalProperties.kappa, physicalProperties.R };
    SrmSolverType srmSolver{ shapeSolverType.grid, physicalProperties, shape, shapeSolverType.initialGasState, gasParameters, 100U, static_cast<ElemType>(0.8)};
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
