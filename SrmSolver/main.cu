
#include <iostream>
#include <string>

#include "filesystem.h"
#include "shape_solver_types.h"
#include "solver_callbacks.h"

#include <CL/cl.h>

int main()
{
  try
  {
    using ShapeSolverType = kae::ShapeSolverTypes<kae::EShapeType::eNozzleLessShape>;
    using SrmSolverType   = ShapeSolverType::SrmSolverType;

    const std::wstring writeFolder{ L"data" };
    const std::wstring currentPath = kae::append(kae::current_path(), writeFolder);
    kae::WriteToFolderCallback callback{ currentPath };

    SrmSolverType srmSolver{ {}, ShapeSolverType::initialGasState, 100U };
    srmSolver.dynamicIntegrate(800U, kae::ETimeDiscretizationOrder::eTwo, callback);
  }
  catch (const std::exception & e)
  {
    std::cout << e.what() << '\n';
  }
}
