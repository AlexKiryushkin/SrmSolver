
#include <iostream>

#include "filesystem.h"
#include "shape_solver_types.h"
#include "solver_callbacks.h"

int main()
{
  try
  {
    using ElemType        = float;
    using ShapeSolverType = kae::ShapeSolverTypes<kae::EShapeType::eWithUmbrellaShape, ElemType>;
    using SrmSolverType   = ShapeSolverType::SrmSolverType;

    const std::wstring writeFolder{ L"data" };
    const std::wstring currentPath = kae::append(kae::current_path(), writeFolder);
    kae::WriteToFolderCallback callback{ currentPath };

    constexpr ElemType deltaT{ static_cast<ElemType>(300.0) };
    SrmSolverType srmSolver{ {}, ShapeSolverType::initialGasState, 100U, 0.7f };
    srmSolver.dynamicIntegrate(800U, deltaT, kae::ETimeDiscretizationOrder::eTwo, callback);
  }
  catch (const std::exception & e)
  {
    std::cout << e.what() << '\n';
  }
}
