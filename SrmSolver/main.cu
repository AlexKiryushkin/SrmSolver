
#include <iostream>

#include "filesystem.h"
#include "shape_solver_types.h"
#include "solver_callbacks.h"

int main()
{
  try
  {
    using ElemType = float;
    using ShapeSolverType = kae::ShapeSolverTypes<kae::EShapeType::eWithUmbrellaShape, ElemType>;
    using SrmSolverType   = ShapeSolverType::SrmSolverType;
    using PropellantPropertiesType = ShapeSolverType::PropellantPropertiesType;

    const std::wstring writeFolder{ L"data" };
    const std::wstring currentPath = kae::append(kae::current_path(), writeFolder);
    kae::WriteToFolderCallback callback{ currentPath };

    constexpr ElemType maximumMeanPressure{ static_cast<ElemType>(2.0) };
    SrmSolverType srmSolver{ {}, ShapeSolverType::initialGasState, 100U, static_cast<ElemType>(0.9) };
    srmSolver.quasiStationaryDynamicIntegrate(1000U, maximumMeanPressure, kae::ETimeDiscretizationOrder::eTwo, callback);
  }
  catch (const std::exception & e)
  {
    std::cout << e.what() << '\n';
  }
}
