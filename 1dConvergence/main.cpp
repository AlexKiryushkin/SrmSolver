
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "boundary.h"
#include "function.h"
#include "grid.h"
#include "solver.h"

template <class MakeProblemT>
void checkApproximationConvergence(const std::vector<unsigned> & nPointsArray,
                                   MakeProblemT                  makeProblem,
                                   const std::string &           filePrefix)
{
  ElemT prevL1BoundaryError{};
  ElemT prevL1GasDynamicsError{};
  ElemT prevLInfGasDynamicsError{};

  std::stringstream gasDynamicsStream{ "Gas dynamics errors:\n", std::ios_base::ate | std::ios_base::in | std::ios_base::out };
  std::stringstream boundaryStream{ "Boundary errors:\n", std::ios_base::ate | std::ios_base::in | std::ios_base::out };
  for (auto nPoints : nPointsArray)
  {
    constexpr unsigned order{ 3U };
    kae::GasDynamicsSolver solver{ makeProblem(nPoints), order };

    const auto numericalGasStates = solver.solve();
    const auto goldGasStates = solver.goldSolution();
    const auto h{ solver.getH() };
    const auto solutionSize{ numericalGasStates.size() };

    ElemT l1BoundaryError{};
    ElemT l1GasDynamicsError{};
    ElemT lInfGasDynamicsError{};

    const std::string fileName = filePrefix + "_" + std::to_string(nPoints) + ".dat";
    std::ofstream solutionFile{ fileName };
    for (std::size_t i{}; i < solutionSize; ++i)
    {
      const auto& numericalGasState = numericalGasStates.at(i);
      const auto& goldGasState = goldGasStates.at(i);
      solutionFile << i * h << ";" << numericalGasState.rho << ";" << numericalGasState.u << ";" << numericalGasState.p << ";"
        << goldGasState.rho << ";" << goldGasState.u << ";" << goldGasState.p << "\n";

      l1GasDynamicsError += std::fabs(numericalGasState.rho - goldGasState.rho) * h;
      lInfGasDynamicsError = std::max(lInfGasDynamicsError, std::fabs(numericalGasState.rho - goldGasState.rho));
    }

    gasDynamicsStream << "1 / N = 1 / "   << (nPoints - 1U)            << ". "
                      << "l1Error = "     << l1GasDynamicsError        << ". order: " << std::log2(prevL1GasDynamicsError / l1GasDynamicsError)
                      << ". lInfError = " << lInfGasDynamicsError      << ". order: " << std::log2(prevLInfGasDynamicsError / lInfGasDynamicsError) << ".\n";

    const auto t = solver.getProblem().getIntegrationTime();
    const auto xBoundary = solver.getProblem().getXBoundaryLeft(t, ElemT{}, 0U);
    const auto xBoundaryAnalytical = solver.getProblem().getXBoundaryLeftAnalytical(t);
    l1BoundaryError = std::fabs(xBoundary - xBoundaryAnalytical);
    boundaryStream << "1 / N = 1 / " << (nPoints - 1U) << ". "
                   << "l1Error = " << l1BoundaryError << ". order: " << std::log2(prevL1BoundaryError / l1BoundaryError) << ".\n";

    std::swap(l1BoundaryError,      prevL1BoundaryError);
    std::swap(l1GasDynamicsError,   prevL1GasDynamicsError);
    std::swap(lInfGasDynamicsError, prevLInfGasDynamicsError);
  }
  
  std::cout << gasDynamicsStream.str() << "\n";
  std::cout << boundaryStream.str() << "\n";
}

int main()
{
  const std::vector<unsigned> nPointsArray{ 41U, 81U, 161U, 321U, 641U, 1281U, 2561U };
  checkApproximationConvergence(nPointsArray, kae::ProblemFactory::makeStationaryMassFlowProblem, "stationary_mass_flow");
  checkApproximationConvergence(nPointsArray, kae::ProblemFactory::makeMovingMassFlowProblem, "moving_mass_flow");
}