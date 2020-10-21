
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
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
  ElemT prevL1Error{};
  ElemT prevLInfError{};

  for (auto nPoints : nPointsArray)
  {
    constexpr unsigned order{ 3U };
    kae::GasDynamicsSolver solver{ makeProblem(nPoints), order };

    const auto numericalGasStates = solver.solve();
    const auto goldGasStates = solver.goldSolution();
    const auto h{ solver.getH() };
    const auto solutionSize{ numericalGasStates.size() };

    ElemT l1Error{};
    ElemT lInfError{};

    const std::string fileName = filePrefix + "_" + std::to_string(nPoints) + ".txt";
    std::ofstream solutionFile{ fileName };
    for (std::size_t i{}; i < solutionSize; ++i)
    {
      const auto& numericalGasState = numericalGasStates.at(i);
      const auto& goldGasState = goldGasStates.at(i);
      solutionFile << i * h << ";" << numericalGasState.rho << ";" << numericalGasState.u << ";" << numericalGasState.p << ";"
        << goldGasState.rho << ";" << goldGasState.u << ";" << goldGasState.p << "\n";

      l1Error += std::fabs(numericalGasState.rho - goldGasState.rho) * h;
      lInfError = std::max(lInfError, std::fabs(numericalGasState.rho - goldGasState.rho));
    }

    std::cout << "1 / N = 1 / "   << (nPoints - 1U) << ". "
              << "l1Error = "     << l1Error        << ". order: " << std::log2(prevL1Error / l1Error)
              << ". lInfError = " << lInfError      << ". order: " << std::log2(prevLInfError / lInfError) << ".\n";

    std::swap(l1Error, prevL1Error);
    std::swap(lInfError, prevLInfError);
  }
}

int main()
{
  const std::vector<unsigned> nPointsArray{ 41U, 81U, 161U, 321U, 641U, 1281U, 2561U };
  checkApproximationConvergence(nPointsArray, kae::ProblemFactory::makeEulerSmoothProblem, "mass_flow");
}