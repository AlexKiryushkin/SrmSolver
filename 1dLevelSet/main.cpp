
#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>

#include "get_state.h"
#include "level_set_integrate_step.h"
#include "quartic_solve.h"


template <class FloatT, std::size_t Nx, std::size_t Ny>
std::pair<FloatT, FloatT> getResiduals()
{
  constexpr auto hx = kae::h<FloatT, Nx>;
  constexpr auto hy = kae::h<FloatT, Ny>;
  constexpr auto shape = kae::EShape::eCircle;

  const std::vector<FloatT> initState = kae::initialState<FloatT, Nx, Ny>(shape);
  const std::vector<FloatT> goldState = kae::reinitializedGoldState<FloatT, Nx, Ny>(shape);

  std::vector<FloatT> prevState = initState;
  std::vector<FloatT> firstState = initState;
  std::vector<FloatT> currState = initState;
  std::vector<FloatT> xRoots(std::size(initState), std::numeric_limits<FloatT>::quiet_NaN());
  std::vector<FloatT> yRoots(std::size(initState), std::numeric_limits<FloatT>::quiet_NaN());

  for (std::size_t i{}; i < Nx - 1; ++i)
  {
    for (std::size_t j{}; j < Ny - 1; ++j)
    {
      const auto idx = j * Nx + i;
      const auto centralState = initState[idx];
      const auto rightState = initState[idx + 1];
      const auto upperState = initState[idx + Nx];
      if ((centralState > 0 && rightState < 0) || (centralState < 0 && rightState > 0))
      {
        xRoots[idx] = kae::quarticSolve(initState, idx, hx, 1U);
        const auto x = i * hx + xRoots[idx];
        const auto y = j * hy;
        const auto residual = std::fabs(kae::reinitializedGoldFunctionValue(x, y, shape));
        if (residual > hx * hx * hx)
        {
          const auto root = kae::quarticSolve(initState, idx, hx, 1U);
          std::cout << "x: " << hx * hx * hx << "  "
                    << i               << "  "
                    << j               << "  "
                    << x + xRoots[idx] << "  "
                    << y               << "  "
                    << kae::reinitializedGoldFunctionValue(x, y, shape) << "\n";
        }
      }
      if ((centralState > 0 && upperState < 0) || (centralState < 0 && upperState > 0))
      {
        yRoots[idx] = kae::quarticSolve(initState, idx, hy, Nx);
        const auto x = i * hx;
        const auto y = j * hy + yRoots[idx];
        const auto residual = std::fabs(kae::reinitializedGoldFunctionValue(x, y, shape));
        if (residual > hy * hy * hy)
        {
          const auto root = kae::quarticSolve(initState, idx, hy, Nx);
          std::cout << "y: " << hy * hy * hy     << "  "
                    << i               << "  "
                    << j               << "  "
                    << x               << "  "
                    << y + yRoots[idx] << "  "
                    << kae::reinitializedGoldFunctionValue(x, y, shape) << "\n";
        }
      }
    }
  }

  constexpr auto iterationCount = 80U;
  for (std::size_t iteration{}; iteration < iterationCount; ++iteration)
  {
    std::swap(prevState, currState);
    kae::reinitializeStep2d(initState, prevState, firstState, currState, xRoots, yRoots, Nx, Ny,
      hx, hy, kae::ETimeDiscretizationOrder::eThree);
  }

  std::ofstream outputFile("sgd_" + std::to_string(Nx) + ".dat");
  for (std::size_t i{}; i < Nx; ++i)
  {
    for (std::size_t j{}; j < Ny; ++j)
    {
      const auto idx = j * Nx + i;
      outputFile << i * hx << ";" << j * hy << ";" << currState[idx] << ";" << goldState[idx] << "\n";
    }
  }

  auto l1Error = static_cast<FloatT>(0.0);
  auto linfError = static_cast<FloatT>(0.0);
  std::size_t nPoints{};
  for (std::size_t i{ 10 }; i < Nx - 10; ++i)
  {
    for (std::size_t j{ 10 }; j < Ny - 10; ++j)
    {
      const auto idx = j * Nx + i;
      const auto isGhost = std::fabs(goldState[idx]) < 4 * hx; /*(goldState[idx] >= 0) &&
        (goldState[idx - 1] < 0 || goldState[idx - 2] < 0 || goldState[idx - 3] < 0 || goldState[idx - 4] < 0 ||
         goldState[idx + 1] < 0 || goldState[idx + 2] < 0 || goldState[idx + 3] < 0 || goldState[idx + 4] < 0 || 
         goldState[idx - Nx] < 0 || goldState[idx - 2 * Nx] < 0 || goldState[idx - 3 * Nx] < 0 || goldState[idx - 4 * Nx] < 0 ||
         goldState[idx + Nx] < 0 || goldState[idx + 2 * Nx] < 0 || goldState[idx + 3 * Nx] < 0 || goldState[idx + 4 * Nx] < 0 );*/
      if (isGhost)
      {
        const auto error = std::fabs(currState[idx] - goldState[idx]);
        l1Error = (l1Error * nPoints + error) / ++nPoints;
        linfError = std::max(error, linfError);
      }
    }
  }

  return { l1Error, linfError };
}

int main()
{
  using FloatT = double;
  const auto [l1Err100, linfErr100] = getResiduals<FloatT, 100, 100>();

  std::cout << "N = 100; "
            << "l1Err = "
            << l1Err100
            << "; order = "
            << "; linfErr = "
            << linfErr100
            << "; order = "
            << "\n";

  const auto [l1Err200, linfErr200] = getResiduals<FloatT, 200, 200>();
  std::cout << "N = 200; "
            << "l1Err = "
            << l1Err200
            << "; order = "
            << std::log2(l1Err100 / l1Err200)
            << "; linfErr = "
            << linfErr200
            << "; order = "
            << std::log2(linfErr100 / linfErr200)
            << "\n";

  const auto [l1Err400, linfErr400] = getResiduals<FloatT, 400, 400>();
  std::cout << "N = 400; "
            << "l1Err = "
            << l1Err400
            << "; order = "
            << std::log2(l1Err200 / l1Err400)
            << "; linfErr = "
            << linfErr400
            << "; order = "
            << std::log2(linfErr200 / linfErr400)
            << "\n";

  const auto [l1Err800, linfErr800] = getResiduals<FloatT, 800, 800>();
  std::cout << "N = 800; "
            << "l1Err = "
            << l1Err800
            << "; order = "
            << std::log2(l1Err400 / l1Err800)
            << "; linfErr = "
            << linfErr800
            << "; order = "
            << std::log2(linfErr400 / linfErr800)
            << "\n";
}