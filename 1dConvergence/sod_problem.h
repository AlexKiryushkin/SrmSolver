#pragma once

#include <fstream>
#include <numeric>

#include "types.h"
#include "gas_state.h"
#include "grid_properties.h"
#include "runge_kutta.h"

template <unsigned order>
void solveSodProblem(unsigned gridSize, ElemT tMax, std::string fileName)
{
  constexpr GasState leftInitialGasState{ 1.0, 0.0, 1.0 };
  constexpr GasState rightInitialGasState{ 0.125, 0.0, 0.1 };

  std::vector<GasState> prevGasValues(gridSize, leftInitialGasState);
  for (std::size_t idx{ gridSize / 2 }; idx < gridSize; ++idx)
  {
    prevGasValues.at(idx) = rightInitialGasState;
  }

  std::vector<GasState> firstGasValues(prevGasValues);
  std::vector<GasState> currGasValues(prevGasValues);

  ElemT t{};
  while (t < tMax)
  {
    std::swap(prevGasValues, currGasValues);
    const auto lambda = std::accumulate(std::begin(prevGasValues), std::end(prevGasValues), 0.0,
      [](const ElemT value, const GasState& state)
    {
      return std::max(value, WaveSpeed::get(state));
    });

    constexpr auto courant = 0.9;
    const auto dt = std::min(courant * h / lambda, tMax - t);
    rungeKuttaStep<order>(prevGasValues, firstGasValues, currGasValues, lambda, dt);
    t += dt;
  }

  std::ofstream file{ fileName };
  for (std::size_t idx{ 0U }; idx < gridSize; ++idx)
  {
    const auto x = idx * h;
    file << x << ";" << currGasValues[idx].rho << ";" << currGasValues[idx].u << ";" << currGasValues[idx].p << "\n";
  }
}
