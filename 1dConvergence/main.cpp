
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>


#include "types.h"
#include "gas_state.h"
#include "runge_kutta.h"
#include "sod_problem.h"

constexpr GasState initialGasState{ 1.0, 0.0, 1.0 };
constexpr GasState shockWaveGasState{ 1.22634, 0.247494, 1.33189 };
constexpr GasState contactDiscontinuityGasState{ 0.0466304, 0.247494, 1.33189 };

constexpr ElemT contactDiscontinuitySpeed = 0.247494;
constexpr ElemT shockWaveSpeed = 1.34099515;

template <unsigned order>
void solve(unsigned gridSize, double tMax, std::string fileName)
{
  std::vector<GasState> prevGasValues(gridSize, initialGasState);
  std::vector<GasState> firstGasValues(gridSize, initialGasState);
  std::vector<GasState> currGasValues(gridSize, initialGasState);

  ElemT t{};
  while (t < tMax)
  {
    std::swap(prevGasValues, currGasValues);
    setGhostValues<order>(prevGasValues);
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

  const auto discontinuityPosition = xBoundary + contactDiscontinuitySpeed * t;
  const auto shockWavePosition = xBoundary + shockWaveSpeed * t;

  ElemT rhoError{};
  ElemT uError{};
  ElemT pError{};
  for (std::size_t idx{ 0U }; idx < gridSize; ++idx)
  {
    const auto x = idx * h;
    const auto& gasState = currGasValues[idx];
    if (x < discontinuityPosition)
    {
      rhoError += std::fabs(contactDiscontinuityGasState.rho - gasState.rho);
      uError += std::fabs(contactDiscontinuityGasState.u - gasState.u);
      pError += std::fabs(contactDiscontinuityGasState.p - gasState.p);
    }
    else if (x < shockWavePosition)
    {
      rhoError += std::fabs(shockWaveGasState.rho - gasState.rho);
      uError += std::fabs(shockWaveGasState.u - gasState.u);
      pError += std::fabs(shockWaveGasState.p - gasState.p);
    }
    else
    {
      rhoError += std::fabs(initialGasState.rho - gasState.rho);
      uError += std::fabs(initialGasState.u - gasState.u);
      pError += std::fabs(initialGasState.p - gasState.p);
    }
  }

  std::cout << "order = " << order << ". h = " << h << ". t = " << t << "\n";
  std::cout << "rhoError = " << rhoError * h << "\n";
  std::cout << "uError = " << uError * h << "\n";
  std::cout << "pError = " << pError * h << "\n";
  std::ofstream file{ fileName };
  for (std::size_t idx{ 0U }; idx < gridSize; ++idx)
  {
    const auto x = idx * h;
    GasState goldValue{};
    if (x < discontinuityPosition)
    {
      goldValue = contactDiscontinuityGasState;
    }
    else if (x < shockWavePosition)
    {
      goldValue = shockWaveGasState;
    }
    else
    {
      goldValue = initialGasState;
    }
    file << x << ";" << currGasValues[idx].rho << ";" << currGasValues[idx].u << ";" << currGasValues[idx].p << ";"
                     << goldValue.rho << ";" << goldValue.u << ";" << goldValue.p << "\n";
  }
}

int main()
{

  solveSodProblem<3U>(arraySize, 0.15, std::string("output_order_3_") + std::to_string(h) + ".txt");
  solveSodProblem<2U>(arraySize, 0.15, std::string("output_order_2_") + std::to_string(h) + ".txt");
  solveSodProblem<1U>(arraySize, 0.15, std::string("output_order_1_") + std::to_string(h) + ".txt");
}