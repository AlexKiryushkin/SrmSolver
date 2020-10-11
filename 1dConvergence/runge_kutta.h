#pragma once

#include <vector>

#include "gas_flux.h"
#include "gas_state.h"
#include "set_ghost_values.h"

template <unsigned order>
void rungeKuttaSubStep(std::vector<GasState>& prevGasValues,
  const std::vector<GasState>& firstGasValues,
  std::vector<GasState>& currGasValues,
  ElemT lambda,
  ElemT dt,
  ElemT prevWeight)
{
  //setGhostValues<order>(prevGasValues);
  for (std::size_t idx{ startIdx }; idx < endIdx; ++idx)
  {
    const auto massFluxLeft = getFlux<Rho, MassFlux>(prevGasValues.data(), idx - size_t(1U), lambda);
    const auto massFluxRight = getFlux<Rho, MassFlux>(prevGasValues.data(), idx, lambda);

    const auto momentumFluxLeft = getFlux<MassFlux, MomentumFlux>(prevGasValues.data(), idx - size_t(1U), lambda);
    const auto momentumFluxRight = getFlux<MassFlux, MomentumFlux>(prevGasValues.data(), idx, lambda);

    const auto enthalpyFluxLeft = getFlux<RhoEnergyFlux, RhoEnthalpyFlux>(prevGasValues.data(), idx - size_t(1U), lambda);
    const auto enthalpyFluxRight = getFlux<RhoEnergyFlux, RhoEnthalpyFlux>(prevGasValues.data(), idx, lambda);

    const auto& prevGasValue = prevGasValues[idx];

    const auto firstWeight = 1 - prevWeight;
    const auto newRho = prevWeight * (Rho::get(prevGasValue) - dt / h * (massFluxRight - massFluxLeft)) +
      firstWeight * firstGasValues[idx].rho;
    const auto newRhoU = prevWeight * (MassFlux::get(prevGasValue) - dt / h * (momentumFluxRight - momentumFluxLeft)) +
      firstWeight * MassFlux::get(firstGasValues[idx]);
    const auto newRhoE = prevWeight * (RhoEnergyFlux::get(prevGasValue) - dt / h * (enthalpyFluxRight - enthalpyFluxLeft)) +
      firstWeight * RhoEnergyFlux::get(firstGasValues[idx]);
    const auto newU = newRhoU / newRho;
    const auto newP = (GasState::kappa - 1) * (newRhoE - newRho * newU * newU / 2);

    currGasValues[idx] = GasState{ newRho, newU, newP };
  }
}

template <unsigned order>
void rungeKuttaStep(std::vector<GasState>& prevGasValues,
  std::vector<GasState>& firstGasValues,
  std::vector<GasState>& currGasValues,
  ElemT lambda,
  ElemT dt)
{
  switch (order)
  {
  case 1U:
  {
    rungeKuttaSubStep<order>(prevGasValues, firstGasValues, currGasValues, lambda, dt, 1.0);
  }
  case 2U:
  {
    rungeKuttaSubStep<order>(prevGasValues, currGasValues, firstGasValues, lambda, dt, 1.0);
    rungeKuttaSubStep<order>(firstGasValues, prevGasValues, currGasValues, lambda, dt, 0.5);
  }
  case 3U:
  {
    rungeKuttaSubStep<order>(prevGasValues, firstGasValues, currGasValues, lambda, dt, 1.0);
    rungeKuttaSubStep<order>(currGasValues, prevGasValues, firstGasValues, lambda, dt, 0.25);
    rungeKuttaSubStep<order>(firstGasValues, prevGasValues, currGasValues, lambda, dt, 2.0 / 3.0);
  }
  default:
  {}
  }
}
