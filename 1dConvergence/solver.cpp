
#include "solver.h"

#include <algorithm>
#include <numeric>

#include "gas_flux.h"

namespace kae {


std::vector<GasState> GasDynamicsSolver::solve()
{
  auto prevGasValues  = m_pProblem->getInitialState();
  auto firstGasValues = m_pProblem->getInitialState();
  auto currGasValues  = m_pProblem->getInitialState();

  const auto h = m_pProblem->getH();
  const auto tMax = m_pProblem->getIntegrationTime();

  ElemT t{};
  while (t < tMax)
  {
    std::swap(prevGasValues, currGasValues);
    const auto lambda = std::accumulate(std::begin(prevGasValues), std::end(prevGasValues), static_cast<ElemT>(0.0),
      [](const ElemT value, const GasState& state) { return std::max(value, WaveSpeed::get(state)); });
    constexpr auto courant = static_cast<ElemT>(0.8);
    const auto dt = std::min(courant * h / lambda, tMax - t);

    rungeKuttaStep(prevGasValues, firstGasValues, currGasValues, lambda, t, dt);
    t += dt;
  }

  return currGasValues;
}

void GasDynamicsSolver::rungeKuttaStep(std::vector<GasState> & prevGasValues,
                                       std::vector<GasState> & firstGasValues, 
                                       std::vector<GasState> & currGasValues, 
                                       ElemT                   lambda, 
                                       ElemT                   t, 
                                       ElemT                   dt) const
{
  switch (m_order)
  {
    case 1U:
    {
      m_pProblem->setGhostValues(prevGasValues, t, dt, 0U);
      rungeKuttaSubStep(prevGasValues, firstGasValues, currGasValues, lambda, dt, static_cast<ElemT>(1.0));
      m_pProblem->updateBoundaries(prevGasValues, t, dt, 0U);
      break;
    }
    case 2U:
    {
      m_pProblem->setGhostValues(prevGasValues, t, dt, 0U);
      rungeKuttaSubStep(prevGasValues, currGasValues, firstGasValues, lambda, dt, static_cast<ElemT>(1.0));
      m_pProblem->updateBoundaries(prevGasValues, t, dt, 0U);

      m_pProblem->setGhostValues(firstGasValues, t, dt, 1U);
      rungeKuttaSubStep(firstGasValues, prevGasValues, currGasValues, lambda, dt, static_cast<ElemT>(0.5));
      m_pProblem->updateBoundaries(firstGasValues, t, dt, 1U);
      break;
    }
    case 3U:
    {
      m_pProblem->setGhostValues(prevGasValues, t, dt, 0U);
      rungeKuttaSubStep(prevGasValues, firstGasValues, currGasValues, lambda, dt, static_cast<ElemT>(1.0));
      m_pProblem->updateBoundaries(prevGasValues, t, dt, 0U);

      m_pProblem->setGhostValues(currGasValues, t, dt, 1U);
      rungeKuttaSubStep(currGasValues, prevGasValues, firstGasValues, lambda, dt, static_cast<ElemT>(0.25));
      m_pProblem->updateBoundaries(currGasValues, t, dt, 1U);

      m_pProblem->setGhostValues(firstGasValues, t, dt, 2U);
      rungeKuttaSubStep(firstGasValues, prevGasValues, currGasValues, lambda, dt, static_cast<ElemT>(2.0 / 3.0));
      m_pProblem->updateBoundaries(firstGasValues, t, dt, 2U);
      break;
    }
    default:
    {}
  }
}

void GasDynamicsSolver::rungeKuttaSubStep(std::vector<GasState> &       prevGasValues,
                                          const std::vector<GasState> & firstGasValues, 
                                          std::vector<GasState> &       currGasValues, 
                                          ElemT                         lambda, 
                                          ElemT                         dt,
                                          ElemT                         prevWeight) const
{
  const auto startIdx = m_pProblem->getStartIdx();
  const auto endIdx = m_pProblem->getEndIdx();
  const auto h = m_pProblem->getH();

  for (std::size_t idx{ startIdx }; idx < endIdx; ++idx)
  {
    const auto massFluxLeft      = getFlux<Rho, MassFlux>                 (prevGasValues.data(), idx - size_t(1U), lambda, h);
    const auto massFluxRight     = getFlux<Rho, MassFlux>                 (prevGasValues.data(), idx,              lambda, h);

    const auto momentumFluxLeft  = getFlux<MassFlux, MomentumFlux>        (prevGasValues.data(), idx - size_t(1U), lambda, h);
    const auto momentumFluxRight = getFlux<MassFlux, MomentumFlux>        (prevGasValues.data(), idx,              lambda, h);

    const auto enthalpyFluxLeft  = getFlux<RhoEnergyFlux, RhoEnthalpyFlux>(prevGasValues.data(), idx - size_t(1U), lambda, h);
    const auto enthalpyFluxRight = getFlux<RhoEnergyFlux, RhoEnthalpyFlux>(prevGasValues.data(), idx,              lambda, h);

    const auto& prevGasValue = prevGasValues[idx];

    const auto firstWeight = 1 - prevWeight;
    const auto newRho  = prevWeight  * (Rho::get(prevGasValue) - dt / h * (massFluxRight - massFluxLeft)) +
                         firstWeight *  Rho::get(firstGasValues[idx]);
    const auto newRhoU = prevWeight  * (MassFlux::get(prevGasValue) - dt / h * (momentumFluxRight - momentumFluxLeft)) +
                         firstWeight *  MassFlux::get(firstGasValues[idx]);
    const auto newRhoE = prevWeight  * (RhoEnergyFlux::get(prevGasValue) - dt / h * (enthalpyFluxRight - enthalpyFluxLeft)) +
                         firstWeight *  RhoEnergyFlux::get(firstGasValues[idx]);
    const auto newU = newRhoU / newRho;
    const auto newP = (GasState::kappa - 1) * (newRhoE - newRho * newU * newU / 2);

    currGasValues[idx] = GasState{ newRho, newU, newP };
  }
}

} // namespace kae
