#pragma once

#include <Eigen/Eigen>

#include "gas_state.h"
#include "grid_properties.h"
#include "mass_flow.h"

template <class U, unsigned order>
Eigen::Matrix<ElemT, order, 1> getPolynomial(const GasState* pState)
{
  Eigen::Matrix<ElemT, order, order> lhs;
  Eigen::Matrix<ElemT, order, 1> rhs;
  for (std::size_t rowIdx{}; static_cast<int>(rowIdx) < lhs.cols(); ++rowIdx)
  {
    const auto i = startIdx + rowIdx;
    const auto dx = xBoundary - i * h;

    ElemT value = 1;
    for (std::size_t colIdx{}; static_cast<int>(colIdx) < lhs.cols(); ++colIdx)
    {
      lhs(rowIdx, colIdx) = value;
      value *= dx;
    }
    rhs(rowIdx) = U::get(pState[i]);
  }
  return lhs.ldlt().solve(rhs);
}

template <unsigned order>
void setGhostValues(std::vector<GasState>& gasValues)
{
  const auto rhoPolynomial = getPolynomial<Rho, order>(gasValues.data());
  const auto uPolynomial   = getPolynomial<MinusU, order>(gasValues.data());
  const auto pPolynomial   = getPolynomial<P, order>(gasValues.data());
  for (std::size_t idx{}; idx < startIdx; ++idx)
  {
    const ElemT dx = xBoundary - idx * h;
    const GasState massFlowGasState = getMassFlowGhostValue(
      GasState{rhoPolynomial(0), uPolynomial(0), pPolynomial(0)}, gasValues[startIdx]);
    if (order == 1U)
    {
      gasValues[idx] = GasState{ massFlowGasState.rho, -massFlowGasState.u, massFlowGasState.p };
      continue;
    }

    constexpr auto kappa = GasState::kappa;
    Eigen::Matrix<ElemT, 3, 3> lhs;
    lhs << 
      massFlowGasState.u * massFlowGasState.u,
      (2 - kappa * nu)* MassFlux::get(massFlowGasState),
      1 - nu * MassFlux::get(massFlowGasState) * massFlowGasState.u / massFlowGasState.p,

      -kappa / (kappa - 1) * massFlowGasState.p * massFlowGasState.u / sqr(massFlowGasState.rho),
      sqr(SonicSpeed::get(massFlowGasState)) + sqr(massFlowGasState.u),
      (2 * kappa - 1) / (kappa - 1) * massFlowGasState.u / massFlowGasState.rho,

      0.0, 0.5 / SonicSpeed::get(gasValues[startIdx]), 0.5 / kappa / gasValues[startIdx].p;

    Eigen::Matrix<ElemT, 3, 1> rhs{ 0.0, 0.0,
      0.5 / SonicSpeed::get(gasValues[startIdx]) * uPolynomial(1U) + 0.5 / kappa / gasValues[startIdx].p * pPolynomial(1U) };

    const auto sol = lhs.llt().solve(rhs);

    if (order == 2)
    {
      gasValues[idx] = GasState{
          massFlowGasState.rho + sol(0) * dx,
        -(massFlowGasState.u + sol(1) * dx),
          massFlowGasState.p + sol(2) * dx
      };
    }
    else
    {
      gasValues[idx] = GasState{
          massFlowGasState.rho + sol(0) * dx + rhoPolynomial(2) * dx * dx,
        -(massFlowGasState.u + sol(1) * dx + uPolynomial(2) * dx * dx),
          massFlowGasState.p + sol(2) * dx + pPolynomial(2) * dx * dx
      };
    }
  }
}
