#pragma once

#include "gas_state.h"

namespace kae {

struct MassFlowParams
{
  ElemT mt;
  ElemT nu;
  ElemT H0;
};

inline GasState getMassFlowGhostValue(const GasState& gasState, 
                                      const GasState& closestGasState,
                                      const MassFlowParams & massFlowParameters,
                                      ElemT rhoPReciprocal)
{
  const auto c = SonicSpeed::get(closestGasState);

  constexpr auto kappa = GasState::kappa;
  const auto coefficient1 = gasState.u + 1 / closestGasState.rho / c * gasState.p;
  const auto coefficient2 = -1 / closestGasState.rho / c;
  const auto coefficient3 = kappa / (kappa - 1) / massFlowParameters.mt * (1 + rhoPReciprocal);

  ElemT p1 = 10 * gasState.p;

  for (unsigned i{ 0u }; i < 100u; ++i)
  {
    const auto power = std::pow(p1, -massFlowParameters.nu);
    const auto fOfP = static_cast<ElemT>(0.5) * coefficient2 * coefficient2 * p1 * p1 +
      coefficient2 * coefficient3 * p1 * p1 * power +
      coefficient1 * coefficient2 * p1 + coefficient1 * coefficient3 * p1 * power +
      static_cast<ElemT>(0.5) * coefficient1 * coefficient1 - massFlowParameters.H0;
    const auto fPrimeOfP = coefficient2 * coefficient2 * p1 +
      (static_cast<ElemT>(2.0) - massFlowParameters.nu) * coefficient2 * coefficient3 * p1 * power +
      (static_cast<ElemT>(1.0) - massFlowParameters.nu) * coefficient1 * coefficient3 * power +
      coefficient1 * coefficient2;

    const auto delta = fOfP / fPrimeOfP;
    p1 = p1 - delta;
    if (std::fabs(delta) <= static_cast<ElemT>(1e-10) * p1)
    {
      break;
    }
  }

  const auto un = coefficient1 + coefficient2 * p1;
  return GasState{ massFlowParameters.mt * std::pow(p1, massFlowParameters.nu) / un, un, p1 };
}

inline Eigen::Matrix<ElemT, 3, 1> getMassFlowDerivatives(const GasState & massFlowGasState,
                                                         const GasState & closestState,
                                                         const GasState & goldState,
                                                         ElemT goldRhoDerivative,
                                                         ElemT calcUDerivative,
                                                         ElemT calcPDerivative,
                                                         const MassFlowParams & massFlowParams,
                                                         ElemT rhoPReciprocal)
{
  constexpr auto kappa = GasState::kappa;
  Eigen::Matrix<ElemT, 3, 3> lhs;
  lhs <<
    massFlowGasState.u * massFlowGasState.u,
    2 * MassFlux::get(massFlowGasState) - kappa * massFlowParams.nu * massFlowParams.mt * std::pow(massFlowGasState.p, massFlowParams.nu),
    1 - massFlowParams.nu * massFlowParams.mt * std::pow(massFlowGasState.p, massFlowParams.nu - 1) * massFlowGasState.u,

    -kappa / (kappa - 1) * massFlowGasState.p * massFlowGasState.u / sqr(massFlowGasState.rho),
    sqr(SonicSpeed::get(massFlowGasState)) + sqr(massFlowGasState.u),
    (2 * kappa - 1) / (kappa - 1) * massFlowGasState.u / massFlowGasState.rho,

    static_cast<ElemT>(0.0),
    static_cast<ElemT>(0.5) / SonicSpeed::get(closestState),
    static_cast<ElemT>(0.5) / kappa / closestState.p;

  Eigen::Matrix<ElemT, 3, 1> rhs{
    goldState.u * std::pow(massFlowGasState.p / goldState.p, massFlowParams.nu) * goldRhoDerivative,
    kappa / (kappa - 1) * goldState.p / sqr(goldState.rho) * goldRhoDerivative,
    static_cast<ElemT>(0.5) / SonicSpeed::get(closestState) * calcUDerivative + 
    static_cast<ElemT>(0.5) / kappa / closestState.p * calcPDerivative };

  return (lhs.transpose() * lhs).llt().solve(lhs.transpose() * rhs);
}

} // namespace kae
