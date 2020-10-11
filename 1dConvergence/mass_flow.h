#pragma once

#include "gas_state.h"
#include "propellant_constants.h"

inline GasState getMassFlowGhostValue(const GasState& gasState, const GasState& closestGasState)
{
  const auto c = SonicSpeed::get(closestGasState);

  constexpr auto kappa = GasState::kappa;
  const auto coefficient1 = gasState.u + 1 / closestGasState.rho / c * gasState.p;
  const auto coefficient2 = -1 / closestGasState.rho / c;
  const auto coefficient3 = kappa / (kappa - 1) / mt;

  ElemT p1 = 10 * gasState.p;

  for (unsigned i{ 0u }; i < 100u; ++i)
  {
    const auto power = std::pow(p1, -nu);
    const auto fOfP = static_cast<ElemT>(0.5) * coefficient2 * coefficient2 * p1 * p1 +
      coefficient2 * coefficient3 * p1 * p1 * power +
      coefficient1 * coefficient2 * p1 + coefficient1 * coefficient3 * p1 * power +
      static_cast<ElemT>(0.5) * coefficient1 * coefficient1 - H0;
    const auto fPrimeOfP = coefficient2 * coefficient2 * p1 +
      (static_cast<ElemT>(2.0) - nu) * coefficient2 * coefficient3 * p1 * power +
      (static_cast<ElemT>(1.0) - nu) * coefficient1 * coefficient3 * power +
      coefficient1 * coefficient2;

    const auto delta = fOfP / fPrimeOfP;
    p1 = p1 - delta;
    if (std::fabs(delta) <= static_cast<ElemT>(1e-10) * p1)
    {
      break;
    }
  }

  const auto un = coefficient1 + coefficient2 * p1;
  return GasState{ mt * std::pow(p1, nu) / un, un, p1 };
}
