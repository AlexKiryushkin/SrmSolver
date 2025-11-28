#pragma once

#include "std_includes.h"

#include "boundary_condition.h"
#include "gas_state.h"

#pragma warning( disable : 4068 )

namespace kae {

namespace detail {

template <class PhysicalPropertiesT, class GasStateT, class ElemType = typename GasStateT::ElemType>
HOST_DEVICE GasStateT getFirstOrderMassFlowExtrapolatedGhostValue(const GasStateT & gasState,
                                                                  const GasStateT& closestGasState,
        const GasParameters<ElemType> &gasParameters)
{
  using ElemType = typename GasStateT::ElemType;
  const auto c = SonicSpeed::get(closestGasState, gasParameters);

  const auto kappa    = gasParameters.kappa;
  constexpr auto nu       = PhysicalPropertiesT::nu;
  const auto coefficient1 = gasState.ux + 1 / closestGasState.rho / c * gasState.p;
  const auto coefficient2 = -1 / closestGasState.rho / c;
  const auto coefficient3 = kappa / (kappa - 1) / PhysicalPropertiesT::mt;

  ElemType p1 = 10 * gasState.p;

#pragma unroll
  for (unsigned i{ 0u }; i < 10u; ++i)
  {
    const auto power = std::pow(p1, -nu);
    const auto fOfP = static_cast<ElemType>(0.5) * coefficient2 * coefficient2 * p1 * p1 +
                      coefficient2 * coefficient3 * p1 * p1 * power +
                      coefficient1 * coefficient2 * p1 + coefficient1 * coefficient3 * p1 * power +
                      static_cast<ElemType>(0.5) * coefficient1 * coefficient1 - PhysicalPropertiesT::H0;
    const auto fPrimeOfP = coefficient2 * coefficient2 * p1 +
                           (static_cast<ElemType>(2.0) - nu) * coefficient2 * coefficient3 * p1 * power +
                           (static_cast<ElemType>(1.0) - nu) * coefficient1 * coefficient3 * power +
                            coefficient1 * coefficient2;

    const auto delta = fOfP / fPrimeOfP;
    p1 = p1 - delta;
    if (std::fabs(delta) <= static_cast<ElemType>(1e-6) * p1)
    {
      break;
    }
  }

  const auto un = coefficient1 + coefficient2 * p1;
  return GasStateT{ PhysicalPropertiesT::mt * std::pow(p1, nu) / un, un, static_cast<ElemType>(0.0), p1 };
}

template <class PhysicalPropertiesT, class GasStateT, class ElemType = typename GasStateT::ElemType>
HOST_DEVICE GasStateT getFirstOrderPressureOutletExtrapolatedGhostValue(const GasStateT & gasState,
    const GasStateT& closestGasState,
    const GasParameters<ElemType>& gasParameters)
{
  const auto c = SonicSpeed::get(closestGasState, gasParameters);
  if (closestGasState.ux >= c)
  {
    return gasState;
  }

  constexpr auto P0    = PhysicalPropertiesT::P0;
  return GasStateT{ gasState.rho - 1 / c / c * (gasState.p - P0),
                    gasState.ux + 1 / closestGasState.rho / c * (gasState.p - P0),
                    gasState.uy,
                    P0 };
}

template <class GasStateT, class ElemType = typename GasStateT::ElemType>
HOST_DEVICE GasStateT getFirstOrderWallExtrapolatedGhostValue(const GasStateT & gasState,
    const GasStateT& closestGasState,
    const GasParameters<ElemType>& gasParameters)
{
  const auto c = SonicSpeed::get(closestGasState, gasParameters);
  return GasStateT{ gasState.rho + closestGasState.rho / c * gasState.ux,
                    0,
                    gasState.uy,
                    gasState.p + c * closestGasState.rho * gasState.ux };
}

template <class GasStateT, class ElemType = typename GasStateT::ElemType>
HOST_DEVICE GasStateT getFirstOrderMirrorExtrapolatedGhostValue(const GasStateT & gasState,
    const GasStateT& closestGasState,
    const GasParameters<ElemType>&)
{
  return kae::MirrorState::get(gasState);
}

template <class PhysicalPropertiesT, class GasStateT, class ElemType = typename GasStateT::ElemType>
HOST_DEVICE GasStateT getFirstOrderExtrapolatedGhostValue(const GasStateT & gasState,
    const GasStateT& closestGasState,
    const GasParameters<ElemType>& gasParameters,
                                                          EBoundaryCondition boundaryCondition)
{
  switch (boundaryCondition)
  {
  case EBoundaryCondition::eMassFlowInlet:
    return getFirstOrderMassFlowExtrapolatedGhostValue<PhysicalPropertiesT>(gasState, closestGasState, gasParameters);
  case EBoundaryCondition::ePressureOutlet:
    return getFirstOrderPressureOutletExtrapolatedGhostValue<PhysicalPropertiesT>(gasState, closestGasState, gasParameters);
  case EBoundaryCondition::eWall:
    return getFirstOrderWallExtrapolatedGhostValue(gasState, closestGasState, gasParameters);
  case EBoundaryCondition::eMirror:
    return getFirstOrderMirrorExtrapolatedGhostValue(gasState, closestGasState, gasParameters);
  default:
    return GasStateT{};
  }
}

} // namespace detail

} // namespace kae
