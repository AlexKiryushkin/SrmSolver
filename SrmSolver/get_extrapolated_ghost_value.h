#pragma once

#include "std_includes.h"

#include "boundary_condition.h"
#include "gas_state.h"

namespace kae {

namespace detail {

template <class PropellantPropertiesT, class GasStateT>
__host__ __device__ GasStateT getFirstOrderMassFlowExtrapolatedGhostValue(const GasStateT & gasState)
{
  using ElemType = typename GasStateT::ElemType;
  const auto c = SonicSpeed::get(gasState);

  constexpr auto kappa    = GasStateT::kappa;
  constexpr auto nu       = PropellantPropertiesT::nu;
  const auto coefficient1 = gasState.ux + c / kappa;
  const auto coefficient2 = -1 / gasState.rho / c;
  const auto coefficient3 = kappa / (kappa - 1) / PropellantPropertiesT::mt;

  ElemType p1 = 10 * gasState.p;

#pragma unroll
  for (unsigned i{ 0u }; i < 10u; ++i)
  {
    const auto power = std::pow(p1, -nu);
    const auto fOfP = static_cast<ElemType>(0.5) * coefficient2 * coefficient2 * p1 * p1 +
                      coefficient2 * coefficient3 * p1 * p1 * power +
                      coefficient1 * coefficient2 * p1 + coefficient1 * coefficient3 * p1 * power +
                      static_cast<ElemType>(0.5) * coefficient1 * coefficient1 - PropellantPropertiesT::H0;
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
  return GasStateT{ PropellantPropertiesT::mt * std::pow(p1, nu) / un, un, static_cast<ElemType>(0.0), p1 };
}

template <class PropellantPropertiesT, class GasStateT>
__host__ __device__ GasStateT getFirstOrderPressureOutletExtrapolatedGhostValue(const GasStateT & gasState)
{
  const auto c = SonicSpeed::get(gasState);
  if (gasState.ux >= c)
  {
    return gasState;
  }

  constexpr auto P0    = PropellantPropertiesT::P0;
  constexpr auto kappa = GasStateT::kappa;
  return GasStateT{ P0 / c / c - (1 - kappa) / kappa * gasState.rho,
                    gasState.ux + c / kappa - P0 / gasState.rho / c,
                    gasState.uy,
                    P0 };
}

template <class GasStateT>
__host__ __device__ GasStateT getFirstOrderWallExtrapolatedGhostValue(const GasStateT & gasState)
{
  const auto c = SonicSpeed::get(gasState);
  return GasStateT{ gasState.rho * (1 + gasState.ux / c),
                    0,
                    gasState.uy,
                    gasState.p * (1 + GasStateT::kappa * gasState.ux / c) };
}

template <class GasStateT>
__host__ __device__ GasStateT getFirstOrderMirrorExtrapolatedGhostValue(const GasStateT & gasState)
{
  return kae::MirrorState::get(gasState);
}

template <class PropellantPropertiesT, class GasStateT>
__host__ __device__ GasStateT getFirstOrderExtrapolatedGhostValue(const GasStateT & gasState, 
                                                                  EBoundaryCondition boundaryCondition)
{
  switch (boundaryCondition)
  {
  case EBoundaryCondition::eMassFlowInlet:
    return getFirstOrderMassFlowExtrapolatedGhostValue<PropellantPropertiesT>(gasState);
  case EBoundaryCondition::ePressureOutlet:
    return getFirstOrderPressureOutletExtrapolatedGhostValue<PropellantPropertiesT>(gasState);
  case EBoundaryCondition::eWall:
    return getFirstOrderWallExtrapolatedGhostValue(gasState);
  case EBoundaryCondition::eMirror:
    return getFirstOrderMirrorExtrapolatedGhostValue(gasState);
  default:
    return GasStateT{};
  }
}

} // namespace detail

} // namespace kae
