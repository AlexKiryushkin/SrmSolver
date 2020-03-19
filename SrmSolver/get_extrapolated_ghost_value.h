#pragma once

#include <cmath>

#include "boundary_condition.h"
#include "gas_state.h"

namespace kae {

namespace detail {

template <class PropellantPropertiesT, class GasStateT>
__host__ __device__ GasStateT getFirstOrderMassFlowExtrapolatedGhostValue(const GasStateT & gasState)
{
  const float c = SonicSpeed::get(gasState);

  const float coefficient1 = gasState.ux + c / GasStateT::kappa;
  const float coefficient2 = -1.0f / gasState.rho / c;
  const float coefficient3 = GasStateT::kappa / (GasStateT::kappa - 1.0f) / PropellantPropertiesT::mt;

  float p1 = 10 * gasState.p;

#pragma unroll
  for (unsigned i{ 0u }; i < 10u; ++i)
  {
    const float power = std::pow(p1, -PropellantPropertiesT::nu);
    const float fOfP = 0.5f * coefficient2 * coefficient2 * p1 * p1 +
                       coefficient2 * coefficient3 * p1 * p1 * power +
                       coefficient1 * coefficient2 * p1 + coefficient1 * coefficient3 * p1 * power +
                       0.5f * coefficient1 * coefficient1 - PropellantPropertiesT::template H0<GasStateT>;
    const float fPrimeOfP = coefficient2 * coefficient2 * p1 +
                           (2.0f - PropellantPropertiesT::nu) * coefficient2 * coefficient3 * p1 * power +
                           (1.0f - PropellantPropertiesT::nu) * coefficient1 * coefficient3 * power +
                            coefficient1 * coefficient2;

    const float delta = fOfP / fPrimeOfP;
    p1 = p1 - delta;
    if (std::fabs(delta) <= 1e-6f * p1)
    {
      break;
    }
  }

  const float un = coefficient1 + coefficient2 * p1;
  return GasStateT{ PropellantPropertiesT::mt * std::pow(p1, PropellantPropertiesT::nu) / un, un, 0.0f, p1 };
}

template <class PropellantPropertiesT, class GasStateT>
__host__ __device__ GasStateT getFirstOrderPressureOutletExtrapolatedGhostValue(const GasStateT & gasState)
{
  const float c = SonicSpeed::get(gasState);
  if (gasState.ux >= c)
  {
    return gasState;
  }

  return GasStateT{ PropellantPropertiesT::P0 / c / c - (1.0f - GasStateT::kappa) / GasStateT::kappa * gasState.rho,
                    gasState.ux + c / GasStateT::kappa - PropellantPropertiesT::P0 / gasState.rho / c,
                    gasState.uy,
                    PropellantPropertiesT::P0 };
}

template <class GasStateT>
__host__ __device__ GasStateT getFirstOrderWallExtrapolatedGhostValue(const GasStateT & gasState)
{
  const float c = SonicSpeed::get(gasState);
  return GasStateT{ gasState.rho * (1 + gasState.ux / c),
                    0.0f,
                    gasState.uy,
                    gasState.p * (1 + GasStateT::kappa * gasState.ux / c) };
}

template <class GasStateT>
__host__ __device__ GasStateT getFirstOrderMirrorExtrapolatedGhostValue(const GasStateT & gasState)
{
  return kae::MirrorState::get(gasState);
}

template <class PropellantPropertiesT, class GasStateT>
__host__ __device__ GasStateT getFirstOrderExtrapolatedGhostValue(const GasStateT & gasState, EBoundaryCondition boundaryCondition)
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
