#pragma once

#include "gas_state.h"
#include "math_utilities.h"

namespace kae {

template <class U, class F, unsigned Step, class KappaType, class CpType>
__host__ __device__ float getFlux(const GasState<KappaType, CpType> * pState, unsigned index, float lambda)
{
  constexpr float epsilon{ 1e-6f };
  const float variables[4U] = { U::get(pState[index - Step]),
                                U::get(pState[index]),
                                U::get(pState[index + Step]),
                                U::get(pState[index + 2 * Step]) };
  const float fluxes[4U] = { F::get(pState[index - Step]),
                             F::get(pState[index]),
                             F::get(pState[index + Step]),
                             F::get(pState[index + 2 * Step]) };

  float s1 = sqr(sqr(fluxes[1U] - fluxes[0U] + lambda * (variables[1U] - variables[0U])) + epsilon);
  float s2 = sqr(sqr(fluxes[2U] - fluxes[1U] + lambda * (variables[2U] - variables[1U])) + epsilon);

  const float r1 = s2 / (4.0f * s2 + 8.0f * s1);

  s1 = sqr(sqr(fluxes[2U] - fluxes[1U] - lambda * (variables[2U] - variables[1U])) + epsilon);
  s2 = sqr(sqr(fluxes[3U] - fluxes[2U] - lambda * (variables[3U] - variables[2U])) + epsilon);

  const float r2 = s1 / (4.0f * s1 + 8.0f * s2);

  return
    0.5f * (fluxes[1U] + fluxes[2U]) -
    r1 * (fluxes[2U] - 2.0f * fluxes[1U] + fluxes[0U] + lambda * (variables[2U] - 2.0f * variables[1U] + variables[0U])) -
    r2 * (fluxes[3U] - 2.0f * fluxes[2U] + fluxes[1U] - lambda * (variables[3U] - 2.0f * variables[2U] + variables[1U]));
}

template <unsigned Step, class KappaType, class CpType>
__host__ __device__ float4 getXFluxes(const GasState<KappaType, CpType> * pState, unsigned index, float lambda)
{
  return float4{ getFlux<Rho,       MassFluxX,      Step>(pState, index, lambda),
                 getFlux<MassFluxX, MomentumFluxXx, Step>(pState, index, lambda),
                 getFlux<MassFluxY, MomentumFluxXy, Step>(pState, index, lambda),
                 getFlux<RhoEnergy, EnthalpyFluxX,  Step>(pState, index, lambda) };
}

template <unsigned Step, class KappaType, class CpType>
__host__ __device__ float4 getYFluxes(const GasState<KappaType, CpType> * pState, unsigned index, float lambda)
{
  return float4{ getFlux<Rho,       MassFluxY,      Step>(pState, index, lambda),
                 getFlux<MassFluxX, MomentumFluxXy, Step>(pState, index, lambda),
                 getFlux<MassFluxY, MomentumFluxYy, Step>(pState, index, lambda),
                 getFlux<RhoEnergy, EnthalpyFluxY,  Step>(pState, index, lambda) };
}

} // namespace kae
