#pragma once

#include "cuda_float_types.h"
#include "gas_state.h"
#include "math_utilities.h"

namespace kae {

template <class U, class F, unsigned Step, class GasStateT, class ElemT = typename GasStateT::ElemType>
__host__ __device__ ElemT getFlux(const GasStateT * pState, unsigned index, ElemT lambda)
{
  constexpr ElemT epsilon{ std::is_same<ElemT, float>::value ? static_cast<ElemT>(1e-12) : static_cast<ElemT>(1e-24) };
  const ElemT variables[4U] = { U::get(pState[index - Step]),
                                U::get(pState[index]),
                                U::get(pState[index + Step]),
                                U::get(pState[index + 2 * Step]) };
  const ElemT fluxes[4U] = { F::get(pState[index - Step]),
                             F::get(pState[index]),
                             F::get(pState[index + Step]),
                             F::get(pState[index + 2 * Step]) };

  ElemT s1 = sqr(sqr(fluxes[1U] - fluxes[0U] + lambda * (variables[1U] - variables[0U])) + epsilon);
  ElemT s2 = sqr(sqr(fluxes[2U] - fluxes[1U] + lambda * (variables[2U] - variables[1U])) + epsilon);

  const ElemT r1 = s2 / (4 * s2 + 8 * s1);

  s1 = sqr(sqr(fluxes[2U] - fluxes[1U] - lambda * (variables[2U] - variables[1U])) + epsilon);
  s2 = sqr(sqr(fluxes[3U] - fluxes[2U] - lambda * (variables[3U] - variables[2U])) + epsilon);

  const ElemT r2 = s1 / (4 * s1 + 8 * s2);

  return
    static_cast<ElemT>(0.5) * (fluxes[1U] + fluxes[2U]) -
    r1 * (fluxes[2U] - 2 * fluxes[1U] + fluxes[0U] + lambda * (variables[2U] - 2 * variables[1U] + variables[0U])) -
    r2 * (fluxes[3U] - 2 * fluxes[2U] + fluxes[1U] - lambda * (variables[3U] - 2 * variables[2U] + variables[1U]));
}

template <unsigned Step, class GasStateT, class ElemT = typename GasStateT::ElemType>
__host__ __device__ CudaFloatT<4U, ElemT> getXFluxes(const GasStateT * pState, unsigned index, ElemT lambda)
{
  return CudaFloatT<4U, ElemT>{ getFlux<Rho,       MassFluxX,      Step>(pState, index, lambda),
                                getFlux<MassFluxX, MomentumFluxXx, Step>(pState, index, lambda),
                                getFlux<MassFluxY, MomentumFluxXy, Step>(pState, index, lambda),
                                getFlux<RhoEnergy, EnthalpyFluxX,  Step>(pState, index, lambda) };
}

template <unsigned Step, class GasStateT, class ElemT = typename GasStateT::ElemType>
__host__ __device__ CudaFloatT<4U, ElemT> getYFluxes(const GasStateT * pState, unsigned index, ElemT lambda)
{
  return CudaFloatT<4U, ElemT>{ getFlux<Rho,       MassFluxY,      Step>(pState, index, lambda),
                                getFlux<MassFluxX, MomentumFluxXy, Step>(pState, index, lambda),
                                getFlux<MassFluxY, MomentumFluxYy, Step>(pState, index, lambda),
                                getFlux<RhoEnergy, EnthalpyFluxY,  Step>(pState, index, lambda) };
}

} // namespace kae
