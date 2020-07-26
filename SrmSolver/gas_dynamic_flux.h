#pragma once

#include "cuda_float_types.h"
#include "gas_state.h"
#include "math_utilities.h"

namespace kae {

template <class U, class F, unsigned Step, class GasStateT, class ElemT = typename GasStateT::ElemType>
__forceinline__ __host__ __device__ ElemT getFlux(const GasStateT * pState, unsigned idx, ElemT lambda)
{
  constexpr ElemT epsilon{ std::is_same<ElemT, float>::value ? static_cast<ElemT>(1e-12) : static_cast<ElemT>(1e-24) };
  const ElemT plusFluxes[2U] = {
    F::get(pState[idx]) - F::get(pState[idx - Step]) + lambda * (U::get(pState[idx]) - U::get(pState[idx - Step])),
    F::get(pState[idx + Step]) - F::get(pState[idx]) + lambda * (U::get(pState[idx + Step]) - U::get(pState[idx])) };
  const ElemT minusFluxes[2U] = {
    F::get(pState[idx + Step]) - F::get(pState[idx]) - lambda * (U::get(pState[idx + Step]) - U::get(pState[idx])),
    F::get(pState[idx + 2 * Step]) - F::get(pState[idx + Step]) - lambda * (U::get(pState[idx + 2 * Step]) - U::get(pState[idx + Step])) };
  const ElemT averageFlux = static_cast<ElemT>(0.5) * (F::get(pState[idx]) + F::get(pState[idx + Step]));

  ElemT s1 = sqr(sqr(plusFluxes[0U]) + epsilon);
  ElemT s2 = sqr(sqr(plusFluxes[1U]) + epsilon);

  const ElemT r1 = s2 / (4 * s2 + 8 * s1);

  s1 = sqr(sqr(minusFluxes[0U]) + epsilon);
  s2 = sqr(sqr(minusFluxes[1U]) + epsilon);

  const ElemT r2 = s1 / (4 * s1 + 8 * s2);

  return averageFlux - r1 * (plusFluxes[1U] - plusFluxes[0U]) - r2 * (minusFluxes[1U] - minusFluxes[0U]);
}

template <unsigned Step, class GasStateT, class ElemT = typename GasStateT::ElemType>
__forceinline__ __host__ __device__ CudaFloat4T<ElemT> getXFluxes(const GasStateT * pState, unsigned index, ElemT lambda)
{
  return CudaFloat4T<ElemT>{ getFlux<Rho,       MassFluxX,      Step>(pState, index, lambda),
                             getFlux<MassFluxX, MomentumFluxXx, Step>(pState, index, lambda),
                             getFlux<MassFluxY, MomentumFluxXy, Step>(pState, index, lambda),
                             getFlux<RhoEnergy, EnthalpyFluxX,  Step>(pState, index, lambda) };
}

template <unsigned Step, class GasStateT, class ElemT = typename GasStateT::ElemType>
__forceinline__ __host__ __device__ CudaFloat4T<ElemT> getYFluxes(const GasStateT * pState, unsigned index, ElemT lambda)
{
  return CudaFloat4T<ElemT>{ getFlux<Rho,       MassFluxY,      Step>(pState, index, lambda),
                             getFlux<MassFluxX, MomentumFluxXy, Step>(pState, index, lambda),
                             getFlux<MassFluxY, MomentumFluxYy, Step>(pState, index, lambda),
                             getFlux<RhoEnergy, EnthalpyFluxY,  Step>(pState, index, lambda) };
}

} // namespace kae
