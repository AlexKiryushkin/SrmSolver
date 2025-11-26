#pragma once

#include "cuda_float_types.h"
#include "gas_state.h"
#include "math_utilities.h"

namespace kae {

template <class U, class F, class GasStateT, class ElemT = typename GasStateT::ElemType>
__forceinline__ HOST_DEVICE ElemT getFlux(const GasStateT * pState, unsigned idx, ElemT lambda, ElemT h, unsigned step)
{
  const ElemT epsilon{ sqr(h) };
  const ElemT plusFluxes[2U] = {
    F::get(pState[idx]) - F::get(pState[idx - step]) + lambda * (U::get(pState[idx]) - U::get(pState[idx - step])),
    F::get(pState[idx + step]) - F::get(pState[idx]) + lambda * (U::get(pState[idx + step]) - U::get(pState[idx])) };
  const ElemT minusFluxes[2U] = {
    F::get(pState[idx + step]) - F::get(pState[idx]) - lambda * (U::get(pState[idx + step]) - U::get(pState[idx])),
    F::get(pState[idx + 2 * step]) - F::get(pState[idx + step]) - lambda * (U::get(pState[idx + 2 * step]) - U::get(pState[idx + step])) };
  const ElemT averageFlux = static_cast<ElemT>(0.5) * (F::get(pState[idx]) + F::get(pState[idx + step]));

  ElemT s1 = sqr(sqr(plusFluxes[0U]) + epsilon);
  ElemT s2 = sqr(sqr(plusFluxes[1U]) + epsilon);

  const ElemT r1 = s2 / (4 * s2 + 8 * s1);

  s1 = sqr(sqr(minusFluxes[0U]) + epsilon);
  s2 = sqr(sqr(minusFluxes[1U]) + epsilon);

  const ElemT r2 = s1 / (4 * s1 + 8 * s2);

  return averageFlux - r1 * (plusFluxes[1U] - plusFluxes[0U]) - r2 * (minusFluxes[1U] - minusFluxes[0U]);
}

template <class GasStateT, class ElemT = typename GasStateT::ElemType>
__forceinline__ HOST_DEVICE CudaFloat4T<ElemT> getXFluxes(const GasStateT * pState, unsigned index, ElemT lambda, ElemT h, unsigned step)
{
  return CudaFloat4T<ElemT>{ getFlux<Rho,       MassFluxX>(pState, index, lambda, h, step),
                             getFlux<MassFluxX, MomentumFluxXx>(pState, index, lambda, h, step),
                             getFlux<MassFluxY, MomentumFluxXy>(pState, index, lambda, h, step),
                             getFlux<RhoEnergy, EnthalpyFluxX>(pState, index, lambda, h, step) };
}

template <class GasStateT, class ElemT = typename GasStateT::ElemType>
__forceinline__ HOST_DEVICE CudaFloat4T<ElemT> getYFluxes(const GasStateT * pState, unsigned index, ElemT lambda, ElemT h, unsigned step)
{
  return CudaFloat4T<ElemT>{ getFlux<Rho,       MassFluxY>(pState, index, lambda, h, step),
                             getFlux<MassFluxX, MomentumFluxXy>(pState, index, lambda, h, step),
                             getFlux<MassFluxY, MomentumFluxYy>(pState, index, lambda, h, step),
                             getFlux<RhoEnergy, EnthalpyFluxY>(pState, index, lambda, h, step) };
}

} // namespace kae
