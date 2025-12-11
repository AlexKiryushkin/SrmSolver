#pragma once

#include "cuda_float_types.h"
#include "gas_state.h"
#include "math_utilities.h"

namespace kae {

template <class U, class F, class GasStateT, class ElemT = typename GasStateT::ElemType>
__forceinline__ HOST_DEVICE ElemT getFlux(const GasStateT * pState, const GasParameters<ElemT> &gasParameters, unsigned idx, ElemT lambda, ElemT h, unsigned step)
{
  const ElemT epsilon{ sqr(h) };
  const ElemT plusFluxes[2U] = {
    F::get(pState[idx], gasParameters) - F::get(pState[idx - step], gasParameters) + lambda * (U::get(pState[idx], gasParameters) - U::get(pState[idx - step], gasParameters)),
    F::get(pState[idx + step], gasParameters) - F::get(pState[idx], gasParameters) + lambda * (U::get(pState[idx + step], gasParameters) - U::get(pState[idx], gasParameters)) };
  const ElemT minusFluxes[2U] = {
    F::get(pState[idx + step], gasParameters) - F::get(pState[idx], gasParameters) - lambda * (U::get(pState[idx + step], gasParameters) - U::get(pState[idx], gasParameters)),
    F::get(pState[idx + 2 * step], gasParameters) - F::get(pState[idx + step], gasParameters) - lambda * (U::get(pState[idx + 2 * step], gasParameters) - U::get(pState[idx + step], gasParameters)) };
  const ElemT averageFlux = static_cast<ElemT>(0.5) * (F::get(pState[idx], gasParameters) + F::get(pState[idx + step], gasParameters));

  ElemT s1 = sqr(sqr(plusFluxes[0U]) + epsilon);
  ElemT s2 = sqr(sqr(plusFluxes[1U]) + epsilon);

  const ElemT r1 = s2 / (4 * s2 + 8 * s1);

  s1 = sqr(sqr(minusFluxes[0U]) + epsilon);
  s2 = sqr(sqr(minusFluxes[1U]) + epsilon);

  const ElemT r2 = s1 / (4 * s1 + 8 * s2);

  return averageFlux - r1 * (plusFluxes[1U] - plusFluxes[0U]) - r2 * (minusFluxes[1U] - minusFluxes[0U]);
}

template <class GasStateT, class ElemT = typename GasStateT::ElemType>
__forceinline__ HOST_DEVICE CudaFloat4T<ElemT> getXFluxes(const GasStateT * pState, const GasParameters<ElemT>& gasParameters, unsigned index, ElemT lambda, ElemT h, unsigned step)
{
  return CudaFloat4T<ElemT>{ getFlux<Rho,       MassFluxX>(pState, gasParameters, index, lambda, h, step),
                             getFlux<MassFluxX, MomentumFluxXx>(pState, gasParameters, index, lambda, h, step),
                             getFlux<MassFluxY, MomentumFluxXy>(pState, gasParameters, index, lambda, h, step),
                             getFlux<RhoEnergy, EnthalpyFluxX>(pState, gasParameters, index, lambda, h, step) };
}

template <class GasStateT, class ElemT = typename GasStateT::ElemType>
__forceinline__ HOST_DEVICE CudaFloat4T<ElemT> getYFluxes(const GasStateT * pState, const GasParameters<ElemT>& gasParameters, unsigned index, ElemT lambda, ElemT h, unsigned step)
{
  return CudaFloat4T<ElemT>{ getFlux<Rho,       MassFluxY>(pState, gasParameters, index, lambda, h, step),
                             getFlux<MassFluxX, MomentumFluxXy>(pState, gasParameters, index, lambda, h, step),
                             getFlux<MassFluxY, MomentumFluxYy>(pState, gasParameters, index, lambda, h, step),
                             getFlux<RhoEnergy, EnthalpyFluxY>(pState, gasParameters, index, lambda, h, step) };
}

} // namespace kae
