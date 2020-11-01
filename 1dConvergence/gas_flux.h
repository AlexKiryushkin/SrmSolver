#pragma once

#include "gas_state.h"
#include "types.h"

namespace kae {

template <class U, class F, unsigned Step = 1U>
ElemT getFlux(const GasState* pState, std::size_t idx, ElemT lambda, ElemT h)
{
  /*const ElemT epsilon{ h * h };
  constexpr ElemT half = static_cast<ElemT>(0.5);

  const ElemT plusFluxes[3U] = {
    half * (F::get(pState[idx - Step]) + lambda * U::get(pState[idx - Step])),
    half * (F::get(pState[idx])        + lambda * U::get(pState[idx])),
    half * (F::get(pState[idx + Step]) + lambda * U::get(pState[idx + Step]))
  };
  const ElemT minusFluxes[3U] = {
    half * (F::get(pState[idx])            - lambda * U::get(pState[idx])),
    half * (F::get(pState[idx + Step])     - lambda * U::get(pState[idx + Step])),
    half * (F::get(pState[idx + 2 * Step]) - lambda * U::get(pState[idx + 2 * Step]))
  };

  const ElemT betta0Plus = sqr(plusFluxes[1U] - plusFluxes[0U]);
  const ElemT betta1Plus = sqr(plusFluxes[2U] - plusFluxes[1U]);

  constexpr ElemT d0Plus = static_cast<ElemT>(1.0 / 3.0);
  constexpr ElemT d1Plus = static_cast<ElemT>(2.0 / 3.0);

  const ElemT alpha0Plus = d0Plus / sqr(epsilon + betta0Plus);
  const ElemT alpha1Plus = d1Plus / sqr(epsilon + betta1Plus);

  const ElemT fPlus = (alpha0Plus * (-half * plusFluxes[0] + 3 * half * plusFluxes[1]) + 
                       alpha1Plus * (half * plusFluxes[1] + half * plusFluxes[2])) / (alpha0Plus + alpha1Plus);

  const ElemT betta0Minus = sqr(minusFluxes[1U] - minusFluxes[0U]);
  const ElemT betta1Minus = sqr(minusFluxes[2U] - minusFluxes[1U]);

  constexpr ElemT d0Minus = static_cast<ElemT>(2.0 / 3.0);
  constexpr ElemT d1Minus = static_cast<ElemT>(1.0 / 3.0);

  const ElemT alpha0Minus = d0Minus / sqr(epsilon + betta0Minus);
  const ElemT alpha1Minus = d1Minus / sqr(epsilon + betta1Minus);
  
  const ElemT fMinus = (alpha0Minus * (half * minusFluxes[0] + half * minusFluxes[1]) +
                       alpha1Minus * (3 * half * minusFluxes[1] - half * minusFluxes[2])) / (alpha0Minus + alpha1Minus);

  return fPlus + fMinus;*/
  const ElemT epsilon{ h * h };
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

} // namespace kae
