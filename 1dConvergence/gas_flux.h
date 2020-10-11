#pragma once

#include "gas_state.h"
#include "types.h"

template <class U, class F, unsigned Step = 1U>
ElemT getFlux(const GasState* pState, std::size_t idx, ElemT lambda)
{
  constexpr ElemT epsilon{ std::is_same<ElemT, float>::value ? static_cast<ElemT>(1e-12) : static_cast<ElemT>(1e-24) };

  lambda = 0.0;
  for (std::size_t i{ idx - Step }; i < idx + 3 * Step; i += Step)
  {
    lambda = std::max(lambda, WaveSpeed::get(pState[i]));
  }

  const ElemT plusFluxes[2U] = {
    F::get(pState[idx])        - F::get(pState[idx - Step]) + lambda * (U::get(pState[idx])        - U::get(pState[idx - Step])),
    F::get(pState[idx + Step]) - F::get(pState[idx])        + lambda * (U::get(pState[idx + Step]) - U::get(pState[idx]))
  };
  const ElemT minusFluxes[2U] = {
    F::get(pState[idx + Step])     - F::get(pState[idx])        - lambda * (U::get(pState[idx + Step])     - U::get(pState[idx])),
    F::get(pState[idx + 2 * Step]) - F::get(pState[idx + Step]) - lambda * (U::get(pState[idx + 2 * Step]) - U::get(pState[idx + Step]))
  };
  const ElemT averageFlux = static_cast<ElemT>(0.5) * (F::get(pState[idx]) + F::get(pState[idx + Step]));

  ElemT s1 = sqr(sqr(plusFluxes[0U]) + epsilon);
  ElemT s2 = sqr(sqr(plusFluxes[1U]) + epsilon);

  const ElemT r1 = s2 / (4 * s2 + 8 * s1);

  s1 = sqr(sqr(minusFluxes[0U]) + epsilon);
  s2 = sqr(sqr(minusFluxes[1U]) + epsilon);

  const ElemT r2 = s1 / (4 * s1 + 8 * s2);

  return averageFlux - r1 * (plusFluxes[1U] - plusFluxes[0U]) - r2 * (minusFluxes[1U] - minusFluxes[0U]);
}
