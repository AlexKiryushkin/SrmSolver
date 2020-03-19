#pragma once

#include "gas_state.h"
#include "to_float.h"

namespace kae {

template <class NuT, class MtT, class TBurnT, class RhoPT, class P0T>
struct PropellantProperties
{
  constexpr static float nu = detail::ToFloatV<NuT>;
  constexpr static float mt = detail::ToFloatV<MtT>;
  constexpr static float TBurn = detail::ToFloatV<TBurnT>;
  constexpr static float rhoP = detail::ToFloatV<RhoPT>;
  constexpr static float P0 = detail::ToFloatV<P0T>;

  template <class GasState>
  constexpr static float H0 = GasState::Cp * TBurn;
};

template <class PropellantPropertiesT>
struct BurningRate
{
  template <class GasStateT>
  __host__ __device__ float operator()(const GasStateT & state)
  {
    return -PropellantPropertiesT::mt * std::pow(kae::P::get(state), PropellantPropertiesT::nu) / PropellantPropertiesT::rhoP;
  }
};

} // namespace kae
