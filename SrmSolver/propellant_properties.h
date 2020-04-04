#pragma once

#include "gas_state.h"
#include "to_float.h"

namespace kae {

template <class NuT, class MtT, class TBurnT, class RhoPT, class P0T, class ElemT>
struct PropellantProperties
{
  constexpr static ElemT nu    = detail::ToFloatV<NuT, ElemT>;
  constexpr static ElemT mt    = detail::ToFloatV<MtT, ElemT>;
  constexpr static ElemT TBurn = detail::ToFloatV<TBurnT, ElemT>;
  constexpr static ElemT rhoP  = detail::ToFloatV<RhoPT, ElemT>;
  constexpr static ElemT P0    = detail::ToFloatV<P0T, ElemT>;

  template <class GasState>
  constexpr static ElemT H0 = GasState::Cp * TBurn;
};

template <class PropellantPropertiesT>
struct BurningRate
{
  template <class GasStateT>
  __host__ __device__ auto operator()(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return -PropellantPropertiesT::mt * std::pow(kae::P::get(state), PropellantPropertiesT::nu) / PropellantPropertiesT::rhoP;
  }
};

} // namespace kae
