#pragma once

#include <SrmSolver/gas_state.h>
#include <SrmSolver/to_float.h>

namespace tests {

template <class KappaT, class RT, class ElemT>
struct GasStateProperties
{
  constexpr static ElemT kappa = kae::detail::ToFloatV<KappaT, ElemT>;
  constexpr static ElemT R     = kae::detail::ToFloatV<RT, ElemT>;
};

template <class KappaT, class RT, class ElemT>
using GasStateType = kae::GasState<GasStateProperties<KappaT, RT, ElemT>, ElemT>;

template <class NuT, class MtT, class TBurnT, class RhoPT, class P0T, class KappaT, class CpT, class ElemT>
struct PhysicalProperties
{
  constexpr static ElemT kappa = kae::detail::ToFloatV<KappaT, ElemT>;
  constexpr static ElemT nu    = kae::detail::ToFloatV<NuT, ElemT>;
  constexpr static ElemT mt    = kae::detail::ToFloatV<MtT, ElemT>;
  constexpr static ElemT TBurn = kae::detail::ToFloatV<TBurnT, ElemT>;
  constexpr static ElemT rhoP  = kae::detail::ToFloatV<RhoPT, ElemT>;
  constexpr static ElemT P0    = kae::detail::ToFloatV<P0T, ElemT>;
  constexpr static ElemT Cp    = kae::detail::ToFloatV<CpT, ElemT>;
  constexpr static ElemT R     = (kappa - static_cast<ElemT>(1.0)) / kappa * Cp;
  constexpr static ElemT H0    = Cp * TBurn;
};

}