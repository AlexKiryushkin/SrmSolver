#pragma once

#include <gcem.hpp>

#include "gas_state.h"
#include "to_float.h"

namespace kae {

template <class NuT, class MtT, class TBurnT, class RhoPT, class P0T, class KappaT, class CpT,
          class ShapeT, class ElemT = typename ShapeT::ElemType>
struct PhysicalProperties
{
private:
  
  constexpr static ElemT kappaDim        = detail::ToFloatV<KappaT, ElemT>;
  constexpr static ElemT gammaComplexDim = gcem::sqrt(kappaDim * gcem::pow(2 / (kappaDim + 1), (kappaDim + 1) / (kappaDim - 1)));
  constexpr static ElemT nuDim           = detail::ToFloatV<NuT, ElemT>;
  constexpr static ElemT mtDim           = detail::ToFloatV<MtT, ElemT>;
  constexpr static ElemT TBurnDim        = detail::ToFloatV<TBurnT, ElemT>;
  constexpr static ElemT rhoPDim         = detail::ToFloatV<RhoPT, ElemT>;
  constexpr static ElemT P0Dim           = detail::ToFloatV<P0T, ElemT>;
  constexpr static ElemT CpDim           = detail::ToFloatV<CpT, ElemT>;
  constexpr static ElemT RDim            = (kappaDim - static_cast<ElemT>(1.0)) / kappaDim * CpDim;
  constexpr static ElemT H0Dim           = CpDim * TBurnDim;

  constexpr static ElemT uScale   = gcem::sqrt((kappaDim - 1) / kappaDim * H0Dim);
  constexpr static ElemT pComplex = -ShapeT::getInitialSBurn() * mtDim * uScale / gammaComplexDim / ShapeT::getFCritical();
  constexpr static ElemT pScale   = gcem::pow(pComplex, 1 / (1 - nuDim));

  constexpr static ElemT rhoScale = pScale / uScale / uScale;
  constexpr static ElemT TScale   = TBurnDim;

public:

  constexpr static ElemT kappa = kappaDim;
  constexpr static ElemT gammaComplex = gammaComplexDim;
  constexpr static ElemT nu    = nuDim;
  constexpr static ElemT mt    = mtDim * uScale / pComplex;
  constexpr static ElemT TBurn = TBurnDim / TScale;
  constexpr static ElemT rhoP  = rhoPDim * uScale * uScale / pScale;
  constexpr static ElemT P0    = P0Dim / pScale;
  constexpr static ElemT Cp    = CpDim * TScale / uScale / uScale;
  constexpr static ElemT R     = RDim * TScale  / uScale / uScale;
  constexpr static ElemT H0    = H0Dim / uScale / uScale;
};

template <class PhysicalPropertiesT>
struct BurningRate
{
  template <class GasStateT, class ElemType = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT& state) -> typename GasStateT::ElemType
  {
    return get(P::get(state));
  }

  template <class ElemT, class = std::enable_if_t<std::is_floating_point<ElemT>::value>>
  HOST_DEVICE static ElemT get(ElemT p)
  {
    return -PhysicalPropertiesT::mt * std::pow(p, PhysicalPropertiesT::nu) / PhysicalPropertiesT::rhoP;
  }
};

template <class NuT, class MtT, class TBurnT, class RhoPT, class P0T, class KappaT, class CpT,
          class ShapeT, class ElemT>
std::ostream& operator<<(std::ostream& os, PhysicalProperties<NuT, MtT, TBurnT, RhoPT, P0T, KappaT, CpT, ShapeT, ElemT> p)
{
  using PhysicalPropertiesT = PhysicalProperties<NuT, MtT, TBurnT, RhoPT, P0T, KappaT, CpT, ShapeT, ElemT>;
  os << "kappa: "         << PhysicalPropertiesT::kappa        << "\n";
  os << "gamma complex: " << PhysicalPropertiesT::gammaComplex << "\n";
  os << "nu: "            << PhysicalPropertiesT::nu           << "\n";
  os << "mt: "            << PhysicalPropertiesT::mt           << "\n";
  os << "TBurn: "         << PhysicalPropertiesT::TBurn        << "\n";
  os << "rhoP: "          << PhysicalPropertiesT::rhoP         << "\n";
  os << "P0: "            << PhysicalPropertiesT::P0           << "\n";
  os << "Cp: "            << PhysicalPropertiesT::Cp           << "\n";
  os << "R: "             << PhysicalPropertiesT::R            << "\n";
  os << "H0: "            << PhysicalPropertiesT::H0           << "\n";
  return os;
}

} // namespace kae
