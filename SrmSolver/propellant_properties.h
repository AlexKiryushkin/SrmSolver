#pragma once

#include <gcem.hpp>

#include "gas_state.h"
#include "to_float.h"

namespace kae {

template <class NuT, class MtT, class TBurnT, class RhoPT, class P0T, class KappaT, class CpT,
          class ShapeT, class ElemT = typename ShapeT::ElemType>
struct PropellantProperties
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
  constexpr static ElemT nu    = nuDim;
  constexpr static ElemT mt    = mtDim * uScale / pComplex;
  constexpr static ElemT TBurn = TBurnDim / TScale;
  constexpr static ElemT rhoP  = rhoPDim * uScale * uScale / pScale;
  constexpr static ElemT P0    = P0Dim / pScale;
  constexpr static ElemT Cp    = CpDim * TScale / uScale / uScale;
  constexpr static ElemT R     = RDim * TScale  / uScale / uScale;
  constexpr static ElemT H0    = H0Dim / uScale / uScale;
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
