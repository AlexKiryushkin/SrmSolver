#pragma once

#include <gcem.hpp>

#include "gas_state.h"
#include "to_float.h"

namespace kae {

template <class ElemT>
struct PhysicalPropertiesData
{
    PhysicalPropertiesData(ElemT nu_, ElemT mt_, ElemT TBurn_, ElemT rhoP_, ElemT P0_, ElemT kappa_, ElemT Cp_, ElemT fCritical_, ElemT initialSBurn_)
    {
        ElemT kappaDim = kappa_;
        ElemT gammaComplexDim = gcem::sqrt(kappaDim * gcem::pow(2 / (kappaDim + 1), (kappaDim + 1) / (kappaDim - 1)));
        ElemT nuDim = nu_;
        ElemT mtDim = mt_;
        ElemT TBurnDim = TBurn_;
        ElemT rhoPDim = rhoP_;
        ElemT P0Dim = P0_;
        ElemT CpDim = Cp_;
        ElemT RDim = (kappaDim - static_cast<ElemT>(1.0)) / kappaDim * CpDim;
        ElemT H0Dim = CpDim * TBurnDim;

        uScale = gcem::sqrt((kappaDim - 1) / kappaDim * H0Dim);
        ElemT pComplex = -initialSBurn_ * mtDim * uScale / gammaComplexDim / fCritical_;
        pScale = gcem::pow(pComplex, 1 / (1 - nuDim));

        rhoScale = pScale / uScale / uScale;
        TScale = TBurnDim;

        kappa = kappaDim;
        gammaComplex = gammaComplexDim;
        nu = nuDim;
        mt = mtDim * uScale / pComplex;
        TBurn = TBurnDim / TScale;
        rhoP = rhoPDim * uScale * uScale / pScale;
        P0 = P0Dim / pScale;
        Cp = CpDim * TScale / uScale / uScale;
        R = RDim * TScale / uScale / uScale;
        H0 = H0Dim / uScale / uScale;
    }

    ElemT uScale;
    ElemT pScale;

    ElemT rhoScale;
    ElemT TScale;

    ElemT kappa;
    ElemT gammaComplex;
    ElemT nu;
    ElemT mt;
    ElemT TBurn;
    ElemT rhoP;
    ElemT P0;
    ElemT Cp;
    ElemT R;
    ElemT H0;
};

struct BurningRate
{
  template <class GasStateT, class ElemType = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT& state, ElemType nu, ElemType mt, ElemType rhoP) -> typename GasStateT::ElemType
  {
    return get(P::get(state), nu, mt, rhoP);
  }

  template <class ElemT, class = std::enable_if_t<std::is_floating_point<ElemT>::value>>
  HOST_DEVICE static ElemT get(ElemT p, ElemT nu, ElemT mt, ElemT rhoP)
  {
    return -mt * std::pow(p, nu) / rhoP;
  }
};

template <class ElemT>
std::ostream& operator<<(std::ostream& os, PhysicalPropertiesData<ElemT> physicalParameters)
{
  os << "kappa: "         << physicalParameters.kappa        << "\n";
  os << "gamma complex: " << physicalParameters.gammaComplex << "\n";
  os << "nu: "            << physicalParameters.nu           << "\n";
  os << "mt: "            << physicalParameters.mt           << "\n";
  os << "TBurn: "         << physicalParameters.TBurn        << "\n";
  os << "rhoP: "          << physicalParameters.rhoP         << "\n";
  os << "P0: "            << physicalParameters.P0           << "\n";
  os << "Cp: "            << physicalParameters.Cp           << "\n";
  os << "R: "             << physicalParameters.R            << "\n";
  os << "H0: "            << physicalParameters.H0           << "\n";
  return os;
}

} // namespace kae
