#pragma once

#include <cmath>

#include "types.h"

namespace kae {

template <class ElemType>
constexpr ElemType sqr(ElemType value)
{
  return value * value;
}

struct GasState
{
  using ElemType = ElemT;

  constexpr static ElemType kappa = static_cast<ElemType>(1.23);

  ElemT rho;
  ElemT u;
  ElemT p;
};

struct Rho
{
  static GasState::ElemType get(const GasState& gasState)
  {
    return gasState.rho;
  }
};

struct U
{
  static GasState::ElemType get(const GasState& gasState)
  {
    return gasState.u;
  }
};

struct MinusU
{
  static GasState::ElemType get(const GasState& gasState)
  {
    return -gasState.u;
  }
};

struct P
{
  static GasState::ElemType get(const GasState& gasState)
  {
    return gasState.p;
  }
};

struct MassFlux
{
  static GasState::ElemType get(const GasState& gasState)
  {
    return gasState.rho * gasState.u;
  }
};

struct RhoEnergyFlux
{
  static GasState::ElemType get(const GasState& gasState)
  {
    using ElemType = GasState::ElemType;
    constexpr ElemType multiplier = static_cast<ElemType>(1.0) / (GasState::kappa - static_cast<ElemType>(1.0));
    return multiplier * gasState.p + static_cast<ElemType>(0.5) * gasState.rho * gasState.u * gasState.u;
  }
};

struct MomentumFlux
{
  static GasState::ElemType get(const GasState& gasState)
  {
    return gasState.rho * gasState.u * gasState.u + gasState.p;
  }
};

struct RhoEnthalpyFlux
{
  static GasState::ElemType get(const GasState& gasState)
  {
    return gasState.u * (RhoEnergyFlux::get(gasState) + gasState.p);
  }
};

struct SonicSpeed
{
  static auto get(const GasState& state) -> GasState::ElemType
  {
    return std::sqrt(GasState::kappa * state.p / state.rho);
  }
};

struct Mach
{
  static auto get(const GasState& state) -> GasState::ElemType
  {
    return std::fabs(state.u) / kae::SonicSpeed::get(state);
  }
};

struct WaveSpeed
{
  auto operator()(const GasState& state) const { return get(state); }
  static auto get(const GasState& state) -> GasState::ElemType
  {
    return SonicSpeed::get(state) + std::fabs(U::get(state));
  }
};

struct MirrorState
{
  static GasState get(const GasState& gasState)
  {
    return GasState{ gasState.rho, -gasState.u, gasState.p };
  }
};

}
