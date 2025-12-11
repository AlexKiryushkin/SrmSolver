#pragma once

#include "cuda_includes.h"

#include "cuda_float_types.h"
#include "math_utilities.h"
#include "matrix/matrix.h"

namespace kae {

template <class ElemT>
struct alignas(16) GasState
{
  using ElemType = ElemT;

  ElemT rho;
  ElemT ux;
  ElemT uy;
  ElemT p;
};

template <class ElemT>
struct GasParameters
{
    ElemT kappa;
    ElemT R;
};

struct IsValid
{
  template <class GasStateT>
  HOST_DEVICE bool operator()(const GasStateT & state)
  {
    return get(state);
  }

  template <class GasStateT>
  HOST_DEVICE static bool get(const GasStateT & state)
  {
    return isfinite(state.rho) && isfinite(state.ux) && isfinite(state.uy) && isfinite(state.p) &&
          (state.rho > 0) && (state.p > 0);
  }
};

struct Rho
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT& state, const GasParameters<ElemT>& dummy = {}) -> ElemT
  {
    return get(state, dummy);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& dummy = {}) -> ElemT
  {
    return state.rho;
  }
};

struct P
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT& state, const GasParameters<ElemT>& dummy = {}) -> ElemT
  {
    return get(state, dummy);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& dummy = {}) -> ElemT
  {
    return state.p;
  }
};

struct VelocitySquared
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT& state, const GasParameters<ElemT>& dummy = {}) -> ElemT
  {
    return get(state, dummy);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& dummy = {}) -> ElemT
  {
    return state.ux * state.ux + state.uy * state.uy;
  }
};

struct Velocity
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT& state, const GasParameters<ElemT>& dummy = {}) -> ElemT
  {
    return get(state, dummy);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT& state, const GasParameters<ElemT>& dummy = {}) -> ElemT
  {
    return std::sqrt(VelocitySquared::get(state, dummy));
  }
};

struct MassFluxX
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT& state, const GasParameters<ElemT>& dummy = {}) -> ElemT
  {
    return get(state, dummy);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& dummy = {}) -> ElemT
  {
    return state.rho * state.ux;
  }
};

struct MassFluxY
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT& state, const GasParameters<ElemT>& dummy = {}) -> ElemT
  {
    return get(state, dummy);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& dummy = {}) -> ElemT
  {
    return state.rho * state.uy;
  }
};

struct MomentumFluxXx
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT& state, const GasParameters<ElemT>& dummy = {}) -> ElemT
  {
    return get(state, dummy);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& dummy = {}) -> ElemT
  {
    return state.rho * state.ux * state.ux + state.p;
  }
};

struct MomentumFluxXy
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT& state, const GasParameters<ElemT>& dummy = {}) -> ElemT
  {
    return get(state, dummy);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& dummy = {}) -> ElemT
  {
    return state.rho * state.ux * state.uy;
  }
};

struct MomentumFluxYy
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT & state, const GasParameters<ElemT>& dummy = {}) -> ElemT
  {
    return get(state, dummy);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& dummy = {}) -> ElemT
  {
    return state.rho * state.uy * state.uy + state.p;
  }
};

struct RhoEnergy
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> ElemT
  {
    return get(state, gasParameters);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> ElemT
  {
    const ElemT multiplier = static_cast<ElemT>(1.0) / (gasParameters.kappa - static_cast<ElemT>(1.0));
    return multiplier * state.p + static_cast<ElemT>(0.5) * state.rho * VelocitySquared::get(state, gasParameters);
  }
};

struct Energy
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> ElemT
  {
    return get(state, gasParameters);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> ElemT
  {
    return RhoEnergy::get(state, gasParameters) / state.rho;
  }
};

struct EnthalpyFluxX
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> ElemT
  {
    return get(state, gasParameters);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> ElemT
  {
    return RhoEnergy::get(state, gasParameters) * state.ux + state.ux * state.p;
  }
};

struct EnthalpyFluxY
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> ElemT
  {
    return get(state, gasParameters);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> ElemT
  {
    return RhoEnergy::get(state, gasParameters) * state.uy + state.uy * state.p;
  }
};

struct SonicSpeedSquared
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> ElemT
  {
    return get(state, gasParameters);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> ElemT
  {
    return gasParameters.kappa * state.p / state.rho;
  }
};

struct SonicSpeed
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> ElemT
  {
    return get(state, gasParameters);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> ElemT
  {
    return std::sqrt(SonicSpeedSquared::get(state, gasParameters));
  }
};

struct Mach
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> ElemT
  {
    return get(state, gasParameters);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> ElemT
  {
    return std::sqrt(VelocitySquared::get(state, gasParameters) / SonicSpeedSquared::get(state, gasParameters));
  }
};

struct Temperature
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> ElemT
  {
    return get(state, gasParameters);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> ElemT
  {
    return state.p / state.rho / gasParameters.R;
  }
};

struct Rotate
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE GasStateT operator()(const GasStateT & state, ElemT nx, ElemT ny)
  {
    return get(state, nx, ny);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static GasStateT get(const GasStateT & state, ElemT nx, ElemT ny)
  {
    ElemT newUx = state.ux * nx + state.uy * ny;
    ElemT newUy = -state.ux * ny + state.uy * nx;
    return { state.rho, newUx, newUy, state.p };
  }
};

struct ReverseRotate
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE GasStateT operator()(const GasStateT & state, ElemT nx, ElemT ny)
  {
    return get(state, nx, ny);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static GasStateT get(const GasStateT & state, ElemT nx, ElemT ny)
  {
    ElemT newUx = state.ux * nx - state.uy * ny;
    ElemT newUy = state.ux * ny + state.uy * nx;
    return { state.rho, newUx, newUy, state.p };
  }
};

struct WaveSpeedX
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> ElemT
  {
    return get(state, gasParameters);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> ElemT
  {
    return SonicSpeed::get(state, gasParameters) + std::fabs(state.ux);
  }
};

struct WaveSpeedY
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> ElemT
  {
    return get(state, gasParameters);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> ElemT
  {
    return SonicSpeed::get(state, gasParameters) + std::fabs(state.uy);
  }
};

struct WaveSpeedXY
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> CudaFloat2T<ElemT>
  {
    return get(state, gasParameters);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> CudaFloat2T<ElemT>
  {
    return { WaveSpeedX::get(state, gasParameters), WaveSpeedY::get(state, gasParameters) };
  }
};

struct WaveSpeed
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> ElemT
  {
    return get(state, gasParameters);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> ElemT
  {
    return SonicSpeed::get(state, gasParameters) + Velocity::get(state, gasParameters);
  }
};

struct MirrorState
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE GasStateT operator()(const GasStateT & state)
  {
    return get(state);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static GasStateT get(const GasStateT & state)
  {
    return { state.rho, -state.ux, state.uy, state.p };
  }
};

struct ConservativeVariables
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> CudaFloat4T<ElemT>
  {
    return get(state, gasParameters);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> CudaFloat4T<ElemT>
  {
    return { Rho::get(state, gasParameters), MassFluxX::get(state, gasParameters), MassFluxY::get(state, gasParameters), RhoEnergy::get(state, gasParameters) };
  }
};

struct XFluxes
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> CudaFloat4T<ElemT>
  {
    return get(state, gasParameters);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> CudaFloat4T<ElemT>
  {
    return { MassFluxX::get(state, gasParameters),
             MomentumFluxXx::get(state, gasParameters),
             MomentumFluxXy::get(state, gasParameters),
             EnthalpyFluxX::get(state, gasParameters) };
  }
};

struct YFluxes
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> CudaFloat4T<ElemT>
  {
    return get(state, gasParameters);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> CudaFloat4T<ElemT>
  {
    return { MassFluxY::get(state, gasParameters),
             MomentumFluxXy::get(state, gasParameters),
             MomentumFluxYy::get(state, gasParameters),
             EnthalpyFluxY::get(state, gasParameters) };
  }
};

struct SourceTerm
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> CudaFloat4T<ElemT>
  {
    return get(state, gasParameters);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> CudaFloat4T<ElemT>
  {
    return { MassFluxY::get(state, gasParameters),
             MomentumFluxXy::get(state, gasParameters),
             MassFluxY::get(state, gasParameters) * state.uy,
             EnthalpyFluxY::get(state, gasParameters) };
  }
};

struct ConservativeToGasState
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE GasStateT operator()(const CudaFloat4T<ElemT> & state, const GasParameters<ElemT>& gasParameters)
  {
    return get<GasStateT>(state, gasParameters);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static GasStateT get(const CudaFloat4T<ElemT> & conservativeVariables, const GasParameters<ElemT>& gasParameters)
  {
    const ElemT ux = conservativeVariables.y / conservativeVariables.x;
    const ElemT uy = conservativeVariables.z / conservativeVariables.x;
    const ElemT p = (gasParameters.kappa - static_cast<ElemT>(1.0)) *
                    (conservativeVariables.w - static_cast<ElemT>(0.5) * conservativeVariables.x * (ux * ux + uy * uy));
    return GasStateT{ conservativeVariables.x, ux, uy, p };
  }
};

struct EigenValuesX
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> CudaFloat4T<ElemT>
  {
    return get(state, gasParameters);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> CudaFloat4T<ElemT>
  {
    const auto c = SonicSpeed::get(state, gasParameters);
    return { state.ux - c, state.ux, state.ux, state.ux + c };
  }
};

struct EigenValuesMatrixX
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> kae::Matrix<ElemT, 4, 4>
  {
    return get(state, gasParameters);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> kae::Matrix<ElemT, 4, 4>
  {
    const auto c = SonicSpeed::get(state, gasParameters);
    constexpr auto zero = static_cast<ElemT>(0);
    return kae::Matrix<ElemT, 4, 4>{ state.ux - c, zero,     zero,     zero,
                                     zero,         state.ux, zero,     zero,
                                     zero,         zero,     state.ux, zero,
                                     zero,         zero,     zero,     state.ux + c };
  }
};

template <bool uyIsZero>
struct LeftPrimitiveEigenVectorsX
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> kae::Matrix<ElemT, 4, 4>
  {
    return get(state, gasParameters);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> kae::Matrix<ElemT, 4, 4>
  {
    constexpr auto zero = static_cast<ElemT>(0.0);
    constexpr auto half = static_cast<ElemT>(0.5);

    const auto cReciprocal = 1 / SonicSpeed::get(state, gasParameters);
    const auto rhoReciprocal = 1 / state.rho;

    return kae::Matrix<ElemT, 4, 4>{
         zero,                          -half * cReciprocal, zero,  half * rhoReciprocal * cReciprocal * cReciprocal,
        -half * state.uy *rhoReciprocal, zero,               half,  half * state.uy * rhoReciprocal * cReciprocal * cReciprocal,
         half * state.uy *rhoReciprocal, zero,               half, -half * state.uy * rhoReciprocal * cReciprocal * cReciprocal,
         zero,                           half * cReciprocal, zero,  half * rhoReciprocal * cReciprocal * cReciprocal};
  }
};

template<>
struct LeftPrimitiveEigenVectorsX<true>
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT& state, const GasParameters<ElemT>& gasParameters) -> kae::Matrix<ElemT, 4, 4>
  {
    return get(state, gasParameters);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT& state, const GasParameters<ElemT>& gasParameters) -> kae::Matrix<ElemT, 4, 4>
  {
    constexpr auto zero = static_cast<ElemT>(0.0);
    constexpr auto half = static_cast<ElemT>(0.5);

    const auto cRec = 1 / SonicSpeed::get(state, gasParameters);
    const auto cRecSqr = sqr(cRec);
    const auto uy = state.uy;
    const auto rho = state.rho;
    const auto mult = 1 / (1 - rho * uy);

    return kae::Matrix<ElemT, 4, 4>{
         zero,     -cRec / 2, zero,        cRecSqr / rho / 2,
         mult,      zero,    -rho * mult, -cRecSqr * mult,
        -uy * mult, zero,     mult,        uy * cRecSqr * mult,
         zero,      cRec / 2, zero,        cRecSqr / rho / 2 };
  }
};

struct DispatchedLeftPrimitiveEigenVectorsX
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT& state, const GasParameters<ElemT>& gasParameters) -> kae::Matrix<ElemT, 4, 4>
  {
    return get(state, gasParameters);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT& state, const GasParameters<ElemT>& gasParameters) -> kae::Matrix<ElemT, 4, 4>
  {
    constexpr auto eps = std::numeric_limits<ElemT>::epsilon();
    return std::fabs(state.uy) > eps ? LeftPrimitiveEigenVectorsX<false>::get(state, gasParameters) :
                                       LeftPrimitiveEigenVectorsX<true>::get(state, gasParameters);
  }
};

template <bool uyIsZero>
struct RightPrimitiveEigenVectorsX
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> kae::Matrix<ElemT, 4, 4>
  {
    return get(state, gasParameters);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> kae::Matrix<ElemT, 4, 4>
  {
    constexpr auto zero = static_cast<ElemT>(0.0);
    constexpr auto one = static_cast<ElemT>(1.0);

    const auto c = SonicSpeed::get(state, gasParameters);
    return kae::Matrix<ElemT, 4, 4>{
      state.rho,        -state.rho / state.uy, state.rho / state.uy, state.rho,
     -c,                 zero,                 zero,                  c,
      zero,              one,                  one,                   zero,
      state.rho * c * c, zero,                 zero,                  state.rho * c * c };
  }
};

template <>
struct RightPrimitiveEigenVectorsX<true>
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT& state, const GasParameters<ElemT>& gasParameters) -> kae::Matrix<ElemT, 4, 4>
  {
    return get(state, gasParameters);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT& state, const GasParameters<ElemT>& gasParameters) -> kae::Matrix<ElemT, 4, 4>
  {
    constexpr auto zero = static_cast<ElemT>(0.0);
    constexpr auto one = static_cast<ElemT>(1.0);

    const auto c  = SonicSpeed::get(state, gasParameters);
    const auto uy = state.uy;
    return kae::Matrix<ElemT, 4, 4>{
      state.rho,         one,  state.rho, state.rho,
     -c,                 zero, zero,      c,
      zero,              uy,   one,       zero,
      state.rho * c * c, zero, zero,      state.rho * c * c };
  }
};

struct DispatchedRightPrimitiveEigenVectorsX
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT& state, const GasParameters<ElemT>& gasParameters) -> kae::Matrix<ElemT, 4, 4>
  {
    return get(state, gasParameters);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT& state, const GasParameters<ElemT>& gasParameters) -> kae::Matrix<ElemT, 4, 4>
  {
    constexpr auto eps = std::numeric_limits<ElemT>::epsilon();
    return std::fabs(state.uy) > eps ? RightPrimitiveEigenVectorsX<false>::get(state, gasParameters) :
                                       RightPrimitiveEigenVectorsX<true>::get(state, gasParameters);
  }
};

struct PrimitiveJacobianMatrixX
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE auto operator()(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> kae::Matrix<ElemT, 4, 4>
  {
    return get(state, gasParameters);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const GasStateT & state, const GasParameters<ElemT>& gasParameters) -> kae::Matrix<ElemT, 4, 4>
  {
    constexpr auto zero = static_cast<ElemT>(0.0);

    const auto c = SonicSpeed::get(state, gasParameters);
    return kae::Matrix<ElemT, 4, 4>{
      state.ux, state.rho,         zero,     zero,
      zero,     state.ux,          zero,     1 / state.rho,
      zero,     zero,              state.ux, zero,
      zero,     state.rho * c * c, zero,     state.ux};
  }
};

struct PrimitiveCharacteristicVariables
{

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  HOST_DEVICE static auto get(const kae::Matrix<ElemT, 4, 4> & leftEigenVectors,
                              const GasStateT & state) -> kae::Matrix<ElemT, 4, 1>
  {
    kae::Matrix<ElemT, 4, 1> characteristicVariables{ state.rho, state.ux, state.uy, state.p };
    return leftEigenVectors * characteristicVariables;
  }
};

} // namespace kae
