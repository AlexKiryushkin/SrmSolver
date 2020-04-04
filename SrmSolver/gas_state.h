#pragma once

#include "cuda_includes.h"

#include "cuda_float_types.h"
#include "to_float.h"

namespace kae {

template <class KappaType, class CpType, class ElemT>
struct alignas(16) GasState
{
  using ElemType = ElemT;

  constexpr static ElemT kappa = detail::ToFloatV<KappaType, ElemT>;
  constexpr static ElemT Cp    = detail::ToFloatV<CpType, ElemT>;
  constexpr static ElemT R     = (kappa - static_cast<ElemT>(1.0)) / kappa * Cp;

  ElemT rho;
  ElemT ux;
  ElemT uy;
  ElemT p;
};

struct Rho
{
  template <class GasStateT>
  __host__ __device__ auto operator()(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static auto get(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return state.rho;
  }
};

struct P
{
  template <class GasStateT>
  __host__ __device__ auto operator()(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static auto get(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return state.p;
  }
};

struct VelocitySquared
{
  template <class GasStateT>
  __host__ __device__ auto operator()(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static auto get(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return state.ux * state.ux + state.uy * state.uy;
  }
};

struct Velocity
{
  template <class GasStateT>
  __host__ __device__ auto operator()(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static auto get(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return std::sqrt(VelocitySquared::get(state));
  }
};

struct MassFluxX
{
  template <class GasStateT>
  __host__ __device__ auto operator()(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static auto get(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return state.rho * state.ux;
  }
};

struct MassFluxY
{
  template <class GasStateT>
  __host__ __device__ auto operator()(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static auto get(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return state.rho * state.uy;
  }
};

struct MomentumFluxXx
{
  template <class GasStateT>
  __host__ __device__ auto operator()(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static auto get(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return state.rho * state.ux * state.ux + state.p;
  }
};

struct MomentumFluxXy
{
  template <class GasStateT>
  __host__ __device__ auto operator()(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static auto get(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return state.rho * state.ux * state.uy;
  }
};

struct MomentumFluxYy
{
  template <class GasStateT>
  __host__ __device__ auto operator()(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static auto get(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return state.rho * state.uy * state.uy + state.p;
  }
};

struct RhoEnergy
{
  template <class GasStateT>
  __host__ __device__ auto operator()(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static auto get(const GasStateT & state) -> typename GasStateT::ElemType
  {
    using ElemType = typename GasStateT::ElemType;
    constexpr ElemType multiplier = static_cast<ElemType>(1.0) / (GasStateT::kappa - static_cast<ElemType>(1.0));
    return multiplier * state.p + static_cast<ElemType>(0.5) * state.rho * VelocitySquared::get(state);
  }
};

struct Energy
{
  template <class GasStateT>
  __host__ __device__ auto operator()(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static auto get(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return RhoEnergy::get(state) / state.rho;
  }
};

struct EnthalpyFluxX
{
  template <class GasStateT>
  __host__ __device__ auto operator()(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static auto get(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return RhoEnergy::get(state) * state.ux + state.ux * state.p;
  }
};

struct EnthalpyFluxY
{
  template <class GasStateT>
  __host__ __device__ auto operator()(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static auto get(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return RhoEnergy::get(state) * state.uy + state.uy * state.p;
  }
};

struct SonicSpeedSquared
{
  template <class GasStateT>
  __host__ __device__ auto operator()(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static auto get(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return GasStateT::kappa * state.p / state.rho;
  }
};

struct SonicSpeed
{
  template <class GasStateT>
  __host__ __device__ auto operator()(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static auto get(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return std::sqrt(SonicSpeedSquared::get(state));
  }
};

struct Mach
{
  template <class GasStateT>
  __host__ __device__ auto operator()(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static auto get(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return std::sqrt(VelocitySquared::get(state) / SonicSpeedSquared::get(state));
  }
};

struct Temperature
{
  template <class GasStateT>
  __host__ __device__ auto operator()(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static auto get(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return state.p / state.rho / GasStateT::R;
  }
};

struct Rotate
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  __host__ __device__ GasStateT operator()(const GasStateT & state, ElemT nx, ElemT ny)
  {
    return get(state, nx, ny);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  __host__ __device__ static GasStateT get(const GasStateT & state, ElemT nx, ElemT ny)
  {
    ElemT newUx = state.ux * nx + state.uy * ny;
    ElemT newUy = -state.ux * ny + state.uy * nx;
    return { state.rho, newUx, newUy, state.p };
  }
};

struct ReverseRotate
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  __host__ __device__ GasStateT operator()(const GasStateT & state, ElemT nx, ElemT ny)
  {
    return get(state, nx, ny);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  __host__ __device__ static GasStateT get(const GasStateT & state, ElemT nx, ElemT ny)
  {
    ElemT newUx = state.ux * nx - state.uy * ny;
    ElemT newUy = state.ux * ny + state.uy * nx;
    return { state.rho, newUx, newUy, state.p };
  }
};

struct WaveSpeedX
{
  template <class GasStateT>
  __host__ __device__ auto operator()(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static auto get(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return SonicSpeed::get(state) + std::fabs(state.ux);
  }
};

struct WaveSpeedY
{
  template <class GasStateT>
  __host__ __device__ auto operator()(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static auto get(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return SonicSpeed::get(state) + std::fabs(state.uy);
  }
};

struct WaveSpeedXY
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  __host__ __device__ auto operator()(const GasStateT & state) -> CudaFloatT<2U, ElemT>
  {
    return get(state);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  __host__ __device__ static auto get(const GasStateT & state) -> CudaFloatT<2U, ElemT>
  {
    return { WaveSpeedX::get(state), WaveSpeedY::get(state) };
  }
};

struct WaveSpeed
{
  template <class GasStateT>
  __host__ __device__ auto operator()(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static auto get(const GasStateT & state) -> typename GasStateT::ElemType
  {
    return SonicSpeed::get(state) + Velocity::get(state);
  }
};

struct MirrorState
{
  template <class GasStateT>
  __host__ __device__ GasStateT operator()(const GasStateT & state)
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static GasStateT get(const GasStateT & state)
  {
    return { state.rho, -state.ux, state.uy, state.p };
  }
};

struct ConservativeVariables
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  __host__ __device__ auto operator()(const GasStateT & state) -> CudaFloatT<4U, ElemT>
  {
    return get(state);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  __host__ __device__ static auto get(const GasStateT & state) -> CudaFloatT<4U, ElemT>
  {
    return { Rho::get(state), MassFluxX::get(state), MassFluxY::get(state), RhoEnergy::get(state) };
  }
};

struct XFluxes
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  __host__ __device__ auto operator()(const GasStateT & state) -> CudaFloatT<4U, ElemT>
  {
    return get(state);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  __host__ __device__ static auto get(const GasStateT & state) -> CudaFloatT<4U, ElemT>
  {
    return { MassFluxX::get(state),
             MomentumFluxXx::get(state),
             MomentumFluxXy::get(state),
             EnthalpyFluxX::get(state) };
  }
};

struct YFluxes
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  __host__ __device__ auto operator()(const GasStateT & state) -> CudaFloatT<4U, ElemT>
  {
    return get(state);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  __host__ __device__ static auto get(const GasStateT & state) -> CudaFloatT<4U, ElemT>
  {
    return { MassFluxY::get(state),
             MomentumFluxXy::get(state),
             MomentumFluxYy::get(state),
             EnthalpyFluxY::get(state) };
  }
};

struct SourceTerm
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  __host__ __device__ auto operator()(const GasStateT & state) -> CudaFloatT<4U, ElemT>
  {
    return get(state);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  __host__ __device__ static auto get(const GasStateT & state) -> CudaFloatT<4U, ElemT>
  {
    return { MassFluxY::get(state),
             MomentumFluxXy::get(state),
             MassFluxY::get(state) * state.uy,
             EnthalpyFluxY::get(state) };
  }
};

struct ConservativeToGasState
{
  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  __host__ __device__ GasStateT operator()(const CudaFloatT<4U, ElemT> & state)
  {
    return get<GasStateT>(state);
  }

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  __host__ __device__ static GasStateT get(const CudaFloatT<4U, ElemT> & conservativeVariables)
  {
    const ElemT ux = conservativeVariables.y / conservativeVariables.x;
    const ElemT uy = conservativeVariables.z / conservativeVariables.x;
    const ElemT p = (GasStateT::kappa - static_cast<ElemT>(1.0)) *
                    (conservativeVariables.w - static_cast<ElemT>(0.5) * conservativeVariables.x * (ux * ux + uy * uy));
    return GasStateT{ conservativeVariables.x, ux, uy, p };
  }
};

} // namespace kae
