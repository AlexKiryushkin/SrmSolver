#pragma once

#include <cuda_runtime_api.h>

#include "to_float.h"

namespace kae {

template <class KappaType, class CpType>
struct alignas(16) GasState
{
  constexpr static float kappa = detail::ToFloatV<KappaType>;
  constexpr static float Cp    = detail::ToFloatV<CpType>;
  constexpr static float R     = (kappa - 1.0f) / kappa * Cp;

  float rho;
  float ux;
  float uy;
  float p;
};

struct Rho
{
  template <class GasStateT>
  __host__ __device__ float operator()(const GasStateT & state)
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static float get(const GasStateT & state)
  {
    return state.rho;
  }
};

struct P
{
  template <class GasStateT>
  __host__ __device__ float operator()(const GasStateT & state)
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static float get(const GasStateT & state)
  {
    return state.p;
  }
};

struct VelocitySquared
{
  template <class GasStateT>
  __host__ __device__ float operator()(const GasStateT & state)
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static float get(const GasStateT & state)
  {
    return state.ux * state.ux + state.uy * state.uy;
  }
};

struct Velocity
{
  template <class GasStateT>
  __host__ __device__ float operator()(const GasStateT & state)
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static float get(const GasStateT & state)
  {
    return std::sqrt(VelocitySquared::get(state));
  }
};

struct MassFluxX
{
  template <class GasStateT>
  __host__ __device__ float operator()(const GasStateT & state)
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static float get(const GasStateT & state)
  {
    return state.rho * state.ux;
  }
};

struct MassFluxY
{
  template <class GasStateT>
  __host__ __device__ float operator()(const GasStateT & state)
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static float get(const GasStateT & state)
  {
    return state.rho * state.uy;
  }
};

struct MomentumFluxXx
{
  template <class GasStateT>
  __host__ __device__ float operator()(const GasStateT & state)
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static float get(const GasStateT & state)
  {
    return state.rho * state.ux * state.ux + state.p;
  }
};

struct MomentumFluxXy
{
  template <class GasStateT>
  __host__ __device__ float operator()(const GasStateT & state)
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static float get(const GasStateT & state)
  {
    return state.rho * state.ux * state.uy;
  }
};

struct MomentumFluxYy
{
  template <class GasStateT>
  __host__ __device__ float operator()(const GasStateT & state)
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static float get(const GasStateT & state)
  {
    return state.rho * state.uy * state.uy + state.p;
  }
};

struct RhoEnergy
{
  template <class GasStateT>
  __host__ __device__ float operator()(const GasStateT & state)
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static float get(const GasStateT & state)
  {
    constexpr float multiplier = 1.0f / (GasStateT::kappa - 1.0f);
    return multiplier * state.p + 0.5f * state.rho * VelocitySquared::get(state);
  }
};

struct Energy
{
  template <class GasStateT>
  __host__ __device__ float operator()(const GasStateT & state)
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static float get(const GasStateT & state)
  {
    return RhoEnergy::get(state) / state.rho;
  }
};

struct EnthalpyFluxX
{
  template <class GasStateT>
  __host__ __device__ float operator()(const GasStateT & state)
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static float get(const GasStateT & state)
  {
    return RhoEnergy::get(state) * state.ux + state.ux * state.p;
  }
};

struct EnthalpyFluxY
{
  template <class GasStateT>
  __host__ __device__ float operator()(const GasStateT & state)
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static float get(const GasStateT & state)
  {
    return RhoEnergy::get(state) * state.uy + state.uy * state.p;
  }
};

struct SonicSpeedSquared
{
  template <class GasStateT>
  __host__ __device__ float operator()(const GasStateT & state)
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static float get(const GasStateT & state)
  {
    return GasStateT::kappa * state.p / state.rho;
  }
};

struct SonicSpeed
{
  template <class GasStateT>
  __host__ __device__ float operator()(const GasStateT & state)
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static float get(const GasStateT & state)
  {
    return std::sqrt(SonicSpeedSquared::get(state));
  }
};

struct Mach
{
  template <class GasStateT>
  __host__ __device__ float operator()(const GasStateT & state)
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static float get(const GasStateT & state)
  {
    return std::sqrt(VelocitySquared::get(state) / SonicSpeedSquared::get(state));
  }
};

struct Temperature
{
  template <class GasStateT>
  __host__ __device__ float operator()(const GasStateT & state)
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static float get(const GasStateT & state)
  {
    return state.p / state.rho / GasStateT::R;
  }
};

struct Rotate
{
  template <class GasStateT>
  __host__ __device__ float operator()(const GasStateT & state, float nx, float ny)
  {
    return get(state, nx, ny);
  }

  template <class GasStateT>
  __host__ __device__ static GasStateT get(const GasStateT & state, float nx, float ny)
  {
    float newUx = state.ux * nx + state.uy * ny;
    float newUy = -state.ux * ny + state.uy * nx;
    return { state.rho, newUx, newUy, state.p };
  }
};

struct ReverseRotate
{
  template <class GasStateT>
  __host__ __device__ float operator()(const GasStateT & state, float nx, float ny)
  {
    return get(state, nx, ny);
  }

  template <class GasStateT>
  __host__ __device__ static GasStateT get(const GasStateT & state, float nx, float ny)
  {
    float newUx = state.ux * nx - state.uy * ny;
    float newUy = state.ux * ny + state.uy * nx;
    return { state.rho, newUx, newUy, state.p };
  }
};

struct WaveSpeedX
{
  template <class GasStateT>
  __host__ __device__ float operator()(const GasStateT & state)
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static float get(const GasStateT & state)
  {
    return SonicSpeed::get(state) + std::fabs(state.ux);
  }
};

struct WaveSpeedY
{
  template <class GasStateT>
  __host__ __device__ float operator()(const GasStateT & state)
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static float get(const GasStateT & state)
  {
    return SonicSpeed::get(state) + std::fabs(state.uy);
  }
};

struct WaveSpeedXY
{
  template <class GasStateT>
  __host__ __device__ float2 operator()(const GasStateT & state)
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static float2 get(const GasStateT & state)
  {
    return { WaveSpeedX::get(state), WaveSpeedY::get(state) };
  }
};

struct WaveSpeed
{
  template <class GasStateT>
  __host__ __device__ float operator()(const GasStateT & state)
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static float get(const GasStateT & state)
  {
    return SonicSpeed::get(state) + Velocity::get(state);
  }
};

struct MirrorState
{
  template <class GasStateT>
  __host__ __device__ float operator()(const GasStateT & state)
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
  template <class GasStateT>
  __host__ __device__ float4 operator()(const GasStateT & state)
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static float4 get(const GasStateT & state)
  {
    return { Rho::get(state), MassFluxX::get(state), MassFluxY::get(state), RhoEnergy::get(state) };
  }
};

struct XFluxes
{
  template <class GasStateT>
  __host__ __device__ float4 operator()(const GasStateT & state)
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static float4 get(const GasStateT & state)
  {
    return { MassFluxX::get(state),
             MomentumFluxXx::get(state),
             MomentumFluxXy::get(state),
             EnthalpyFluxX::get(state) };
  }
};

struct YFluxes
{
  template <class GasStateT>
  __host__ __device__ float4 operator()(const GasStateT & state)
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static float4 get(const GasStateT & state)
  {
    return { MassFluxY::get(state),
             MomentumFluxXy::get(state),
             MomentumFluxYy::get(state),
             EnthalpyFluxY::get(state) };
  }
};

struct SourceTerm
{
  template <class GasStateT>
  __host__ __device__ float4 operator()(const GasStateT & state)
  {
    return get(state);
  }

  template <class GasStateT>
  __host__ __device__ static float4 get(const GasStateT & state)
  {
    return { MassFluxY::get(state),
             MomentumFluxXy::get(state),
             MassFluxY::get(state) * state.uy,
             EnthalpyFluxY::get(state) };
  }
};

struct ConservativeToGasState
{
  template <class GasStateT>
  __host__ __device__ GasStateT operator()(const float4 & state)
  {
    return get<GasStateT>(state);
  }

  template <class GasStateT>
  __host__ __device__ static GasStateT get(const float4 & conservativeVariables)
  {
    const float ux = conservativeVariables.y / conservativeVariables.x;
    const float uy = conservativeVariables.z / conservativeVariables.x;
    const float p = (GasStateT::kappa - 1.0f) * (conservativeVariables.w - 0.5f * conservativeVariables.x * (ux * ux + uy * uy));
    return GasStateT{ conservativeVariables.x, ux, uy, p };
  }
};

} // namespace kae
