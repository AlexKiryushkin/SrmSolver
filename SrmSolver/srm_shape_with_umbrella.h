#pragma once

#include "std_includes.h"

#include "boundary_condition.h"

namespace kae {

template <class GpuGridT>
class SrmShapeWithUmbrella
{
public:

  using ElemType = typename GpuGridT::ElemType;

  __host__ __device__ ElemType operator()(unsigned i, unsigned j) const;

  __host__ __device__ static bool shouldApplyScheme(unsigned i, unsigned j);

  __host__ __device__ static bool isPointOnGrain(ElemType x, ElemType y);

  __host__ __device__ static EBoundaryCondition getBoundaryCondition(ElemType x, ElemType y);

  __host__ __device__ static ElemType getRadius(unsigned i, unsigned j);

private:

  __host__ __device__ ElemType F_prime(ElemType x) const;
  __host__ __device__ ElemType F(ElemType x) const;

private:

  constexpr static ElemType L        = static_cast<ElemType>(2.4);
  constexpr static ElemType R0       = static_cast<ElemType>(0.2);
  constexpr static ElemType Rk       = static_cast<ElemType>(0.9);
  constexpr static ElemType H        = static_cast<ElemType>(0.55);
  constexpr static ElemType rkr      = static_cast<ElemType>(0.1);
  constexpr static ElemType l        = static_cast<ElemType>(0.5);
  constexpr static ElemType h        = static_cast<ElemType>(0.08);
  constexpr static ElemType l_nozzle = static_cast<ElemType>(0.4);
  constexpr static ElemType alpha    = static_cast<ElemType>(M_PI_2);

  constexpr static ElemType k_cos    = 2 * static_cast<ElemType>(M_PI) / l_nozzle;
  __host__ __device__ static ElemType k_line()
  {
    return static_cast<ElemType>(-0.25) * R0 * k_cos * std::sin(static_cast<ElemType>(0.55) * k_cos * l_nozzle);
  }
  __host__ __device__ static ElemType b_line()
  {
    return R0 * (static_cast<ElemType>(0.75) + 
      static_cast<ElemType>(0.25) * std::cos(static_cast<ElemType>(0.55) * k_cos * l_nozzle)) -
      k_line() * (L + static_cast<ElemType>(0.55) * l_nozzle);
  }

  constexpr static unsigned offsetPoints = 20U;

  constexpr static ElemType x_left   = (offsetPoints + static_cast<ElemType>(0.5)) * GpuGridT::hx;
  constexpr static ElemType x_junc   = x_left + L;
  constexpr static ElemType x_right  = x_junc + static_cast<ElemType>(1.5) * l_nozzle + static_cast<ElemType>(0.5) * GpuGridT::hx;
  constexpr static ElemType y_bottom = (offsetPoints + static_cast<ElemType>(0.5)) * GpuGridT::hy;

  __host__ __device__ static ElemType k_normal_line() { return -1 / k_line(); }
  __host__ __device__ static ElemType b_normal_line() { return b_line() + (x_right - x_left) * (k_line() + 1 / k_line()); }

  constexpr static ElemType nozzle_lengthening = static_cast<ElemType>(0.1);
};

} // namespace kae

#include "srm_shape_with_umbrella_def.h"
