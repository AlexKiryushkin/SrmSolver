#pragma once

#include "std_includes.h"

#include "boundary_condition.h"

namespace kae {

template <class GpuGridT>
class SrmShapeWithUmbrella
{
public:

  using ElemType = typename GpuGridT::ElemType;

  HOST_DEVICE ElemType operator()(unsigned i, unsigned j) const;

  HOST_DEVICE static bool shouldApplyScheme(unsigned i, unsigned j);

  HOST_DEVICE static bool isPointOnGrain(ElemType x, ElemType y);

  HOST_DEVICE static EBoundaryCondition getBoundaryCondition(ElemType x, ElemType y);

  HOST_DEVICE static ElemType getRadius(unsigned i, unsigned j);

  HOST_DEVICE static ElemType getRadius(ElemType x, ElemType y);

  HOST_DEVICE constexpr static ElemType getInitialSBurn();

  HOST_DEVICE constexpr static ElemType getFCritical();

  HOST_DEVICE static bool isChamber(ElemType x, ElemType y);

  HOST_DEVICE static bool isBurningSurface(ElemType x, ElemType y);

  HOST_DEVICE constexpr static ElemType getOutletCoordinate() { return x_right; }

private:

  HOST_DEVICE static ElemType F_prime(ElemType x);
  HOST_DEVICE static ElemType F(ElemType x);

private:

  constexpr static ElemType hx       = GpuGridT::hx;
  constexpr static ElemType hy       = GpuGridT::hy;
  constexpr static ElemType L        = static_cast<ElemType>(2.4);
  constexpr static ElemType R0       = static_cast<ElemType>(0.2);
  constexpr static ElemType Rk       = static_cast<ElemType>(0.9);
  constexpr static ElemType H        = static_cast<ElemType>(0.6);
  constexpr static ElemType rkr      = static_cast<ElemType>(0.1);
  constexpr static ElemType l        = static_cast<ElemType>(0.7);
  constexpr static ElemType h        = static_cast<ElemType>(0.12);
  constexpr static ElemType l_nozzle = static_cast<ElemType>(0.4);
  constexpr static ElemType alpha    = static_cast<ElemType>(M_PI / 3.0);

  constexpr static ElemType k_cos    = 2 * static_cast<ElemType>(M_PI) / l_nozzle;
  HOST_DEVICE static ElemType k_line()
  {
    return static_cast<ElemType>(-0.25) * R0 * k_cos * std::sin(static_cast<ElemType>(0.55) * k_cos * l_nozzle);
  }
  HOST_DEVICE static ElemType b_line()
  {
    return R0 * (static_cast<ElemType>(0.75) + 
      static_cast<ElemType>(0.25) * std::cos(static_cast<ElemType>(0.55) * k_cos * l_nozzle)) -
      k_line() * (L + static_cast<ElemType>(0.55) * l_nozzle);
  }

  constexpr static unsigned offsetPointsX = 32U;
  constexpr static unsigned offsetPointsY = 16U;

  constexpr static ElemType x_left   = (offsetPointsX + static_cast<ElemType>(0.5)) * hx;
  constexpr static ElemType x_junc   = x_left + L;
  constexpr static ElemType x_right  = x_junc + static_cast<ElemType>(1.5) * l_nozzle + hx / 2;
  constexpr static ElemType y_bottom = (offsetPointsY + static_cast<ElemType>(0.5)) * hy;

  HOST_DEVICE static ElemType k_normal_line() { return -1 / k_line(); }
  HOST_DEVICE static ElemType b_normal_line() { return F(x_right - x_left) + (x_right - x_left) / k_line(); }

  constexpr static ElemType nozzle_lengthening = static_cast<ElemType>(0.2);
};

} // namespace kae

#include "srm_shape_with_umbrella_def.h"
