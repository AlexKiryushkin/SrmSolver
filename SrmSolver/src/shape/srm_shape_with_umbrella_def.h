#pragma once

#include <gcem.hpp>

#include "math_utilities.h"

namespace kae {

template <class GpuGridT>
HOST_DEVICE auto SrmShapeWithUmbrella<GpuGridT>::F(ElemType x) -> ElemType
{
  if (x < l)
  {
    return R0;
  }

  if (x < L + nozzle_lengthening)
  {
    return R0;
  }

  if (x < L + nozzle_lengthening + static_cast<ElemType>(0.55) * l_nozzle)
  {
    return R0 * (static_cast<ElemType>(0.75) + 
                 static_cast<ElemType>(0.25) * cos(k_cos * (x - nozzle_lengthening - L)));
  }

  return k_line() * (x - nozzle_lengthening) + b_line();
}

template <class GpuGridT>
HOST_DEVICE auto SrmShapeWithUmbrella<GpuGridT>::F_prime(ElemType x) -> ElemType
{
  if (x < l)
  {
    return 0;
  }

  if (x < L + nozzle_lengthening)
  {
    return 0;
  }

  if (x < L + nozzle_lengthening + static_cast<ElemType>(0.55) * l_nozzle)
  {
    return static_cast<ElemType>(-0.25) * R0 * k_cos * sin(k_cos * (x - nozzle_lengthening - L));
  }

  return k_line();
}

template <class GpuGridT>
DEVICE EBoundaryCondition SrmShapeWithUmbrella<GpuGridT>::getBoundaryCondition(ElemType x, ElemType y)
{
  if ((x > x_left) && (x < x_junc) && (y - y_bottom >= R0 - static_cast<ElemType>(1e-4)) && (y - y_bottom < Rk))
  {
    return EBoundaryCondition::eMassFlowInlet;
  }

  if (std::fabs(x - x_right) < static_cast<ElemType>(0.1) * GpuGridT::hx)
  {
    return EBoundaryCondition::ePressureOutlet;
  }
  
  return EBoundaryCondition::eWall;
}

template <class GpuGridT>
DEVICE bool SrmShapeWithUmbrella<GpuGridT>::shouldApplyScheme(unsigned i, unsigned j)
{
  return i * GpuGridT::hx <= x_junc + l_nozzle;
}

template <class GpuGridT>
DEVICE auto SrmShapeWithUmbrella<GpuGridT>::getRadius(unsigned i, unsigned j) -> ElemType
{
  return getRadius(i * GpuGridT::hx, j * GpuGridT::hy);
}

template <class GpuGridT>
DEVICE auto SrmShapeWithUmbrella<GpuGridT>::getRadius(ElemType x, ElemType y) -> ElemType
{
  return y - y_bottom;
}

template <class GpuGridT>
DEVICE constexpr auto SrmShapeWithUmbrella<GpuGridT>::getInitialSBurn() -> ElemType
{
  return 2 * static_cast<ElemType>(M_PI) * 
    (R0 * L +
      H * h + 
      R0 * R0 +
      (1 + gcem::sin(alpha)) / gcem::sin(alpha) * R0 * H + 
      static_cast<ElemType>(0.5) * (1 + gcem::sin(alpha)) / gcem::sin(alpha) * H * H);
}

template <class GpuGridT>
DEVICE constexpr auto SrmShapeWithUmbrella<GpuGridT>::getFCritical() -> ElemType
{
  return static_cast<ElemType>(M_PI) * rkr * rkr;
}

template <class GpuGridT>
DEVICE bool SrmShapeWithUmbrella<GpuGridT>::isChamber(ElemType x, ElemType y)
{
  return (x >= x_left) && (x <= x_junc);
}

template <class GpuGridT>
DEVICE bool SrmShapeWithUmbrella<GpuGridT>::isBurningSurface(ElemType x, ElemType y)
{
  return (x > x_left) && (x < x_junc) && 
         (y >= y_bottom + R0 - static_cast<ElemType>(1e-4)) && (y < y_bottom + Rk);
}

template <class GpuGridT>
DEVICE bool SrmShapeWithUmbrella<GpuGridT>::isPointOnGrain(ElemType x, ElemType y)
{
  return (x > x_left) && (x < x_junc) &&
         (y - y_bottom >= R0 - static_cast<ElemType>(1e-4)) && (y < y_bottom + Rk);
}

template <class GpuGridT>
HOST_DEVICE auto SrmShapeWithUmbrella<GpuGridT>::operator()(unsigned i, unsigned j) const -> ElemType
{
  ElemType x = i * GpuGridT::hx;
  ElemType y = j * GpuGridT::hy;

  bool Zone1 = x <= x_left && y >= y_bottom && y <= y_bottom + R0;
  bool Zone2 = x <= x_left && y <= y_bottom;
  bool Zone3 = x >= x_left && x <= x_right && y <= y_bottom;
  bool Zone4 = x >= x_right && y <= y_bottom;
  bool Zone5 = x >= x_right && y >= y_bottom && y - y_bottom <= F(x_right - x_left);
  bool Zone6 = y - y_bottom >= k_normal_line() * (x - x_left) + b_normal_line() && y - y_bottom >= F(x_right - x_left);
  bool Zone7 = y - y_bottom <= k_normal_line() * (x - x_left) + b_normal_line() && y - y_bottom - R0 - cos(alpha / 2) / sin(alpha / 2) * (x - x_left - l - h) <= 0 && y - y_bottom >= F(x - x_left);
  bool Zone8 = y - y_bottom - R0 - cos(alpha / 2) / sin(alpha / 2) * (x - x_left - l - h) >= 0 && cos(alpha) * (y - y_bottom - R0) + sin(alpha) * (x - x_left - l - h) >= 0 && sin(alpha) * (y - y_bottom - R0) - cos(alpha) * (x - x_left - l - h) - H / sin(alpha) <= 0;
  bool Zone9 = sin(alpha) * (y - y_bottom - R0) - cos(alpha) * (x - x_left - l - h) - H / sin(alpha) >= 0 && x - x_left - l - h + cos(alpha) / sin(alpha) * H >= 0 && y - y_bottom >= F(x - x_left);
  bool Zone10 = x - x_left - l - h + cos(alpha) / sin(alpha) * H <= 0 && x - x_left - l + cos(alpha) / sin(alpha) * H >= 0 && y - y_bottom - R0 - H >= 0;
  bool Zone11 = x - x_left - l + cos(alpha) / sin(alpha) * H <= 0 && sin(alpha) * (y - y_bottom - R0) - cos(alpha) * (x - x_left - l) - H / sin(alpha) >= 0 && cos(alpha / 2) * (y - y_bottom - R0) + sin(alpha / 2) * (x - x_left - l) >= 0;
  bool Zone12 = sin(alpha) * (y - y_bottom - R0) - cos(alpha) * (x - x_left - l) - H / sin(alpha) <= 0 && cos(alpha) * (y - y_bottom - R0) + sin(alpha) * (x - x_left - l) <= 0 && cos(alpha / 2) * (y - y_bottom - R0) + sin(alpha / 2) * (x - x_left - l) >= 0;
  bool Zone13 = cos(alpha / 2) * (y - y_bottom - R0) + sin(alpha / 2) * (x - x_left - l) <= 0 && y - y_bottom >= F(x - x_left) && x - x_left >= 0;
  bool Zone14 = x - x_left <= 0 && y - y_bottom >= R0;
  bool Zone15 = y - y_bottom <= F(x - x_left) && x - x_left <= l;
  bool Zone16 = y - y_bottom <= F(x - x_left) && x - x_left >= l + h;
  bool Zone17 = y - y_bottom <= F(x - x_left) && x - x_left <= l + h && x - x_left >= l && sin(alpha) * (y - y_bottom - R0) - cos(alpha) * (x - x_left - l - h) <= 0;
  bool Zone18 = y - y_bottom >= R0 && y - y_bottom <= R0 + H && cos(alpha) * (y - y_bottom - R0) + sin(alpha) * (x - x_left - l - h) <= 0 && cos(alpha) * (y - y_bottom - R0) + sin(alpha) * (x - x_left - l) >= 0 && sin(alpha) * (y - y_bottom - R0) - cos(alpha) * (x - x_left - l) >= 0;
  bool Zone19 = sin(alpha) * (y - y_bottom - R0) - cos(alpha) * (x - x_left - l - h) >= 0 && sin(alpha) * (y - y_bottom - R0) - cos(alpha) * (x - x_left - l) <= 0;

  if (Zone1)
  {
    return x_left - x;
  }

  if (Zone2)
  {
    return std::hypot(x - x_left, y - y_bottom);
  }

  if (Zone3)
  {
    return y_bottom - y;
  }

  if (Zone4)
  {
    return std::hypot(x - x_right, y - y_bottom);
  }

  if (Zone5)
  {
    return x - x_right;
  }

  if (Zone6)
  {
    return std::hypot(x - x_right, y - y_bottom - F(x_right - x_left));
  }

  if (Zone7)
  {
    return (y - y_bottom - F(x - x_left)) / std::hypot(static_cast<ElemType>(1.0), F_prime(x - x_left));
  }

  if (Zone8)
  {
    return cos(alpha) * (y - y_bottom - R0) + sin(alpha) * (x - x_left - l - h);
  }

  if (Zone9)
  {
    return std::hypot(x - x_left - l - h + cos(alpha) / sin(alpha) * H, y - y_bottom - R0 - H);
  }

  if (Zone10)
  {
    return y - y_bottom - R0 - H;
  }

  if (Zone11)
  {
    return std::hypot(x - x_left - l + cos(alpha) / sin(alpha) * H, y - y_bottom - R0 - H);
  }

  if (Zone12)
  {
    return -(cos(alpha)*(y - y_bottom - R0) + sin(alpha)*(x - x_left - l));
  }

  if (Zone13)
  {
    return y - y_bottom - F(x - x_left);
  }

  if (Zone14)
  {
    return std::hypot(x - x_left, y - y_bottom - R0);
  }

  if (Zone15)
  {
    return absmin((y - y_bottom - F(x - x_left)) / std::hypot(static_cast<ElemType>(1.0), F_prime(x - x_left)),
                   y_bottom - y, 
                   x_left - x);
  }

  if (Zone16)
  {
    return absmin((y - y_bottom - F(x - x_left)) / std::hypot(static_cast<ElemType>(1.0), F_prime(x - x_left)),
                   y_bottom - y,
                   x - x_right);
  }

  if (Zone17)
  {
    ElemType val1 = -std::hypot(x - x_left - l, y - y_bottom - R0);
    ElemType val2 = -std::hypot(x - x_left - l - h, y - y_bottom - R0);
    ElemType val3 = absmin(val1, val2);

    return absmin(val3, y_bottom - y);
  }

  if (Zone18)
  {
    ElemType val1 = cos(alpha) * (y - y_bottom - R0) + sin(alpha) * (x - x_left - l - h);
    ElemType val2 = -(cos(alpha) * (y - y_bottom - R0) + sin(alpha) * (x - x_left - l));
    ElemType val3 = absmin(val1, val2);

    return absmin(val3, y - y_bottom - R0 - H);
  }

  if (Zone19)
  {
    ElemType val1 = -std::hypot(x - x_left - l, y - y_bottom - R0);
    ElemType val2 = cos(alpha) * (y - y_bottom - R0) + sin(alpha) * (x - x_left - l - h);
    ElemType val3 = absmin(val1, val2);

    return absmin(val3, y_bottom - y);

  }

  return 0;
}

} // namespace kae
