#pragma once

#include <cmath>

#include "math_utilities.h"

namespace kae {

template <class GpuGridT>
__host__ __device__ float SrmShapeWithUmbrella<GpuGridT>::F(float x) const
{
  if (x < l)
  {
    return R0;
  }

  if (x < L + nozzle_lengthening)
  {
    return R0;
  }

  if (x < L + nozzle_lengthening + 0.55f * l_nozzle)
  {
    return R0 * (0.75f + 0.25f * cos(k_cos * (x - nozzle_lengthening - L)));
  }

  return k_line() * (x - nozzle_lengthening) + b_line();
}

template <class GpuGridT>
__host__ __device__ float SrmShapeWithUmbrella<GpuGridT>::F_prime(float x) const
{
  if (x < l)
  {
    return 0;
  }

  if (x < L + nozzle_lengthening)
  {
    return 0;
  }

  if (x < L + nozzle_lengthening + 0.55f * l_nozzle)
  {
    return -0.25f * R0 * k_cos * sin(k_cos * (x - nozzle_lengthening - L));// R0 * (0.75f + 0.25f*cos(k*(x - L)));
  }

  return k_line();
}

template <class GpuGridT>
__host__ __device__ EBoundaryCondition SrmShapeWithUmbrella<GpuGridT>::getBoundaryCondition(float x, float y)
{
  if ((x > x_left) && (x < x_junc) && (y - y_bottom >= R0 - 1e-4) && (y - y_bottom < Rk))
  {
    return EBoundaryCondition::eMassFlowInlet;
  }

  if (std::fabs(x - x_right) < 0.1f * GpuGridT::hx)
  {
    return EBoundaryCondition::ePressureOutlet;
  }
  
  return EBoundaryCondition::eWall;
}

template <class GpuGridT>
__host__ __device__ bool SrmShapeWithUmbrella<GpuGridT>::shouldApplyScheme(unsigned i, unsigned j)
{
  return i * GpuGridT::hx <= x_junc + l_nozzle;
}

template <class GpuGridT>
__host__ __device__ float SrmShapeWithUmbrella<GpuGridT>::getRadius(unsigned i, unsigned j)
{
  return j * GpuGridT::hy - y_bottom;
}

template <class GpuGridT>
__host__ __device__ bool SrmShapeWithUmbrella<GpuGridT>::isPointOnGrain(float x, float y)
{
  bool isOnGrain = (x > x_left) && (x < x_junc) && (y - y_bottom >= R0 - 1e-4f) && (y < y_bottom + Rk);
  bool isOnCorner = (x >= x_junc) &&
    (x <= x_junc + 20 * GpuGridT::hx) &&
    (y - y_bottom >= R0) &&
    (y - y_bottom <= R0 + 20 * GpuGridT::hx) &&
    (std::powf(x - x_junc - 20 * GpuGridT::hx, 2) + std::powf(y - y_bottom - R0 - 20 * GpuGridT::hy, 2) >= 400 * GpuGridT::hx * GpuGridT::hx);

  bool shouldBeAdvanced = isOnGrain || isOnCorner;
  return shouldBeAdvanced;
}

template <class GpuGridT>
__host__ __device__ float SrmShapeWithUmbrella<GpuGridT>::operator()(unsigned i, unsigned j) const
{
  float x = i * GpuGridT::hx;
  float y = j * GpuGridT::hy;

  bool Zone1 = x <= x_left && y >= y_bottom && y <= y_bottom + R0;
  bool Zone2 = x <= x_left && y <= y_bottom;
  bool Zone3 = x >= x_left && x <= x_right && y <= y_bottom;
  bool Zone4 = x >= x_right && y <= y_bottom;
  bool Zone5 = x >= x_right && y >= y_bottom && y - y_bottom <= F(x_right - x_left);
  bool Zone6 = y - y_bottom >= k_normal_line() * (x - x_left) + b_normal_line() && y - y_bottom >= F(x_right - x_left);
  bool Zone7 = y - y_bottom <= k_normal_line() * (x - x_left) + b_normal_line() && y - y_bottom - R0 - cos(0.5f*alpha) / sin(0.5f*alpha)*(x - x_left - l - h) <= 0.0f && y - y_bottom >= F(x - x_left);
  bool Zone8 = y - y_bottom - R0 - cos(0.5f*alpha) / sin(0.5f*alpha)*(x - x_left - l - h) >= 0.0f && cos(alpha)*(y - y_bottom - R0) + sin(alpha)*(x - x_left - l - h) >= 0.0f && sin(alpha)*(y - y_bottom - R0) - cos(alpha)*(x - x_left - l - h) - H / sin(alpha) <= 0.0f;
  bool Zone9 = sin(alpha)*(y - y_bottom - R0) - cos(alpha)*(x - x_left - l - h) - H / sin(alpha) >= 0.0f && x - x_left - l - h + cos(alpha) / sin(alpha)*H >= 0.0f && y - y_bottom >= F(x - x_left);
  bool Zone10 = x - x_left - l - h + cos(alpha) / sin(alpha)*H <= 0.0f && x - x_left - l + cos(alpha) / sin(alpha)*H >= 0.0f && y - y_bottom - R0 - H >= 0.0f;
  bool Zone11 = x - x_left - l + cos(alpha) / sin(alpha)*H <= 0.0f && sin(alpha)*(y - y_bottom - R0) - cos(alpha)*(x - x_left - l) - H / sin(alpha) >= 0.0f && cos(0.5f*alpha)*(y - y_bottom - R0) + sin(0.5f*alpha)*(x - x_left - l) >= 0.0f;
  bool Zone12 = sin(alpha)*(y - y_bottom - R0) - cos(alpha)*(x - x_left - l) - H / sin(alpha) <= 0.0f && cos(alpha)*(y - y_bottom - R0) + sin(alpha)*(x - x_left - l) <= 0.0f && cos(0.5f*alpha)*(y - y_bottom - R0) + sin(0.5f*alpha)*(x - x_left - l) >= 0.0f;
  bool Zone13 = cos(0.5f*alpha)*(y - y_bottom - R0) + sin(0.5f*alpha)*(x - x_left - l) <= 0.0f && y - y_bottom >= F(x - x_left) && x - x_left >= 0.0f;
  bool Zone14 = x - x_left <= 0.0f && y - y_bottom >= R0;
  bool Zone15 = y - y_bottom <= F(x - x_left) && x - x_left <= l;
  bool Zone16 = y - y_bottom <= F(x - x_left) && x - x_left >= l + h;
  bool Zone17 = y - y_bottom <= F(x - x_left) && x - x_left <= l + h && x - x_left >= l && sin(alpha)*(y - y_bottom - R0) - cos(alpha)*(x - x_left - l - h) <= 0.0f;
  bool Zone18 = y - y_bottom >= R0 && y - y_bottom <= R0 + H && cos(alpha)*(y - y_bottom - R0) + sin(alpha)*(x - x_left - l - h) <= 0.0f && cos(alpha)*(y - y_bottom - R0) + sin(alpha)*(x - x_left - l) >= 0.0f && sin(alpha)*(y - y_bottom - R0) - cos(alpha)*(x - x_left - l) >= 0.0f;
  bool Zone19 = sin(alpha)*(y - y_bottom - R0) - cos(alpha)*(x - x_left - l - h) >= 0.0f && sin(alpha)*(y - y_bottom - R0) - cos(alpha)*(x - x_left - l) <= 0.0f;

  if (Zone1)
  {
    return x_left - x;
  }

  if (Zone2)
  {
    return sqrt((x - x_left)*(x - x_left) + (y - y_bottom)*(y - y_bottom));
  }

  if (Zone3)
  {
    return y_bottom - y;
  }

  if (Zone4)
  {
    return sqrt((x - x_right)*(x - x_right) + (y - y_bottom)*(y - y_bottom));
  }

  if (Zone5)
  {
    return x - x_right;
  }

  if (Zone6)
  {
    return sqrt((x - x_right)*(x - x_right) + (y - y_bottom - F(x_right - x_left))*(y - y_bottom - F(x_right - x_left)));
  }

  if (Zone7)
  {
    return (y - y_bottom - F(x - x_left)) / sqrt(1.0f + F_prime(x - x_left)*F_prime(x - x_left));
  }

  if (Zone8)
  {
    return cos(alpha)*(y - y_bottom - R0) + sin(alpha)*(x - x_left - l - h);
  }

  if (Zone9)
  {
    return sqrt(std::powf(x - x_left - l - h + cos(alpha) / sin(alpha) * H, 2) + std::powf(y - y_bottom - R0 - H, 2));
  }

  if (Zone10)
  {
    return y - y_bottom - R0 - H;
  }

  if (Zone11)
  {
    return sqrt(std::powf(x - x_left - l + cos(alpha) / sin(alpha) * H, 2) + std::powf(y - y_bottom - R0 - H, 2));
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
    return sqrt(std::powf(x - x_left, 2) + std::powf(y - y_bottom - R0, 2));
  }

  if (Zone15)
  {
    float val1 = absmin((y - y_bottom - F(x - x_left)) / sqrt(1.0f + F_prime(x - x_left)*F_prime(x - x_left)), y_bottom - y);

    return absmin(val1, x_left - x);
  }

  if (Zone16)
  {
    float val1 = absmin((y - y_bottom - F(x - x_left)) / sqrt(1.0f + F_prime(x - x_left)*F_prime(x - x_left)), y_bottom - y);

    return absmin(val1, x - x_right);
  }

  if (Zone17)
  {
    float val1 = -sqrt(std::powf(x - x_left - l, 2) + std::powf(y - y_bottom - R0, 2));
    float val2 = -sqrt(std::powf(x - x_left - l - h, 2) + std::powf(y - y_bottom - R0, 2));
    float val3 = absmin(val1, val2);


    return absmin(val3, y_bottom - y);
  }

  if (Zone18)
  {
    float val1 = cos(alpha)*(y - y_bottom - R0) + sin(alpha)*(x - x_left - l - h);
    float val2 = -(cos(alpha)*(y - y_bottom - R0) + sin(alpha)*(x - x_left - l));
    float val3 = absmin(val1, val2);


    return absmin(val3, y - y_bottom - R0 - H);

  }

  if (Zone19)
  {
    float val1 = -sqrt(std::powf(x - x_left - l, 2) + std::powf(y - y_bottom - R0, 2));
    float val2 = (cos(alpha)*(y - y_bottom - R0) + sin(alpha)*(x - x_left - l - h));
    float val3 = absmin(val1, val2);

    return absmin(val3, y_bottom - y);

  }

  return 0;
}

} // namespace kae
