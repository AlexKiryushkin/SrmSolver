#pragma once

#include <cmath>

#include <cuda_runtime_api.h>

#include "boundary_condition.h"

namespace kae {

template <class GpuGridT>
class SrmShapeWithUmbrella
{
public:

  __host__ __device__ float operator()(unsigned i, unsigned j) const;

  __host__ __device__ static bool shouldApplyScheme(unsigned i, unsigned j);

  __host__ __device__ static bool isPointOnGrain(float x, float y);

  __host__ __device__ static EBoundaryCondition getBoundaryCondition(float x, float y);

  __host__ __device__ static float getRadius(unsigned i, unsigned j);

private:

  __host__ __device__ float F_prime(float x) const;
  __host__ __device__ float F(float x) const;

private:

  constexpr static float L = 2.4f;
  constexpr static float R0 = 0.2f;
  constexpr static float Rk = 0.9f;
  constexpr static float H = 0.55f;
  constexpr static float rkr = 0.1f;
  constexpr static float l = 0.5f;
  constexpr static float h = 0.08f;
  constexpr static float l_nozzle = 0.4f;
  constexpr static float alpha = 0.5f * static_cast<float>(M_PI);

  constexpr static float k_cos = 2.0f * static_cast<float>(M_PI) / l_nozzle;
  __host__ __device__ static float k_line() { return -0.25f * R0 * k_cos * std::sin(0.55f * k_cos * l_nozzle); }
  __host__ __device__ static float b_line() { return R0 * (0.75f + 0.25f * std::cos(0.55f*l_nozzle*k_cos)) - k_line() * (L + 0.55f * l_nozzle); }

  constexpr static unsigned offsetPoints = 20;

  constexpr static float x_left = (offsetPoints + 0.5f) * GpuGridT::hx;
  constexpr static float x_junc = x_left + L;
  constexpr static float x_right = x_left + L + 1.5f * l_nozzle + 0.5f * GpuGridT::hx;
  constexpr static float y_bottom = (offsetPoints + 0.5f) * GpuGridT::hy;

  __host__ __device__ static float k_normal_line() { return -1.0f / k_line(); }
  __host__ __device__ static float b_normal_line() { return b_line() + (x_right - x_left)*(k_line() + 1.0f / k_line()); }

  constexpr static float nozzle_lengthening = 0.1f;
};

} // namespace kae

#include "srm_shape_with_umbrella_def.h"
