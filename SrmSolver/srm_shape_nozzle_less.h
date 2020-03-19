#pragma once

#include <thrust/host_vector.h>

#include <cuda_runtime_api.h>

#include "boundary_condition.h"

namespace kae {

template <class GpuGridT>
class SrmShapeNozzleLess
{
public:

  SrmShapeNozzleLess();

  __host__ __device__ static bool shouldApplyScheme(unsigned i, unsigned j);

  __host__ __device__ static bool isPointOnGrain(float x, float y);

  __host__ __device__ static EBoundaryCondition getBoundaryCondition(float x, float y);

  __host__ __device__ static float getRadius(unsigned i, unsigned j);

  const thrust::host_vector<float> & values() const;

private:

  constexpr static unsigned offsetPoints = 20;

  constexpr static float xLeft = (offsetPoints + 0.5f) * GpuGridT::hx;
  constexpr static float delta = 0.01f;
  constexpr static float xStartPropellant = xLeft + delta;
  constexpr static float xRight = xLeft + 1.274f;
  constexpr static float yBottom = (offsetPoints + 0.5f) * GpuGridT::hy;
  constexpr static float Rk = 0.1f;
  constexpr static float rkr = 0.0245f;
private:

  thrust::host_vector<float> m_distances;
};

} // namespace kae

#include "srm_shape_nozzle_less_def.h"
