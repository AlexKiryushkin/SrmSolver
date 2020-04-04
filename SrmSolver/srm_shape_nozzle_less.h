#pragma once

#include "cuda_includes.h"

#include "boundary_condition.h"

namespace kae {

template <class GpuGridT>
class SrmShapeNozzleLess
{
public:

  using ElemType = typename GpuGridT::ElemType;

  SrmShapeNozzleLess();

  __host__ __device__ static bool shouldApplyScheme(unsigned i, unsigned j);

  __host__ __device__ static bool isPointOnGrain(ElemType x, ElemType y);

  __host__ __device__ static EBoundaryCondition getBoundaryCondition(ElemType x, ElemType y);

  __host__ __device__ static ElemType getRadius(unsigned i, unsigned j);

  const thrust::host_vector<ElemType> & values() const;

private:

  constexpr static unsigned offsetPoints = 20;

  constexpr static ElemType xLeft            = (offsetPoints + static_cast<ElemType>(0.5)) * GpuGridT::hx;
  constexpr static ElemType delta            = static_cast<ElemType>(0.01);
  constexpr static ElemType xStartPropellant = xLeft + delta;
  constexpr static ElemType xRight           = xLeft + static_cast<ElemType>(1.274);
  constexpr static ElemType yBottom          = (offsetPoints + static_cast<ElemType>(0.5)) * GpuGridT::hy;
  constexpr static ElemType Rk               = static_cast<ElemType>(0.1);
  constexpr static ElemType rkr              = static_cast<ElemType>(0.0245);
private:

  thrust::host_vector<ElemType> m_distances;
};

} // namespace kae

#include "srm_shape_nozzle_less_def.h"
