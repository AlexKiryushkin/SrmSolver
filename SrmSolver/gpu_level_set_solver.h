#pragma once

#include "discretization_order.h"
#include "gpu_matrix.h"

namespace kae {

template <class GpuGridT, class ShapeT>
class GpuLevelSetSolver
{
public:

  using ElemType = typename GpuGridT::ElemType;

  explicit GpuLevelSetSolver(ShapeT shape,
                             unsigned iterationCount = 0,
                             ETimeDiscretizationOrder timeOrder = ETimeDiscretizationOrder::eThree);

  ElemType integrateInTime(const GpuMatrix<GpuGridT, ElemType> & velocities,
                           unsigned                              iterationCount,
                           ETimeDiscretizationOrder              timeOrder = ETimeDiscretizationOrder::eThree);
  ElemType integrateInTime(const GpuMatrix<GpuGridT, ElemType> & velocities,
                           ElemType                              deltaT,
                           ETimeDiscretizationOrder              timeOrder = ETimeDiscretizationOrder::eThree);

  void reinitialize(unsigned iterationCount, ETimeDiscretizationOrder timeOrder = ETimeDiscretizationOrder::eThree);

  const GpuMatrix<GpuGridT, ElemType> & currState() const { return m_currState; }

private:

  ElemType integrateInTimeStep(const GpuMatrix<GpuGridT, ElemType> & velocities,
                               ETimeDiscretizationOrder              timeOrder);
  ElemType integrateInTimeStep(const GpuMatrix<GpuGridT, ElemType> & velocities,
                               ETimeDiscretizationOrder              timeOrder,
                               ElemType dt);

  void reinitializeStep(ETimeDiscretizationOrder timeOrder);

  ElemType getMaxVelocity(const thrust::device_vector<ElemType> & velocities);

private:
  GpuMatrix<GpuGridT, ElemType> m_currState;
  GpuMatrix<GpuGridT, ElemType> m_prevState;
  GpuMatrix<GpuGridT, ElemType> m_firstState;
  GpuMatrix<GpuGridT, ElemType> m_secondState;
};

} // namespace kae

#include "gpu_level_set_solver_def.h"
