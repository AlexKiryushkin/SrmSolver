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

  template <class PropellantPropertiesT, class GasStateT>
  ElemType integrateInTime(const GpuMatrix<GpuGridT, GasStateT> & gasValues,
                           const GpuMatrix<GpuGridT, unsigned>  & closestIndices,
                           unsigned                               iterationCount,
                           ETimeDiscretizationOrder               timeOrder = ETimeDiscretizationOrder::eThree);
  template <class PropellantPropertiesT, class GasStateT>
  ElemType integrateInTime(const GpuMatrix<GpuGridT, GasStateT> & gasValues,
                           const GpuMatrix<GpuGridT, unsigned>  & closestIndices,
                           ElemType                               deltaT,
                           ETimeDiscretizationOrder               timeOrder = ETimeDiscretizationOrder::eThree);

  void reinitialize(unsigned iterationCount, ETimeDiscretizationOrder timeOrder = ETimeDiscretizationOrder::eThree);

  const GpuMatrix<GpuGridT, ElemType> & currState() const { return m_currState; }

private:

  template <class PropellantPropertiesT, class GasStateT>
  ElemType integrateInTimeStep(const GpuMatrix<GpuGridT, GasStateT> & gasValues,
                               const GpuMatrix<GpuGridT, unsigned>  & closestIndices,
                               ETimeDiscretizationOrder               timeOrder);

  template <class PropellantPropertiesT, class GasStateT>
  ElemType integrateInTimeStep(const GpuMatrix<GpuGridT, GasStateT> & gasValues,
                               const GpuMatrix<GpuGridT, unsigned>  & closestIndices,
                               ETimeDiscretizationOrder               timeOrder,
                               ElemType dt);

  void reinitializeStep(ETimeDiscretizationOrder timeOrder);

  template <class PropellantPropertiesT, class GasStateT>
  ElemType getMaxBurningRate(const thrust::device_vector<GasStateT> & gasValues);

private:
  GpuMatrix<GpuGridT, ElemType> m_currState;
  GpuMatrix<GpuGridT, ElemType> m_prevState;
  GpuMatrix<GpuGridT, ElemType> m_firstState;
  GpuMatrix<GpuGridT, ElemType> m_secondState;
};

} // namespace kae

#include "gpu_level_set_solver_def.h"
