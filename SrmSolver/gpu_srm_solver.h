#pragma once

#include "boundary_condition.h"
#include "gpu_level_set_solver.h"
#include "gpu_matrix.h"

namespace kae {

template <class GpuGridT, class ShapeT, class GasStateT, class PropellantPropertiesT>
class GpuSrmSolver
{
public:

  using GasStateType             = GasStateT;
  using PropellantPropertiesType = PropellantPropertiesT;

  GpuSrmSolver(ShapeT shape, GasStateT initialState, unsigned iterationCount = 0U, float courant = 0.8f);

  template <class CallbackT>
  void dynamicIntegrate(unsigned iterationCount, ETimeDiscretizationOrder timeOrder, CallbackT callback);

  float staticIntegrate(unsigned iterationCount, ETimeDiscretizationOrder timeOrder);
  float staticIntegrate(float deltaT, ETimeDiscretizationOrder timeOrder);

  const GpuMatrix<GpuGridT, GasStateType> & currState() const { return m_currState; }
  const GpuMatrix<GpuGridT, float>        & currPhi()   const { return m_levelSetSolver.currState(); }

private:

  float staticIntegrateStep(ETimeDiscretizationOrder timeOrder);
  float staticIntegrateStep(ETimeDiscretizationOrder timeOrder, float dt, float2 lambdas);

private:

  GpuMatrix<GpuGridT, EBoundaryCondition> m_boundaryConditions;
  GpuMatrix<GpuGridT, unsigned>           m_closestIndices;
  GpuMatrix<GpuGridT, float2>             m_normals;
  GpuMatrix<GpuGridT, GasStateType>       m_currState;
  GpuMatrix<GpuGridT, GasStateType>       m_prevState;
  GpuMatrix<GpuGridT, GasStateType>       m_firstState;
  GpuMatrix<GpuGridT, GasStateType>       m_secondState;
  GpuLevelSetSolver<GpuGridT, ShapeT>     m_levelSetSolver;

  float m_courant{ 0.8f };
};

} // namespace kae

#include "gpu_srm_solver_def.h"
