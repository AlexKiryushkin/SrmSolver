#pragma once

#include "discretization_order.h"
#include "gpu_matrix.h"

namespace kae {

template <class GpuGridT, class ShapeT>
class GpuLevelSetSolver
{
public:

  explicit GpuLevelSetSolver(ShapeT shape,
                             unsigned iterationCount = 0,
                             ETimeDiscretizationOrder timeOrder = ETimeDiscretizationOrder::eThree);

  template <class PropellantPropertiesT, class GasStateT>
  float integrateInTime(const GpuMatrix<GpuGridT, GasStateT> & gasValues,
                        const GpuMatrix<GpuGridT, unsigned>  & closestIndices,
                        unsigned                               iterationCount,
                        ETimeDiscretizationOrder               timeOrder = ETimeDiscretizationOrder::eThree);
  template <class PropellantPropertiesT, class GasStateT>
  float integrateInTime(const GpuMatrix<GpuGridT, GasStateT> & gasValues,
                        const GpuMatrix<GpuGridT, unsigned>  & closestIndices,
                        float                                  deltaT,
                        ETimeDiscretizationOrder               timeOrder = ETimeDiscretizationOrder::eThree);

  void reinitialize(unsigned iterationCount, ETimeDiscretizationOrder timeOrder = ETimeDiscretizationOrder::eThree);

  const GpuMatrix<GpuGridT, float> & currState() const { return m_currState; }

private:

  template <class PropellantPropertiesT, class GasStateT>
  float integrateInTimeStep(const GpuMatrix<GpuGridT, GasStateT> & gasValues,
                            const GpuMatrix<GpuGridT, unsigned>  & closestIndices,
                            ETimeDiscretizationOrder               timeOrder);

  template <class PropellantPropertiesT, class GasStateT>
  float integrateInTimeStep(const GpuMatrix<GpuGridT, GasStateT> & gasValues,
    const GpuMatrix<GpuGridT, unsigned>  & closestIndices,
    ETimeDiscretizationOrder               timeOrder,
    float dt);

  void reinitializeStep(ETimeDiscretizationOrder timeOrder);

  template <class PropellantPropertiesT, class GasStateT>
  float getMaxBurningRate(const thrust::device_vector<GasStateT> & gasValues);

private:
  GpuMatrix<GpuGridT, float> m_currState;
  GpuMatrix<GpuGridT, float> m_prevState;
  GpuMatrix<GpuGridT, float> m_firstState;
  GpuMatrix<GpuGridT, float> m_secondState;
};

} // namespace kae

#include "gpu_level_set_solver_def.h"
