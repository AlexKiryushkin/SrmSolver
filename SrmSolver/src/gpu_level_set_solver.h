#pragma once

#include "discretization_order.h"
#include "gpu_grid.h"
#include "gpu_matrix.h"
#include "shape/shape.h"

namespace kae {

template <class ElemT>
class GpuLevelSetSolver
{
public:

    using ElemType = typename ElemT;

    explicit GpuLevelSetSolver(const GpuGridT<ElemType>& grid, thrust::host_vector<ElemType> signedDistances, Shape<ElemType> shape,
        unsigned iterationCount = 0,
        ETimeDiscretizationOrder timeOrder = ETimeDiscretizationOrder::eThree);

    ElemType integrateInTime(const GpuMatrix<ElemType>& velocities,
        unsigned                              iterationCount,
        ETimeDiscretizationOrder              timeOrder = ETimeDiscretizationOrder::eThree);
    void integrateInTime(const GpuMatrix<ElemType>& velocities,
        ElemType                              deltaT,
        ETimeDiscretizationOrder              timeOrder = ETimeDiscretizationOrder::eThree);

    void reinitialize(unsigned iterationCount, ETimeDiscretizationOrder timeOrder = ETimeDiscretizationOrder::eThree);

    const GpuMatrix<ElemType>& currState() const { return m_currState; }

private:

    ElemType integrateInTimeStep(const GpuMatrix<ElemType>& velocities,
        ETimeDiscretizationOrder              timeOrder);
    ElemType integrateInTimeStep(const GpuMatrix<ElemType>& velocities,
        ETimeDiscretizationOrder              timeOrder,
        ElemType dt);

    void reinitializeStep(ETimeDiscretizationOrder timeOrder);

    ElemType getMaxVelocity(const thrust::device_vector<ElemType>& velocities);

private:
    GpuGridT<ElemType> m_grid;
    GpuMatrix<ElemType> m_currState;
    GpuMatrix<ElemType> m_prevState;
    GpuMatrix<ElemType> m_firstState;
    GpuMatrix<ElemType> m_secondState;
    Shape<ElemType> m_shape;
};

} // namespace kae

#include "gpu_level_set_solver_def.h"
