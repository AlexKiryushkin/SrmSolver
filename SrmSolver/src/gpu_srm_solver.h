#pragma once

#include "boundary_condition.h"
#include "cuda_float_types.h"
#include "empty_callback.h"
#include "gpu_level_set_solver.h"
#include "gpu_matrix.h"

namespace kae {

template <class ShapeT, class GasStateT, class PhysicalPropertiesT>
class GpuSrmSolver
{
public:

  using GasStateType             = GasStateT;
  using PhysicalPropertiesType   = PhysicalPropertiesT;
  using ElemType                 = typename GasStateType::ElemType;

  GpuSrmSolver(GpuGridT<ElemType> grid, ShapeT    shape,
               GasStateT initialState, 
               unsigned  iterationCount = 0U, 
               ElemType  courant = static_cast<ElemType>(0.8));

  template <class CallbackT = detail::EmptyCallback>
  void quasiStationaryDynamicIntegrate(unsigned                 iterationCount, 
                                       ElemType                 deltaT,
                                       ETimeDiscretizationOrder timeOrder, 
                                       CallbackT &&             callback = CallbackT{});

  template <class CallbackT = detail::EmptyCallback>
  void dynamicIntegrate(unsigned                 iterationCount, 
                        ElemType                 deltaT, 
                        ETimeDiscretizationOrder timeOrder, 
                        CallbackT &&             callback = CallbackT{});

  template <class CallbackT = detail::EmptyCallback>
  ElemType staticIntegrate(unsigned                 iterationCount,
                           ETimeDiscretizationOrder timeOrder, 
                           CallbackT &&             callback = CallbackT{});

  template <class CallbackT = detail::EmptyCallback>
  ElemType staticIntegrate(ElemType                 deltaT, 
                           ETimeDiscretizationOrder timeOrder, 
                           CallbackT &&             callback = CallbackT{});

  const GpuMatrix<GasStateType> & currState() const { return m_currState; }
  const GpuMatrix<ElemType>     & currPhi()   const { return m_levelSetSolver.currState(); }

private:

  ElemType staticIntegrateStep(ETimeDiscretizationOrder timeOrder, ElemType dt, CudaFloat2T<ElemType> lambdas);
  ElemType integrateInTime(ElemType deltaT);
  CudaFloat4T<ElemType> getMaxEquationDerivatives() const;
  void findClosestIndices();
  void fillCalculateBlockMatrix();
  void writeIfNotValid() const;

private:

  constexpr static unsigned order{ 2U };
  using IndexMatrixT = kae::Matrix<unsigned, order, order>;

  GpuGridT<ElemType> m_grid;
  GpuMatrix<EBoundaryCondition>       m_boundaryConditions;
  GpuMatrix<CudaFloat2T<ElemType>>    m_normals;
  GpuMatrix<CudaFloat2T<ElemType>>    m_surfacePoints;
  GpuMatrix<IndexMatrixT>             m_indexMatrices;
  GpuMatrix<GasStateType>             m_currState;
  GpuMatrix<GasStateType>             m_prevState;
  GpuMatrix<GasStateType>             m_firstState;
  GpuMatrix<GasStateType>             m_secondState;
  GpuLevelSetSolver<ElemType, ShapeT>           m_levelSetSolver;

  thrust::device_vector<thrust::pair<unsigned, unsigned>> m_closestIndicesMap;
  std::vector<int8_t> m_calculateBlocks;

  ElemType m_courant{ static_cast<ElemType>(0.8) };
};

} // namespace kae

#include "gpu_srm_solver_def.h"
