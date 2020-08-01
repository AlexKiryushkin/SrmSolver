#pragma once

#include "boundary_condition.h"
#include "cuda_float_types.h"
#include "empty_callback.h"
#include "gpu_level_set_solver.h"
#include "gpu_matrix.h"

namespace kae {

template <class GpuGridT, class ShapeT, class GasStateT, class PhysicalPropertiesT>
class GpuSrmSolver
{
public:

  using GasStateType             = GasStateT;
  using PhysicalPropertiesType   = PhysicalPropertiesT;
  using ElemType                 = typename GasStateType::ElemType;

  static_assert(std::is_same<typename GasStateType::ElemType, typename GpuGridT::ElemType>::value, 
                "Error! Precisions differ.");

  GpuSrmSolver(ShapeT    shape, 
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

  const GpuMatrix<GpuGridT, GasStateType> & currState() const { return m_currState; }
  const GpuMatrix<GpuGridT, ElemType>     & currPhi()   const { return m_levelSetSolver.currState(); }

private:

  ElemType staticIntegrateStep(ETimeDiscretizationOrder timeOrder, ElemType dt, CudaFloat2T<ElemType> lambdas);
  ElemType integrateInTime(ElemType deltaT);
  CudaFloat4T<ElemType> getMaxEquationDerivatives() const;
  void findClosestIndices();
  void fillCalculateBlockMatrix();
  void writeIfNotValid() const;

private:

  constexpr static unsigned order{ 3U };
  using IndexMatrixT = Eigen::Matrix<unsigned, order, order>;

  GpuMatrix<GpuGridT, EBoundaryCondition>       m_boundaryConditions;
  GpuMatrix<GpuGridT, CudaFloat2T<ElemType>>    m_normals;
  GpuMatrix<GpuGridT, CudaFloat2T<ElemType>>    m_surfacePoints;
  GpuMatrix<GpuGridT, IndexMatrixT>             m_indexMatrices;
  GpuMatrix<GpuGridT, GasStateType>             m_currState;
  GpuMatrix<GpuGridT, GasStateType>             m_prevState;
  GpuMatrix<GpuGridT, GasStateType>             m_firstState;
  GpuMatrix<GpuGridT, GasStateType>             m_secondState;
  GpuLevelSetSolver<GpuGridT, ShapeT>           m_levelSetSolver;

  thrust::device_vector<thrust::pair<unsigned, unsigned>> m_closestIndicesMap;
  std::vector<int8_t> m_calculateBlocks;

  ElemType m_courant{ static_cast<ElemType>(0.8) };
};

} // namespace kae

#include "gpu_srm_solver_def.h"
