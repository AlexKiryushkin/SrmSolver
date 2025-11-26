#pragma once

#include "cuda_includes.h"

#include "boundary_condition.h"
#include "cuda_float_types.h"
#include "gas_state.h"
#include "get_coordinates_matrix.h"
#include "get_extrapolated_ghost_value.h"
#include "get_polynomial.h"
#include "matrix/matrix_operations.h"

template <class T>
using DevicePtr = thrust::device_ptr<T>;

template <class ElemT, unsigned order>
using LhsMatrix = kae::Matrix<ElemT, order* order, order* (order + 1U) / 2U>;

template <class ElemT, unsigned order>
using RhsMatrix = kae::Matrix<ElemT, order* order, 4U>;

template <class ElemT, unsigned order>
using SquareMatrix = kae::Matrix<ElemT, order, order>;

namespace kae {

namespace detail {

template <class GpuGridT, class ShapeT, class GasStateT, class PhysicalPropertiesT, unsigned order, unsigned smSizeX,
          class InputMatrixT, class ElemT = typename GasStateT::ElemType>
__global__ void setGhostValues(GasStateT *                              pGasValues,
                               const thrust::pair<unsigned, unsigned> * pClosestIndicesMap,
                               const EBoundaryCondition *               pBoundaryConditions,
                               CudaFloat2T<ElemT> *                     pNormals,
                               CudaFloat2T<ElemT> *                     pSurfacePoints,
                               InputMatrixT *                           pIndexMatrix,
                               unsigned                                 nClosestIndexElems)
{
  const auto i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i >= nClosestIndexElems)
  {
    return;
  }

  const auto indexMap            = pClosestIndicesMap[i];
  const auto ghostIdx            = indexMap.first;
  const auto closestIdx          = indexMap.second;
  const auto surfacePoint        = pSurfacePoints[ghostIdx];
  const auto normal              = pNormals[ghostIdx];

  const auto rotatedClosestState = Rotate::get(pGasValues[closestIdx], normal.x, normal.y);
  const auto boundaryCondition   = pBoundaryConditions[ghostIdx];
  const auto closestSonic        = SonicSpeed::get(rotatedClosestState);

  const auto indexMatrix = pIndexMatrix[ghostIdx];
  const auto x = getWenoPolynomial<GpuGridT>(surfacePoint, normal, pGasValues, indexMatrix);

  const auto ghostI = ghostIdx % GpuGridT::nx;
  const auto ghostJ = ghostIdx / GpuGridT::nx;
  const auto dn = std::hypot(ghostI * GpuGridT::hx - surfacePoint.x, ghostJ * GpuGridT::hy - surfacePoint.y);

  if ((boundaryCondition == EBoundaryCondition::ePressureOutlet) && (closestSonic < rotatedClosestState.ux))
  {
    const auto rho  = x(0, 0) + x(1, 0) * dn;
    const auto un   = x(0, 1) + x(1, 1) * dn;
    const auto utau = x(0, 2) + x(1, 2) * dn;
    const auto p    = x(0, 3) + x(1, 3) * dn;
    pGasValues[ghostIdx] = ReverseRotate::get(GasStateT{ rho, un, utau, p }, normal.x, normal.y);
    return;
  }
  else if (boundaryCondition == EBoundaryCondition::eWall)
  {
    const ElemT rho_0  = x(0, 0) + rotatedClosestState.rho / closestSonic * x(0, 1);
    const ElemT un_0   = 0;
    const ElemT utau_0 = x(0, 2);
    const ElemT p_0    = x(0, 3) + rotatedClosestState.rho * closestSonic * x(0, 1);

    const ElemT p_1    = -rho_0 * utau_0 * x(2, 1);
    const ElemT rho_1  = x(1, 0) - 1 / closestSonic / closestSonic * (x(1, 3) - p_1);
    const ElemT un_1   = x(1, 1) + 1 / rotatedClosestState.rho / closestSonic * (x(1, 3) - p_1);
    const ElemT utau_1 = x(1, 2);

    const auto rho  = rho_0  + rho_1  * dn;
    const auto un   = un_0   + un_1   * dn;
    const auto utau = utau_0 + utau_1 * dn;
    const auto p    = p_0    + p_1    * dn;
    
    const auto wallGasState = GasStateT{ rho, un, utau, p };
    if (kae::IsValid::get(wallGasState))
    {
      pGasValues[ghostIdx] = ReverseRotate::get(wallGasState, normal.x, normal.y);
      return;
    }
  }
  else if (boundaryCondition == EBoundaryCondition::eMassFlowInlet)
  {
    constexpr auto kappa = PhysicalPropertiesT::kappa;
    constexpr auto nu    = PhysicalPropertiesT::nu;
    constexpr auto zero  = static_cast<ElemT>(0);
    constexpr auto one   = static_cast<ElemT>(1);

    const auto extrapolatedGasState = GasStateT{ x(0, 0), x(0, 1), x(0, 2), x(0, 3) };
    const auto mfs   = getFirstOrderMassFlowExtrapolatedGhostValue<PhysicalPropertiesT>(extrapolatedGasState, rotatedClosestState);
    const auto unSqr = sqr(mfs.ux);
    const auto cSqr  = kappa * mfs.p / mfs.rho;

    kae::Matrix<ElemT, 4U, 4U> lhs{
      { unSqr,                                       (2 - kappa * nu) * mfs.rho * mfs.ux, zero,   1 - nu * mfs.rho * unSqr / mfs.p },
      { - 1 / (kappa - 1) * cSqr * mfs.ux / mfs.rho, cSqr + unSqr,                        zero,  (2 * kappa - 1) / (kappa - 1) * mfs.ux / mfs.rho },
      { zero,                                        zero,                                mfs.ux, zero },
      { zero,                                        one,                                 zero,   1 / rotatedClosestState.rho / closestSonic}
    };
    const auto r = ShapeT::getRadius(surfacePoint.x, surfacePoint.y);
    kae::Matrix<ElemT, 4U, 1U> rhs{
     (kappa * nu - 1) * mfs.rho * mfs.ux * (x(2, 2) + mfs.ux * normal.y / r),
      -cSqr * (x(2, 2) + mfs.ux * normal.y / r),
      -1 / mfs.rho * x(2, 3),
      x(1, 1) + 1 / rotatedClosestState.rho / closestSonic * x(1, 3)};

    const auto lhsTr = kae::transpose(lhs);
    const auto A = lhsTr * lhs;
    const auto b = lhsTr * rhs;
    const auto sol = kae::detail::choleskySolve(A, b);
    const auto massFlowGasState = GasStateT{ mfs.rho, mfs.ux, mfs.uy, mfs.p };
    if (kae::IsValid::get(massFlowGasState))
    {
      pGasValues[ghostIdx] = ReverseRotate::get(massFlowGasState, normal.x, normal.y);
      return;
    }
  }
  
  const auto extrapolatedState = getFirstOrderExtrapolatedGhostValue<PhysicalPropertiesT>(rotatedClosestState, 
    rotatedClosestState,
    boundaryCondition);
  pGasValues[ghostIdx] = ReverseRotate::get(extrapolatedState, normal.x, normal.y);
}

template <class GpuGridT, class ShapeT, class GasStateT, class PhysicalPropertiesT, unsigned order,
          class InputMatrixT, class ElemT = typename GasStateT::ElemType>
void setGhostValuesWrapper(DevicePtr<GasStateT>                              pGasValues,
                           DevicePtr<const thrust::pair<unsigned, unsigned>> pClosestIndices,
                           DevicePtr<const EBoundaryCondition>               pBoundaryConditions,
                           DevicePtr<CudaFloat2T<ElemT>>                     pNormals,
                           DevicePtr<CudaFloat2T<ElemT>>                     pSurfacePoints,
                           DevicePtr<InputMatrixT>                           pIndexMatrix,
                           unsigned                                          nClosestIndexElems)
{
  constexpr unsigned blockSize = 64U;
  const unsigned gridSize = (nClosestIndexElems + blockSize - 1U) / blockSize;
  setGhostValues<GpuGridT, ShapeT, GasStateT, PhysicalPropertiesT, order, blockSize><<<gridSize, blockSize>>>
  (pGasValues.get(), pClosestIndices.get(), pBoundaryConditions.get(), pNormals.get(), 
    pSurfacePoints.get(), pIndexMatrix.get(), nClosestIndexElems);
}

} // namespace detail

} // namespace kae

