
#include "problem.h"

#include <Eigen/Eigen>

#include "mass_flow.h"

namespace kae {

Problem::Problem(IGridPtr             pGrid, 
                 IBoundaryPtr         pBoundary, 
                 GasStateFunctionPtr  pGasStateFunction,
                 IGhostValueSetterPtr pGhostValueSetter, 
                 ElemT                integrationTime)
  : m_pGrid            { std::move(pGrid) },
    m_pBoundary        { std::move(pBoundary) },
    m_pGasStateFunction{ std::move(pGasStateFunction) },
    m_pGhostValueSetter{ std::move(pGhostValueSetter) },
    m_integrationTime  { integrationTime }
{
}

void Problem::updateBoundaries(const std::vector<GasState>& gasValues, ElemT t, ElemT dt, unsigned rkStep)
{
    m_pBoundary->updateBoundaries(gasValues, t, dt, rkStep);
}

void Problem::setGhostValues(std::vector<GasState> & gasStates, ElemT t, ElemT dt, unsigned rkStep)
{
  m_pGhostValueSetter->setGhostValues(gasStates, *this, t, dt, rkStep);
}

std::vector<GasState> Problem::getGasState(ElemT t) const
{
  std::vector<GasState> gasStates;
  for (std::size_t idx{}; idx < m_pGrid->getNPoints(); ++idx)
  {
    const auto x = getX(idx);
    gasStates.push_back({ m_pGasStateFunction->getRho(x, t), m_pGasStateFunction->getU(x, t), m_pGasStateFunction->getP(x, t) });
  }
  return gasStates;
}

ProblemPtr ProblemFactory::makeEulerSmoothProblem(std::size_t nPoints)
{
  constexpr auto pi = static_cast<ElemT>(M_PI);
  auto pGrid = GridFactory::makeSimpleGrid(nPoints, -pi, pi);
  auto pGasStateFunction = GasStateFunctionFactory::makeSmoothSineGasStateFunction(
    static_cast<ElemT>(1.0), static_cast<ElemT>(0.2), static_cast<ElemT>(1.0));
  auto pBoundary = BoundaryFactory::makeStationaryBoundary(
    pGrid->getXLeft()  + static_cast<ElemT>(3.5) * pGrid->getH(),
    pGrid->getXRight() - static_cast<ElemT>(3.5) * pGrid->getH());
  auto pGhostValueSetter = GhostValueSetterFactory::makeExactGhostValueSetter();
  return std::make_unique<Problem>(std::move(pGrid), 
                                   std::move(pBoundary), 
                                   std::move(pGasStateFunction),
                                   std::move(pGhostValueSetter),
                                   static_cast<ElemT>(2.0));
}

ProblemPtr ProblemFactory::makeStationaryMassFlowProblem(std::size_t nPoints)
{
  constexpr auto pi = static_cast<ElemT>(M_PI);
  auto pGrid = GridFactory::makeSimpleGrid(nPoints, -pi, pi);
  auto pGasStateFunction = GasStateFunctionFactory::makeSmoothSineGasStateFunction(
    static_cast<ElemT>(1.0), static_cast<ElemT>(0.2), static_cast<ElemT>(1.0));
  auto pBoundary = BoundaryFactory::makeStationaryBoundary(
    pGrid->getXLeft()  + static_cast<ElemT>(3.5) * pGrid->getH(),
    pGrid->getXRight() - static_cast<ElemT>(3.5) * pGrid->getH());
  auto pGhostValueSetter = GhostValueSetterFactory::makeMassFlowGhostValueSetter();
  return std::make_unique<Problem>(std::move(pGrid),
    std::move(pBoundary),
    std::move(pGasStateFunction),
    std::move(pGhostValueSetter),
    static_cast<ElemT>(2.0));
}

ProblemPtr ProblemFactory::makeMovingMassFlowProblem(std::size_t nPoints)
{
  constexpr auto pi = static_cast<ElemT>(M_PI);
  auto pGrid = GridFactory::makeSimpleGrid(nPoints, -pi, pi);
  auto pGasStateFunction = GasStateFunctionFactory::makeSmoothSineGasStateFunction(
    static_cast<ElemT>(1.0), static_cast<ElemT>(0.2), static_cast<ElemT>(1.0));
  auto pBoundary = BoundaryFactory::makeStationaryBoundary(
    pGrid->getXLeft()  + static_cast<ElemT>(3.5) * pGrid->getH(),
    pGrid->getXRight() - static_cast<ElemT>(3.5) * pGrid->getH());
  auto pGhostValueSetter = GhostValueSetterFactory::makeMassFlowGhostValueSetter(static_cast<ElemT>(0.01));
  return std::make_unique<Problem>(std::move(pGrid),
    std::move(pBoundary),
    std::move(pGasStateFunction),
    std::move(pGhostValueSetter),
    static_cast<ElemT>(2.0));
}

} // namespace kae
