#pragma once

#include <cmath>
#include <vector>

#include "boundary.h"
#include "function.h"
#include "gas_state.h"
#include "ghost_value_setter.h"
#include "grid.h"

namespace kae {

class Problem
{
public:

  Problem(IGridPtr             pGrid, 
          IBoundaryPtr         pBoundary, 
          GasStateFunctionPtr  pGasStateFunction, 
          IGhostValueSetterPtr pGhostValueSetter,
          ElemT                integrationTime);

  std::size_t              getStartIdx()                                         const { return m_pBoundary->getStartIdx(*m_pGrid); }
  std::size_t              getEndIdx()                                           const { return m_pBoundary->getEndIdx(*m_pGrid); }
  ElemT                    getXBoundaryLeft(ElemT t, ElemT dt, unsigned rkStep)  const { return m_pBoundary->getXBoundaryLeft(t, dt, rkStep); }
  ElemT                    getXBoundaryRight(ElemT t, ElemT dt, unsigned rkStep) const { return m_pBoundary->getXBoundaryRight(t, dt, rkStep); }
  ElemT                    getXBoundaryLeftAnalytical(ElemT t)                   const { return m_pBoundary->getXBoundaryLeftAnalytical(t); }
  ElemT                    getXBoundaryRightAnalytical(ElemT t)                  const { return m_pBoundary->getXBoundaryRightAnalytical(t); }
  ElemT                    getH()                                                const { return m_pGrid->getH(); }
  ElemT                    getX(std::size_t i)                                   const { return m_pGrid->getX(i); }
  ElemT                    getXLeft()                                            const { return m_pGrid->getXLeft(); }
  ElemT                    getXRight()                                           const { return m_pGrid->getXRight(); }
  ElemT                    getIntegrationTime()                                  const { return m_integrationTime; }
  std::vector<GasState>    getInitialState()                                     const { return getGasState(0.0); }
  std::vector<GasState>    getIAnalyticalSolution()                              const { return getGasState(getIntegrationTime()); }

  void updateBoundaries(const std::vector<GasState>& gasValues, ElemT t, ElemT dt, unsigned rkStep);
  void setGhostValues(std::vector<GasState>& gasStates, ElemT t, ElemT dt, unsigned rkStep);

  const GasStateFunction & getGasStateFunction()    const { return *m_pGasStateFunction; }

private:

  std::vector<GasState> getGasState(ElemT t) const;

  IGridPtr             m_pGrid;
  IBoundaryPtr         m_pBoundary;
  GasStateFunctionPtr  m_pGasStateFunction;
  IGhostValueSetterPtr m_pGhostValueSetter;
  ElemT                m_integrationTime;
};
using ProblemPtr = std::unique_ptr<Problem>;

class ProblemFactory
{
public:

  static ProblemPtr makeEulerSmoothProblem(std::size_t nPoints);
  static ProblemPtr makeStationaryMassFlowProblem(std::size_t nPoints);
  static ProblemPtr makeMovingMassFlowProblem(std::size_t nPoints);

};

} // namespace kae
