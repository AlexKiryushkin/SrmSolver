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

  std::size_t              getStartIdx()            const { return m_pBoundary->getStartIdx(*m_pGrid); }
  std::size_t              getEndIdx()              const { return m_pBoundary->getEndIdx(*m_pGrid); }
  ElemT                    getXBoundaryLeft()       const { return m_pBoundary->getXBoundaryLeft(); }
  ElemT                    getXBoundaryRight()      const { return m_pBoundary->getXBoundaryRight(); }
  ElemT                    getH()                   const { return m_pGrid->getH(); }
  ElemT                    getX(std::size_t i)      const { return m_pGrid->getX(i); }
  ElemT                    getIntegrationTime()     const { return m_integrationTime; }
  std::vector<GasState>    getInitialState()        const { return getGasState(0.0); }
  std::vector<GasState>    getIAnalyticalSolution() const { return getGasState(getIntegrationTime()); }

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

};

class IProblem
{
public:

  virtual ~IProblem() = default;

  virtual unsigned              getStartIdx()            const = 0;
  virtual unsigned              getEndIdx()              const = 0;
  virtual ElemT                 getH()                   const = 0;
  virtual ElemT                 getX(std::size_t i)      const = 0;
  virtual void setGhostValues(std::vector<GasState>& gasStates, ElemT t, ElemT dt, unsigned rkStep) = 0;
  virtual ElemT                 getIntegrationTime()     const = 0;
  virtual std::vector<GasState> getInitialState()        const = 0;
  virtual std::vector<GasState> getIAnalyticalSolution() const = 0;

};

class EulerSmoothProblem : public IProblem
{
public:

  explicit EulerSmoothProblem(std::size_t nPoints) : m_nPoints{ nPoints } {}

  constexpr static ElemT xLeft  = -static_cast<ElemT>(M_PI);
  constexpr static ElemT xRight = static_cast<ElemT>(M_PI);

  constexpr static unsigned stencilWidth = 10U;

  unsigned getStartIdx() const override { return stencilWidth / 2U; };
  unsigned getEndIdx() const override { return static_cast<unsigned>(m_nPoints - stencilWidth / 2U); };

  void setGhostValues(std::vector<GasState>& gasStates, ElemT t, ElemT dt, unsigned rkStep) override;
  ElemT getH() const override { return (xRight - xLeft) / static_cast<ElemT>(m_nPoints - 1U); }
  ElemT getIntegrationTime() const override { return 2.0; }
  std::vector<GasState> getInitialState() const override;
  std::vector<GasState> getIAnalyticalSolution() const override;

  ElemT getX(std::size_t i) const override { return xLeft + i * getH(); }

protected:

  GasState getGhostValue(ElemT x, ElemT t, ElemT dt, unsigned rkStep) const;
  std::size_t getNPoints() const { return m_nPoints; }

protected:

  std::vector<GasState> getGoldState(ElemT time) const;

  ElemT getRho(ElemT x, ElemT t, ElemT dt = 0, unsigned rkStep = 0) const;
  ElemT getU  (ElemT x, ElemT t, ElemT dt = 0, unsigned rkStep = 0) const;
  ElemT getP  (ElemT x, ElemT t, ElemT dt = 0, unsigned rkStep = 0) const;

  ElemT getRhoFirstTimeDerivative    (ElemT x, ElemT t, ElemT dt = 0, unsigned rkStep = 0) const;
  ElemT getRhoSecondTimeDerivative   (ElemT x, ElemT t, ElemT dt = 0, unsigned rkStep = 0) const;
  ElemT getRhoThirdTimeDerivative    (ElemT x, ElemT t, ElemT dt = 0, unsigned rkStep = 0) const;
  ElemT getRhoFourthTimeDerivative   (ElemT x, ElemT t, ElemT dt = 0, unsigned rkStep = 0) const;

private:

  constexpr static ElemT multiplier{ static_cast<ElemT>(0.2) };
  constexpr static ElemT period{ static_cast<ElemT>(4) };

  std::size_t m_nPoints;
};

class StationaryMassFlowProblem : public EulerSmoothProblem
{
public:

  explicit StationaryMassFlowProblem(std::size_t nPoints) : EulerSmoothProblem{ nPoints } {}

  void setGhostValues(std::vector<GasState>& gasStates, ElemT t, ElemT dt, unsigned rkStep) override;

private:

  ElemT getXBoundary() const { return xLeft + (static_cast<ElemT>(getStartIdx()) - static_cast<ElemT>(0.1)) * getH(); }
};

} // namespace kae
