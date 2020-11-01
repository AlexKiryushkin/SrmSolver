#pragma once

#include <memory>
#include <vector>

#include "gas_state.h"
#include "problem.h"
#include "types.h"

namespace kae {

class GasDynamicsSolver
{
public:

  GasDynamicsSolver(ProblemPtr pProblem, unsigned order) : m_pProblem{ std::move(pProblem) }, m_order{ order } {}

  std::vector<GasState> solve();
  std::vector<GasState> goldSolution() const { return m_pProblem->getIAnalyticalSolution(); }
  const Problem& getProblem() const { return *m_pProblem; }

  ElemT getH() const { return  m_pProblem->getH(); }

private:

  void rungeKuttaStep(std::vector<GasState> & prevGasValues,
                      std::vector<GasState> & firstGasValues,
                      std::vector<GasState> & currGasValues,
                      ElemT                   lambda,
                      ElemT                   t,
                      ElemT                   dt) const;

  void rungeKuttaSubStep(std::vector<GasState> &       prevGasValues,
                         const std::vector<GasState> & firstGasValues,
                         std::vector<GasState> &       currGasValues,
                         ElemT                         lambda,
                         ElemT                         dt,
                         ElemT                         prevWeight) const;

private:

  ProblemPtr m_pProblem;
  unsigned m_order;
};

} // namespace kae 
