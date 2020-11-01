
#include "ghost_value_setter.h"

#include <Eigen/Eigen>

#include "mass_flow.h"
#include "problem.h"

namespace {

template <class U, unsigned order = 3U>
Eigen::Matrix<ElemT, order, 1> getPolynomial(const kae::GasState* pState,
                                             std::size_t startIdx,
                                             ElemT xBoundary,
                                             ElemT xLeft,
                                             ElemT h)
{
  Eigen::Matrix<ElemT, order, order> lhs;
  Eigen::Matrix<ElemT, order, 1> rhs;
  for (std::size_t rowIdx{}; static_cast<int>(rowIdx) < lhs.cols(); ++rowIdx)
  {
    const auto i = startIdx + rowIdx;
    const auto dx = xBoundary - xLeft - i * h;

    ElemT value = 1;
    for (std::size_t colIdx{}; static_cast<int>(colIdx) < lhs.cols(); ++colIdx)
    {
      lhs(rowIdx, colIdx) = value;
      value *= dx;
    }
    rhs(rowIdx) = U::get(pState[i]);
  }

  return (lhs.transpose() * lhs).llt().solve(lhs.transpose() * rhs);
}

template <class U>
Eigen::Matrix<ElemT, 3, 1> getWenoPolynomial(const kae::GasState* pState,
    std::size_t startIdx,
                                             ElemT xBoundary,
                                             ElemT xLeft,
                                             ElemT h)
{
  constexpr ElemT epsilon{ static_cast<ElemT>(1e-6) };
  const auto p0 = getPolynomial<U, 1U>(pState, startIdx, xBoundary, xLeft, h);
  const auto p1 = getPolynomial<U, 2U>(pState, startIdx, xBoundary, xLeft, h);
  const auto p2 = getPolynomial<U, 3U>(pState, startIdx, xBoundary, xLeft, h);

  const auto d0 = h * h;
  const auto d1 = h;
  const auto d2 = 1 - d1 - d0;

  const auto betta0 = h * h;
  const auto betta1 = kae::sqr(p1(1));
  const auto betta2 = kae::sqr(h * p2(1)) + h * h * h * p2(1) * p2(2)
    + static_cast<ElemT>(13.0 / 12.0) * kae::sqr(h * h * p2(2));

  const auto alpha0 = d0 / kae::sqr(betta0 + epsilon);
  const auto alpha1 = d1 / kae::sqr(betta1 + epsilon);
  const auto alpha2 = d2 / kae::sqr(betta2 + epsilon);
  const auto sum = alpha0 + alpha1 + alpha2;

  const auto omega0 = alpha0 / sum;
  const auto omega1 = alpha1 / sum;
  const auto omega2 = alpha2 / sum;

  return Eigen::Matrix<ElemT, 3, 1>{
    omega0* p0(0) + omega1 * p1(0) + omega2 * p2(0),
      omega1* p1(1) + omega2 * p2(1),
      omega2* p2(2)
  };
}

} // namespace 

namespace kae {

void IGhostValueSetter::setGhostValues(std::vector<GasState>& gasStates, const Problem& problem, 
                                       ElemT t, ElemT dt, unsigned rkStep) const
{
  const auto startIdx = problem.getStartIdx();
  const auto endIdx = problem.getEndIdx();

  for (std::size_t i{}; i < startIdx; ++i)
  {
    const auto x = problem.getX(i);
    gasStates.at(i) = getLeftGhostValue(gasStates, problem, x, t, dt, rkStep);
  }

  for (std::size_t i{ gasStates.size() - 1U }; i >= endIdx; --i)
  {
    const auto x = problem.getX(i);
    gasStates.at(i) = getRightGhostValue(gasStates, problem, x, t, dt, rkStep);
  }
}

class ExactGhostValueSetter : public IGhostValueSetter
{
  GasState getLeftGhostValue(std::vector<GasState>& gasStates, const Problem& problem, ElemT x, ElemT t, ElemT dt, unsigned rkStep) const override
  {
    auto && gasStateFunction = problem.getGasStateFunction();
    return GasState{ gasStateFunction.getRho(x, t, dt, rkStep),
                     gasStateFunction.getU(x, t, dt, rkStep),
                     gasStateFunction.getP(x, t, dt, rkStep) };
  }

  GasState getRightGhostValue(std::vector<GasState>& gasStates, const Problem& problem, ElemT x, ElemT t, ElemT dt, unsigned rkStep) const override
  {
    return getLeftGhostValue(gasStates, problem, x, t, dt, rkStep);
  }
};

class MassFlowGhostValueSetter : public IGhostValueSetter
{
public:
  MassFlowGhostValueSetter(ElemT rhoPReciprocal) : m_rhoPReciprocal{ rhoPReciprocal } {}

private:

  GasState getLeftGhostValue(std::vector<GasState>& gasStates, const Problem& problem, ElemT x, ElemT t, ElemT dt, unsigned rkStep) const override
  {
    const auto startIdx = problem.getStartIdx();
    const auto endIdx = problem.getEndIdx();
    const auto xBoundary = problem.getXBoundaryLeft(t, dt, rkStep);
    const auto h = problem.getH();

    const auto& closestState = gasStates.at(startIdx);
    const auto   rhoP3 = getWenoPolynomial<Rho>(gasStates.data(), startIdx, xBoundary, problem.getXLeft(), h);
    const auto   uP3   = getWenoPolynomial<MinusU>(gasStates.data(), startIdx, xBoundary, problem.getXLeft(), h);
    const auto   pP3   = getWenoPolynomial<P>(gasStates.data(), startIdx, xBoundary, problem.getXLeft(), h);

    auto&& gasStateFunction = problem.getGasStateFunction();
    const auto goldState = gasStateFunction.getGasState(xBoundary, t, dt, rkStep);
    constexpr auto nu = static_cast<ElemT>(0.7);
    const auto massFlowParams = kae::MassFlowParams{
      -(1 + m_rhoPReciprocal) * goldState.rho * goldState.u / std::pow(goldState.p, nu),
      nu,
      kae::RhoEnthalpyFlux::get(goldState) / kae::MassFlux::get(goldState) };

    const auto massFlowGasState = getMassFlowGhostValue(GasState{ rhoP3(0), uP3(0), pP3(0) }, closestState, massFlowParams, m_rhoPReciprocal);

    const auto dx = xBoundary - x;
    const auto rhoDerivative = gasStateFunction.getRhoDerivative(xBoundary, t, dt, rkStep);
    const auto sol = getMassFlowDerivatives(massFlowGasState, closestState, goldState, rhoDerivative, uP3(1U), pP3(1U), massFlowParams, m_rhoPReciprocal);

    return GasState{
          massFlowGasState.rho + sol(0) * dx + rhoP3(2) * dx * dx,
        -(massFlowGasState.u + sol(1) * dx + uP3(2) * dx * dx),
          massFlowGasState.p + sol(2) * dx + pP3(2) * dx * dx
    };
  }

  GasState getRightGhostValue(std::vector<GasState>& gasStates, const Problem& problem, ElemT x, ElemT t, ElemT dt, unsigned rkStep) const override
  {
    auto&& gasStateFunction = problem.getGasStateFunction();
    return GasState{ gasStateFunction.getRho(x, t, dt, rkStep),
                     gasStateFunction.getU(x, t, dt, rkStep),
                     gasStateFunction.getP(x, t, dt, rkStep) };
  }

private:

  ElemT m_rhoPReciprocal;
};

IGhostValueSetterPtr GhostValueSetterFactory::makeExactGhostValueSetter()
{
  return std::make_unique<ExactGhostValueSetter>();
}

IGhostValueSetterPtr GhostValueSetterFactory::makeMassFlowGhostValueSetter(ElemT rhoPReciprocal)
{
    return std::make_unique<MassFlowGhostValueSetter>(rhoPReciprocal);
}

} // namespace kae
