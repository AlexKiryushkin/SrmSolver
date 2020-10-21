
#include "problem.h"

#include <Eigen/Eigen>

#include "mass_flow.h"

namespace {

template <class U, unsigned order = 3U>
Eigen::Matrix<ElemT, order, 1> getPolynomial(const kae::GasState* pState,
                                             unsigned startIdx, 
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

  return (lhs.transpose() * lhs).ldlt().solve(lhs.transpose() * rhs);
}

template <class U>
Eigen::Matrix<ElemT, 3, 1> getWenoPolynomial(const kae::GasState* pState,
                                             unsigned startIdx,
                                             ElemT xBoundary,
                                             ElemT xLeft,
                                             ElemT h)
{
  constexpr ElemT epsilon{ static_cast<ElemT>( 1e-6 ) };
  const auto p0 = getPolynomial<U, 1U>(pState, startIdx, xBoundary, xLeft, h);
  const auto p1 = getPolynomial<U, 2U>(pState, startIdx, xBoundary, xLeft, h);
  const auto p2 = getPolynomial<U, 3U>(pState, startIdx, xBoundary, xLeft, h);

  const auto d0 = h * h;
  const auto d1 = h;
  const auto d2 = 1 - d1 - d0;

  const auto betta0 = h * h;
  const auto betta1 = kae::sqr( p1( 1 ) );
  const auto betta2 = kae::sqr( h * p2( 1 ) ) + h * h * h * p2( 1 ) * p2( 2 )
    + static_cast<ElemT>(13.0 / 12.0) * kae::sqr( h * h * p2( 2 ) );

  const auto alpha0 = d0 / kae::sqr( betta0 + epsilon );
  const auto alpha1 = d1 / kae::sqr( betta1 + epsilon );
  const auto alpha2 = d2 / kae::sqr( betta2 + epsilon );
  const auto sum = alpha0 + alpha1 + alpha2;

  const auto omega0 = alpha0 / sum;
  const auto omega1 = alpha1 / sum;
  const auto omega2 = alpha2 / sum;

  return Eigen::Matrix<ElemT, 3, 1>{
    omega0 * p0(0) + omega1 * p1(0) + omega2 * p2(0),
                     omega1 * p1(1) + omega2 * p2(1),
                                      omega2 * p2(2)
    };
}

} // namespace 

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
  auto pGrid             = GridFactory::makeSimpleGrid(nPoints, -M_PI, M_PI);
  auto pGasStateFunction = GasStateFunctionFactory::makeSmoothSineGasStateFunction(1.0, 0.2, 1.0);
  auto pBoundary         = BoundaryFactory::makeStationaryBoundary(pGrid->getXLeft() + 3.5 * pGrid->getH(),
                                                                   pGrid->getXRight() - 3.5 * pGrid->getH());
  auto pGhostValueSetter = GhostValueSetterFactory::makeExactGhostValueSetter();
  return std::make_unique<Problem>(std::move(pGrid), 
                                   std::move(pBoundary), 
                                   std::move(pGasStateFunction),
                                   std::move(pGhostValueSetter),
                                   static_cast<ElemT>(2.0));
}

ProblemPtr ProblemFactory::makeStationaryMassFlowProblem(std::size_t nPoints)
{
}

void EulerSmoothProblem::setGhostValues(std::vector<GasState> & gasStates, ElemT t, ElemT dt, unsigned rkStep)
{
  const auto startIdx = getStartIdx();
  const auto endIdx   = getEndIdx();

  for (std::size_t i{}; i < startIdx; ++i)
  {
    const auto x    = getX(i);
    gasStates.at(i) = getGhostValue(x, t, dt, rkStep);
  }
  for (std::size_t i{ m_nPoints - 1U }; i >= endIdx; --i)
  {
    const auto x    = getX(i);
    gasStates.at(i) = getGhostValue(x, t, dt, rkStep);
  }
}

std::vector<GasState> EulerSmoothProblem::getInitialState() const
{
  return getGoldState(0.0);
}

std::vector<GasState> EulerSmoothProblem::getIAnalyticalSolution() const
{
  return getGoldState(getIntegrationTime());
}

GasState EulerSmoothProblem::getGhostValue(ElemT x, ElemT t, ElemT dt, unsigned rkStep) const
{
  return GasState{ getRho(x, t, dt, rkStep), getU(x, t, dt, rkStep), getP(x, t, dt, rkStep) };
}

std::vector<GasState> EulerSmoothProblem::getGoldState(ElemT time) const
{
  std::vector<GasState> initialGasValues(m_nPoints);

  for (std::size_t i{}; i < m_nPoints; ++i)
  {
    const auto x = getX(i);
    initialGasValues.at(i) = GasState{ getRho(x, time), getU(x, time), getP(x, time) };
  }

  return initialGasValues;
}

ElemT EulerSmoothProblem::getRho(ElemT x, ElemT t, ElemT dt, unsigned rkStep) const
{
  constexpr auto half = static_cast<ElemT>(0.5);
  const auto rho = static_cast<ElemT>(1.0) + multiplier * sin(period * (x - t));
  if (rkStep == 0U)
  {
    return rho;
  }
  else if (rkStep == 1U)
  {
    return rho + dt * getRhoFirstTimeDerivative(x, t);
  }
  else if (rkStep == 2U)
  {
    return rho + half * dt * getRhoFirstTimeDerivative(x, t) + sqr(half) * dt * dt * getRhoSecondTimeDerivative(x, t);
  }

  return static_cast<ElemT>(0.0);
}

ElemT EulerSmoothProblem::getP(ElemT, ElemT, ElemT, unsigned) const
{
  return static_cast<ElemT>(2.0);
}

ElemT EulerSmoothProblem::getU(ElemT, ElemT, ElemT, unsigned) const
{
  return static_cast<ElemT>(1.0);
}

ElemT EulerSmoothProblem::getRhoFirstTimeDerivative(ElemT x, ElemT t, ElemT dt, unsigned rkStep) const
{
  constexpr auto half = static_cast<ElemT>(0.5);
  const auto rhoDerivative = -period * multiplier * cos(period * (x - t));
  if (rkStep == 0U)
  {
    return rhoDerivative;
  }
  else if (rkStep == 1U)
  {
    return rhoDerivative + dt * getRhoSecondTimeDerivative(x, t);
  }
  else if (rkStep == 2U)
  {
    return rhoDerivative + half * dt * getRhoSecondTimeDerivative(x, t) + sqr(half) * dt * dt * getRhoThirdTimeDerivative(x, t);
  }

  return static_cast<ElemT>(0.0);
}

ElemT EulerSmoothProblem::getRhoSecondTimeDerivative(ElemT x, ElemT t, ElemT dt, unsigned rkStep) const
{
  constexpr auto half = static_cast<ElemT>(0.5);
  const auto rhoDerivative = -period * period * multiplier * sin(period * (x - t));
  if (rkStep == 0U)
  {
    return rhoDerivative;
  }
  else if (rkStep == 1U)
  {
    return rhoDerivative + dt * getRhoThirdTimeDerivative(x, t);
  }
  else if (rkStep == 2U)
  {
    return rhoDerivative + half * dt * getRhoThirdTimeDerivative(x, t) + sqr(half) * dt * dt * getRhoFourthTimeDerivative(x, t);
  }

  return static_cast<ElemT>(0.0);
}

ElemT EulerSmoothProblem::getRhoThirdTimeDerivative(ElemT x, ElemT t, ElemT, unsigned) const
{
  return period * period * period * multiplier * cos(period * (x - t));
}

ElemT EulerSmoothProblem::getRhoFourthTimeDerivative(ElemT x, ElemT t, ElemT, unsigned) const
{
  return period * period * period * period * multiplier * sin(period * (x - t));
}

/**
 *  StationaryMassFlowProblem
 */

void StationaryMassFlowProblem::setGhostValues(std::vector<GasState> & gasStates, ElemT t, ElemT dt, unsigned rkStep)
{
  const auto startIdx = getStartIdx();
  const auto endIdx   = getEndIdx();

  const auto & closestState  = gasStates.at(startIdx);
  const auto   rhoP3 = getWenoPolynomial<Rho>   (gasStates.data(), startIdx, getXBoundary(), xLeft, getH());
  const auto   uP3   = getWenoPolynomial<MinusU>(gasStates.data(), startIdx, getXBoundary(), xLeft, getH());
  const auto   pP3   = getWenoPolynomial<P>     (gasStates.data(), startIdx, getXBoundary(), xLeft, getH());

  const auto goldState      = getGhostValue(getXBoundary(), t, dt, rkStep);
  constexpr auto nu         = static_cast<ElemT>(0.7);
  const auto massFlowParams = kae::MassFlowParams{
    -goldState.rho * goldState.u / std::pow(goldState.p, nu),
    nu,
    kae::RhoEnthalpyFlux::get(goldState) / kae::MassFlux::get(goldState) };

  const auto massFlowGasState = getMassFlowGhostValue(GasState{ rhoP3(0), uP3(0), pP3(0) }, closestState, massFlowParams);

  for (std::size_t i{}; i < startIdx; ++i)
  {
    const auto dx = getXBoundary() - getX(i);
    const auto rhoDerivative = getRhoFirstTimeDerivative(getXBoundary(), t, dt, rkStep);
    const auto sol = getMassFlowDerivatives(massFlowGasState, closestState, goldState, rhoDerivative, uP3(1U), pP3(1U), massFlowParams);

    gasStates.at(i) = GasState{
          massFlowGasState.rho + sol(0) * dx + rhoP3(2) * dx * dx,
        -(massFlowGasState.u   + sol(1) * dx + uP3(2)   * dx * dx),
          massFlowGasState.p   + sol(2) * dx + pP3(2)   * dx * dx
    };
  }

  for (std::size_t i{ getNPoints() - 1U }; i >= endIdx; --i)
  {
    const auto x = getX(i);
    gasStates.at(i) = getGhostValue(x, t, dt, rkStep);
  }

}

} // namespace kae
