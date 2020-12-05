
#include <algorithm>
#include <bit>
#include <bitset>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>

#include "level_set_integrate_step.h"
#include "quartic_solve.h"

template <class FloatT>
constexpr FloatT leftX = static_cast<FloatT>(0.0);

template <class FloatT>
constexpr FloatT rightX = static_cast<FloatT>(10.0);

template <class FloatT, std::size_t nPoints>
constexpr FloatT h = (rightX<FloatT> - leftX<FloatT>) / static_cast<FloatT>(nPoints - 1U);

template <std::size_t nPoints, class FloatT>
FloatT initialFunctionValue(FloatT x)
{
  constexpr auto offset = static_cast<FloatT>(0.4)* h<FloatT, nPoints>;
  return ((x - 5.0 - offset) * (x - 8.0 - offset) * (x + 7) / 2) / 8;
}

template <std::size_t nPoints, class FloatT>
FloatT reinitializedGoldFunctionValue(FloatT x)
{
  constexpr auto offset = static_cast<FloatT>(0.4)* h<FloatT, nPoints>;
  return std::fabs(x - static_cast<FloatT>(6.5) - offset) - static_cast<FloatT>(1.5);
}

template <class FloatT, std::size_t nPoints>
std::vector<FloatT> initialState()
{
  std::vector<FloatT> initState(nPoints);
  for (std::size_t i{}; i < nPoints; ++i)
  {
    const FloatT x = leftX<FloatT> + i * h<FloatT, nPoints>;
    initState.at(i) = initialFunctionValue<nPoints>(x);
  }
  return initState;
}

template <class FloatT, std::size_t nPoints>
std::vector<FloatT> reinitializedGoldState()
{
  std::vector<FloatT> goldState(nPoints);
  for (std::size_t i{}; i < nPoints; ++i)
  {
    const FloatT x = leftX<FloatT> + i * h<FloatT, nPoints>;
    goldState.at(i) = reinitializedGoldFunctionValue<nPoints>(x);
  }
  return goldState;
}

template <class FloatT, std::size_t nPoints>
FloatT getResidual()
{
  const std::vector<double> initState = initialState<double, nPoints>();
  const std::vector<double> goldState = reinitializedGoldState<double, nPoints>();

  std::vector<double> prevState = initState;
  std::vector<double> firstState = initState;
  std::vector<double> currState = initState;
  std::vector<double> roots(std::size(initState), std::numeric_limits<double>::infinity());

  for (std::size_t idx{}; idx < nPoints - 1; ++idx)
  {
    if (initState[idx] * initState[idx + 1] <= 0)
    {
      roots[idx] = leftX<FloatT> +kae::quarticSolve(initState, idx, h<FloatT, nPoints>);
    }
  }

  constexpr auto iterationCount = nPoints;
  for (std::size_t iteration{}; iteration < iterationCount; ++iteration)
  {
    std::swap(prevState, currState);
    kae::reinitializeStep(prevState, firstState, currState, roots, h<double, nPoints>, kae::ETimeDiscretizationOrder::eThree);
  }

  for (std::size_t idx{}; idx < nPoints - 1; ++idx)
  {
    if (initState[idx] * initState[idx + 1] <= 0)
    {
      const auto numericalRoot = leftX<FloatT> +kae::quarticSolve(currState, idx, h<FloatT, nPoints>);
      std::cout << "gold root " << roots[idx] << ". numerical root " << numericalRoot
                << ". diff " << std::fabs(roots[idx] - numericalRoot) << "\n";
    }
  }

  constexpr auto offset = 20U;
  const auto error = std::transform_reduce(
    std::next(std::begin(goldState), offset),
    std::prev(std::end(goldState), offset),
    std::next(std::begin(currState), offset),
    0.0, std::plus<>{},
    [](auto&& lhs, auto&& rhs) { return std::fabs(lhs - rhs); });
  return error;
}

int main()
{
  const auto err100 = getResidual<double, 100>();
  const auto err200 = getResidual<double, 200>();
  const auto err400 = getResidual<double, 400>();
  const auto err800 = getResidual<double, 800>();

  std::cout << "N = 100; " << "err = " << err100 << "; order = " << "\n";
  std::cout << "N = 200; " << "err = " << err200 << "; order = " << std::log2(err100 / err200) << "\n";
  std::cout << "N = 400; " << "err = " << err400 << "; order = " << std::log2(err200 / err400) << "\n";
  std::cout << "N = 800; " << "err = " << err800 << "; order = " << std::log2(err400 / err800) << "\n";
}