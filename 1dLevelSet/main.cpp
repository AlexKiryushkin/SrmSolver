
#include <algorithm>
#include <cmath>
#include <iostream>

#include "quartic_solve.h"

template <class FloatT>
constexpr FloatT leftX = static_cast<FloatT>(-5.0);

template <class FloatT>
constexpr FloatT rightX = static_cast<FloatT>(5.0);

constexpr std::size_t nPoints = 100U;

template <class FloatT>
constexpr FloatT h = (rightX<FloatT> -leftX<FloatT>) / static_cast<FloatT>(nPoints - 1U);

template <class FloatT>
FloatT initialFunctionValue(FloatT x)
{
  return (x - static_cast<FloatT>(0.4) * h<FloatT>) * (x - 3.456) * (x + 7) / 2 + 1;
}

template <class FloatT>
std::vector<FloatT> initialState()
{
  std::vector<FloatT> initState(nPoints);
  for (std::size_t i{}; i < nPoints; ++i)
  {
    const FloatT x = leftX<FloatT> + i * h<FloatT>;
    initState.at(i) = initialFunctionValue(x);
  }
  return initState;
}

int main()
{
  std::vector<double> initState = initialState<double>();
  std::vector<double> roots(std::size(initState), std::numeric_limits<double>::infinity());
  for (std::size_t idx{ 5U }; idx < nPoints - 5U; ++idx)
  {
    if (initState.at(idx) * initState.at(idx + 1) < 0)
    {
      roots.at(idx) = leftX<double> +kae::quarticSolve(initState, idx, h<double>);
      std::cout << "Root between " << idx << " and " << idx + 1 << "\n";
      std::cout << "Calculated root is " << roots.at(idx) << "\n";
      std::cout << "Function value at this point " << initialFunctionValue(roots.at(idx)) << "\n";
    }
  }
  std::erase_if(roots, std::isinf<double>);
}