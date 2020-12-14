#pragma once

#include <cmath>
#include <vector>

namespace kae
{
template <class FloatT>
constexpr FloatT left = static_cast<FloatT>(0.0);

template <class FloatT>
constexpr FloatT right = static_cast<FloatT>(4.0);

template <class FloatT, std::size_t nPoints>
constexpr FloatT h = (right<FloatT> -left<FloatT>) / static_cast<FloatT>(nPoints - 1U);

enum class EShape
{
  eCircle,
  eSquare,
  eGradientVaryingCircle
};

template <class FloatT>
FloatT initialFunctionValue(FloatT x, FloatT y, EShape shape);

template <class FloatT>
FloatT reinitializedGoldFunctionValue(FloatT x, FloatT y, EShape shape);

template <class FloatT, std::size_t Nx, std::size_t Ny>
std::vector<FloatT> initialState(EShape shape)
{
  std::vector<FloatT> initState(Nx * Ny);
  for (std::size_t i{}; i < Nx; ++i)
  {
    for (std::size_t j{}; j < Ny; ++j)
    {
      const FloatT x = left<FloatT> +i * h<FloatT, Nx>;
      const FloatT y = left<FloatT> +j * h<FloatT, Ny>;
      const auto index = j * Nx + i;
      initState.at(index) = initialFunctionValue(x, y, shape);
    }
  }
  return initState;
}

template <class FloatT, std::size_t Nx, std::size_t Ny>
std::vector<FloatT> reinitializedGoldState(EShape shape)
{
  std::vector<FloatT> goldState(Nx * Ny);
  for (std::size_t i{}; i < Nx; ++i)
  {
    for (std::size_t j{}; j < Ny; ++j)
    {
      const FloatT x = left<FloatT> + i * h<FloatT, Nx>;
      const FloatT y = left<FloatT> + j * h<FloatT, Ny>;
      const auto index = j * Nx + i;
      goldState.at(index) = reinitializedGoldFunctionValue(x, y, shape);
    }
  }
  return goldState;
}

} // namespace kae
