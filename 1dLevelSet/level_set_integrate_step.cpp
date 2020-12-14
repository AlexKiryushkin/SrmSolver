
#include "level_set_integrate_step.h"

#include <cmath>
#include <iostream>

#include "level_set_derivatives.h"

namespace kae
{

namespace detail
{

template <class FloatT>
void reinitializeSubStep2d(const std::vector<FloatT> & initState,
                           const std::vector<FloatT> & prevState,
                           const std::vector<FloatT> & firstState,
                           std::vector<FloatT> &       currState,
                           const std::vector<FloatT> & xRoots,
                           const std::vector<FloatT> & yRoots,
                           std::size_t                 nx,
                           std::size_t                 ny,
                           FloatT                      prevWeight, 
                           FloatT                      hx,
                           FloatT                      hy)
{
  constexpr auto zero = static_cast<FloatT>(0.0);
  constexpr auto one = static_cast<FloatT>(1.0);
  const auto eps = std::numeric_limits<FloatT>::epsilon() / hx;

  for (std::size_t i{ 5U }; i < nx - 5U; ++i)
  {
    for (std::size_t j{ 5U }; j < ny - 5U; ++j)
    {
      const auto idx = j * nx + i;
      FloatT firstValue = (prevWeight != one) ? firstState[idx] : zero;

      const auto courant    = static_cast<FloatT>(0.8);
      const auto xRootPlus  = std::isfinite(xRoots[idx])      ? xRoots[idx]           : hx;
      const auto xRootMinus = std::isfinite(xRoots[idx - 1])  ? hx - xRoots[idx - 1]  : hx;
      const auto yRootPlus  = std::isfinite(yRoots[idx])      ? yRoots[idx]           : hy;
      const auto yRootMinus = std::isfinite(yRoots[idx - nx]) ? hy - yRoots[idx - nx] : hy;
      const auto minRoot    = std::min({ xRootPlus, xRootMinus, yRootPlus, yRootMinus });
      const auto dt = courant * minRoot / 2;

      const FloatT initValue = initState[idx];
      const FloatT sgdValue = prevState[idx];
      const FloatT grad = getLevelSetAbsGradient(prevState.data(), xRoots.data(), yRoots.data(), idx, nx, (initValue > 0), hx, hy);
      const FloatT sgn = ( std::fabs(initValue) < eps ) ?  zero : ( ( initValue > 0 ) ? one : -one );
      const FloatT val = sgdValue - dt * sgn * (grad - static_cast<FloatT>(1.0));
      currState[idx] = (1 - prevWeight) * firstValue + prevWeight * val;
    }
  }
}

} // namespace detail

template <class FloatT>
void reinitializeStep2d(const std::vector<FloatT> & initState,
                        std::vector<FloatT> & prevState,
                        std::vector<FloatT> & firstState,
                        std::vector<FloatT> & currState,
                        const std::vector<FloatT> & xRoots,
                        const std::vector<FloatT> & yRoots,
                        std::size_t nx,
                        std::size_t ny,
                        FloatT hx,
                        FloatT hy,
                        ETimeDiscretizationOrder timeOrder)
{
  switch (timeOrder)
  {
  case ETimeDiscretizationOrder::eOne:
  {
    detail::reinitializeSubStep2d(initState, prevState, firstState, currState, xRoots, yRoots, nx, ny, static_cast<FloatT>(1.0), hx, hy);
    break;
  }
  case ETimeDiscretizationOrder::eTwo:
  {
    detail::reinitializeSubStep2d(initState, prevState, currState, firstState, xRoots, yRoots, nx, ny, static_cast<FloatT>(1.0), hx, hy);
    detail::reinitializeSubStep2d(initState, firstState, prevState, currState, xRoots, yRoots, nx, ny, static_cast<FloatT>(0.5), hx, hy);
    break;
  }
  case ETimeDiscretizationOrder::eThree:
  {
    detail::reinitializeSubStep2d(initState, prevState, firstState, currState, xRoots, yRoots, nx, ny, static_cast<FloatT>(1.0), hx, hy);
    detail::reinitializeSubStep2d(initState, currState, prevState, firstState, xRoots, yRoots, nx, ny, static_cast<FloatT>(0.25), hx, hy);
    detail::reinitializeSubStep2d(initState, firstState, prevState, currState, xRoots, yRoots, nx, ny, static_cast<FloatT>(2.0 / 3.0), hx, hy);
    break;
  }
  default:
    break;
  }
}

template void reinitializeStep2d<float>(const std::vector<float>& initState,
                                        std::vector<float>& prevState,
                                        std::vector<float>& firstState,
                                        std::vector<float>& currState,
                                        const std::vector<float> & xRoots,
                                        const std::vector<float> & yRoots,
                                        std::size_t nx,
                                        std::size_t ny,
                                        float hx,
                                        float hy,
                                        ETimeDiscretizationOrder timeOrder);
template void reinitializeStep2d<double>(const std::vector<double>& initState,
                                        std::vector<double>& prevState,
                                        std::vector<double>& firstState,
                                        std::vector<double>& currState,
                                        const std::vector<double> & xRoots,
                                        const std::vector<double> & yRoots,
                                        std::size_t nx,
                                        std::size_t ny,
                                        double hx,
                                        double hy,
                                        ETimeDiscretizationOrder timeOrder);

} // namespace kae
