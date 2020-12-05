
#include "level_set_integrate_step.h"

#include <cmath>

#include "level_set_derivatives.h"

namespace kae
{

namespace detail
{

template <class FloatT>
void reinitializeSubStep(const std::vector<FloatT> & prevState,
                         const std::vector<FloatT> & firstState,
                         std::vector<FloatT> &       currState, 
                         const std::vector<FloatT> & roots,
                         FloatT                      dt, 
                         FloatT                      prevWeight, 
                         FloatT                      h)
{
  for (std::size_t idx{ 5U }; idx < std::size(prevState) - 5U; ++idx)
  {
    FloatT firstValue{};
    if (prevWeight != static_cast<FloatT>(1.0))
    {
      firstValue = firstState[idx];
    }

    const FloatT sgdValue = prevState[idx];
    const FloatT grad = getLevelSetAbsGradient(prevState.data(), roots.data(), idx, (sgdValue > 0), h);
    const FloatT sgn = sgdValue / std::hypot(sgdValue, grad * h);
    const FloatT val = sgdValue - dt * sgn * (grad - static_cast<FloatT>(1.0));

    currState[idx] = (1 - prevWeight) * firstValue + prevWeight * val;
  }
}

} // namespace detail

template <class FloatT>
void reinitializeStep(std::vector<FloatT> & prevState,
                      std::vector<FloatT> & firstState,
                      std::vector<FloatT> & currState,
                      const std::vector<FloatT> & roots,
                      FloatT h,
                      ETimeDiscretizationOrder timeOrder)
{
  const auto courant = static_cast<FloatT>(0.8);
  const auto dt = courant * h;

  switch (timeOrder)
  {
    case ETimeDiscretizationOrder::eOne:
    {
      detail::reinitializeSubStep(prevState, firstState, currState, roots, dt, static_cast<FloatT>(1.0), h);
      break;
    }
    case ETimeDiscretizationOrder::eTwo:
    {
      detail::reinitializeSubStep(prevState, currState, firstState, roots, dt, static_cast<FloatT>(1.0), h);
      detail::reinitializeSubStep(firstState, prevState, currState, roots, dt, static_cast<FloatT>(0.5), h);
      break;
    }
    case ETimeDiscretizationOrder::eThree:
    {
      detail::reinitializeSubStep(prevState, firstState, currState, roots, dt, static_cast<FloatT>(1.0),       h);
      detail::reinitializeSubStep(currState, prevState, firstState, roots, dt, static_cast<FloatT>(0.25),      h);
      detail::reinitializeSubStep(firstState, prevState, currState, roots, dt, static_cast<FloatT>(2.0 / 3.0), h);
      break;
    }
    default:
      break;
  }
}

template void reinitializeStep<float>(std::vector<float>& prevState,
                                      std::vector<float>& firstState,
                                      std::vector<float>& currState,
                                      const std::vector<float> & roots,
                                      float h,
                                      ETimeDiscretizationOrder timeOrder);
template void reinitializeStep<double>(std::vector<double>& prevState,
                                       std::vector<double>& firstState,
                                       std::vector<double>& currState,
                                       const std::vector<double> & roots,
                                       double h,
                                       ETimeDiscretizationOrder timeOrder);

} // namespace kae
