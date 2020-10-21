
#include "function.h"

namespace kae {

ElemT IFunction::getRkValue(ElemT x, ElemT t, ElemT dt, unsigned rkStep) const
{
  constexpr auto half = static_cast<ElemT>(0.5);
  const auto trueValue = getTrueValue(x, t);
  if (rkStep == 0U)
  {
    return trueValue;
  }
  else if (rkStep == 1U)
  {
    return trueValue + dt * getTrueValueFirstTimeDerivative(x, t);
  }
  else if (rkStep == 2U)
  {
    return trueValue +
      half * dt * getTrueValueFirstTimeDerivative(x, t) +
      half * half * dt * dt * getTrueValueSecondTimeDerivative(x, t);
  }

  return static_cast<ElemT>(0.0);
}

ElemT IFunction::getRkValueFirstTimeDerivative(ElemT x, ElemT t, ElemT dt, unsigned rkStep) const
{
  constexpr auto half = static_cast<ElemT>(0.5);
  const auto trueValue = getTrueValueFirstTimeDerivative(x, t);
  if (rkStep == 0U)
  {
    return trueValue;
  }
  else if (rkStep == 1U)
  {
    return trueValue + dt * getTrueValueSecondTimeDerivative(x, t);
  }
  else if (rkStep == 2U)
  {
    return trueValue +
      half * dt * getTrueValueSecondTimeDerivative(x, t) +
      half * half * dt * dt * getTrueValueThirdTimeDerivative(x, t);
  }

  return static_cast<ElemT>(0.0);
}

/**
 * Constant function
 */
class ConstantFunction : public IFunction
{
public:

  ConstantFunction(ElemT value) : m_value{ value } {}

private:

  ElemT getTrueValue                    (ElemT x, ElemT t) const override { return m_value; };
  ElemT getTrueValueFirstTimeDerivative (ElemT x, ElemT t) const override { return ElemT{}; };
  ElemT getTrueValueSecondTimeDerivative(ElemT x, ElemT t) const override { return ElemT{}; };
  ElemT getTrueValueThirdTimeDerivative (ElemT x, ElemT t) const override { return ElemT{}; };

private:

  ElemT m_value;
};

class SineFunction : public IFunction
{
public:

  SineFunction(ElemT average, ElemT amplitude, ElemT period)
    : m_average{ average }, m_amplitude{ amplitude }, m_period{ period } {}

private:

  ElemT getTrueValue(ElemT x, ElemT t) const override
  {
    return m_average + m_amplitude * sin(m_period * (x - t));
  }
  ElemT getTrueValueFirstTimeDerivative(ElemT x, ElemT t) const override
  {
    return -m_period * m_amplitude * cos(m_period * (x - t));
  };
  ElemT getTrueValueSecondTimeDerivative(ElemT x, ElemT t) const override
  {
    return -m_period * m_period * m_amplitude * sin(m_period * (x - t));
  };
  ElemT getTrueValueThirdTimeDerivative(ElemT x, ElemT t) const override
  {
    return m_period * m_period * m_period * m_amplitude * cos(m_period * (x - t));
  };

private:

  ElemT m_average;
  ElemT m_amplitude;
  ElemT m_period;
};

/*
 * GasStateFunction
 */

GasStateFunction::GasStateFunction(IFunctionPtr pRhoFunction, IFunctionPtr pUFunction, IFunctionPtr pPFunction)
  : m_pRhoFunction(std::move(pRhoFunction)),
    m_pUFunction(std::move(pUFunction)),
    m_pPFunction(std::move(pPFunction))
{
}

ElemT GasStateFunction::getRho(ElemT x, ElemT t, ElemT dt, unsigned rkStep) const
{
  return m_pRhoFunction->getRkValue(x, t, dt, rkStep);
}

ElemT GasStateFunction::getU(ElemT x, ElemT t, ElemT dt, unsigned rkStep) const
{
  return m_pUFunction->getRkValue(x, t, dt, rkStep);
}

ElemT GasStateFunction::getP(ElemT x, ElemT t, ElemT dt, unsigned rkStep) const
{
  return m_pPFunction->getRkValue(x, t, dt, rkStep);
}

ElemT GasStateFunction::getRhoDerivative(ElemT x, ElemT t, ElemT dt, unsigned rkStep) const
{
  return m_pRhoFunction->getRkValueFirstTimeDerivative(x, t, dt, rkStep);
}

ElemT GasStateFunction::getUDerivative(ElemT x, ElemT t, ElemT dt, unsigned rkStep) const
{
  return m_pUFunction->getRkValueFirstTimeDerivative(x, t, dt, rkStep);
}

ElemT GasStateFunction::getPDerivative(ElemT x, ElemT t, ElemT dt, unsigned rkStep) const
{
  return m_pPFunction->getRkValueFirstTimeDerivative(x, t, dt, rkStep);
}

GasStateFunctionPtr GasStateFunctionFactory::makeSmoothSineGasStateFunction(ElemT rhoAverage, ElemT rhoAmplitude, ElemT rhoPeriod)
{
  return std::make_unique<GasStateFunction>(
    std::make_unique<SineFunction>(rhoAverage, rhoAmplitude, rhoPeriod),
    std::make_unique<ConstantFunction>(static_cast<ElemT>(1.0)),
    std::make_unique<ConstantFunction>(static_cast<ElemT>(2.0)));
}

} // namespace kae 
