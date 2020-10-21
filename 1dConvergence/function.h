#pragma once

#include <memory>

#include "types.h"

namespace kae {

class IFunction
{
public:

  virtual ~IFunction() = default;

  ElemT getRkValue(ElemT x, ElemT t, ElemT dt, unsigned rkStep) const;
  ElemT getRkValueFirstTimeDerivative(ElemT x, ElemT t, ElemT dt, unsigned rkStep) const;

private:

  virtual ElemT getTrueValue                    (ElemT x, ElemT t) const = 0;
  virtual ElemT getTrueValueFirstTimeDerivative (ElemT x, ElemT t) const = 0;
  virtual ElemT getTrueValueSecondTimeDerivative(ElemT x, ElemT t) const = 0;
  virtual ElemT getTrueValueThirdTimeDerivative (ElemT x, ElemT t) const = 0;
};
using IFunctionPtr = std::unique_ptr<IFunction>;

class GasStateFunction
{
public:

  GasStateFunction(IFunctionPtr pRhoFunction, IFunctionPtr pUFunction, IFunctionPtr pPFunction);

  ElemT getRho(ElemT x, ElemT t, ElemT dt = 0, unsigned rkStep = 0) const;
  ElemT getU(ElemT x, ElemT t, ElemT dt = 0, unsigned rkStep = 0) const;
  ElemT getP(ElemT x, ElemT t, ElemT dt = 0, unsigned rkStep = 0) const;

  ElemT getRhoDerivative(ElemT x, ElemT t, ElemT dt = 0, unsigned rkStep = 0) const;
  ElemT getUDerivative(ElemT x, ElemT t, ElemT dt = 0, unsigned rkStep = 0) const;
  ElemT getPDerivative(ElemT x, ElemT t, ElemT dt = 0, unsigned rkStep = 0) const;

private:
  IFunctionPtr m_pRhoFunction;
  IFunctionPtr m_pUFunction;
  IFunctionPtr m_pPFunction;
};
using GasStateFunctionPtr = std::unique_ptr<GasStateFunction>;

class GasStateFunctionFactory
{
public:

  static GasStateFunctionPtr makeSmoothSineGasStateFunction(ElemT rhoAverage, ElemT rhoAmplitude, ElemT rhoPeriod);
};

} // namespace kae
