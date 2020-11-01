#pragma once

#include <memory>
#include <vector>

#include "gas_state.h"
#include "types.h"

namespace kae {

class Problem;

class IGhostValueSetter
{
public:
  virtual ~IGhostValueSetter() = default;

  void setGhostValues(std::vector<GasState>& gasStates, const Problem & problem, ElemT t, ElemT dt, unsigned rkStep) const;

private:

  virtual GasState getLeftGhostValue(std::vector<GasState>& gasStates, const Problem& problem, ElemT x, ElemT t, ElemT dt, unsigned rkStep) const = 0;
  virtual GasState getRightGhostValue(std::vector<GasState>& gasStates, const Problem& problem, ElemT x, ElemT t, ElemT dt, unsigned rkStep) const = 0;

};
using IGhostValueSetterPtr = std::unique_ptr<IGhostValueSetter>;

class GhostValueSetterFactory
{
public:

  static IGhostValueSetterPtr makeExactGhostValueSetter();
  static IGhostValueSetterPtr makeMassFlowGhostValueSetter(ElemT rhoPReciprocal = ElemT{});
};

} // namespace kae
