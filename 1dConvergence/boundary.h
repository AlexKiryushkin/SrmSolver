#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "gas_state.h"
#include "grid.h"
#include "types.h"

namespace kae {

class IBoundary
{
public:
  virtual ~IBoundary() = default;

  virtual std::size_t getStartIdx(const IGrid & grid) const = 0;
  virtual std::size_t getEndIdx(const IGrid& grid) const = 0;

  virtual ElemT getXBoundaryLeft(ElemT t, ElemT dt, unsigned rkStep) const = 0;
  virtual ElemT getXBoundaryRight(ElemT t, ElemT dt, unsigned rkStep) const = 0;

  virtual ElemT getXBoundaryLeftAnalytical(ElemT t) const = 0;
  virtual ElemT getXBoundaryRightAnalytical(ElemT t) const = 0;

  virtual void updateBoundaries(const std::vector<GasState> & gasValues, ElemT t, ElemT dt, unsigned rkStep) = 0;
};
using IBoundaryPtr = std::unique_ptr<IBoundary>;

class BoundaryFactory
{
public:

  static IBoundaryPtr makeStationaryBoundary(ElemT xBoundaryLeft, ElemT xBoundaryRight);
  static IBoundaryPtr makeLeftMovingBoundary(ElemT xBoundaryLeft, ElemT xBoundaryRight);

};

} // namespace kae
