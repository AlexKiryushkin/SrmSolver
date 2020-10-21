#pragma once

#include <memory>
#include <cstddef>

#include "types.h"

namespace kae {

class IGrid
{
public:

  virtual ~IGrid() = default;  

  virtual std::size_t getNPoints()        const = 0;
  virtual ElemT       getH()              const = 0;
  virtual ElemT       getX(std::size_t i) const = 0;
  virtual ElemT       getXLeft()          const = 0;
  virtual ElemT       getXRight()         const = 0;
};
using IGridPtr = std::unique_ptr<IGrid>;

class GridFactory
{
public:
  static IGridPtr makeSimpleGrid(std::size_t nPoints, ElemT xLeft, ElemT xRight);
};

} // namespace kae
