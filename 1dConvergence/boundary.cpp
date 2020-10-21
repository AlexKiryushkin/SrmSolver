
#include "boundary.h"

namespace kae {

class StationaryBoundary : public IBoundary
{
public:

  StationaryBoundary(ElemT xBoundaryLeft, ElemT xBoundaryRight)
    : m_xBoundaryLeft{ xBoundaryLeft }, m_xBoundaryRight{ xBoundaryRight } {}

  std::size_t getStartIdx(const IGrid& grid) const override
  {
    return static_cast<std::size_t>(std::floor((m_xBoundaryLeft - grid.getXLeft()) / grid.getH()));
  }
  std::size_t getEndIdx(const IGrid& grid) const override
  {
    return static_cast<std::size_t>( std::floor( ( m_xBoundaryRight - grid.getXLeft() ) / grid.getH() ) + 1U);
  }

  ElemT getXBoundaryLeft() const override { return m_xBoundaryLeft; }
  ElemT getXBoundaryRight() const override { return m_xBoundaryRight; }

  void updateBoundaries(const std::vector<GasState>&, ElemT, ElemT, unsigned) override {}

private:
  ElemT m_xBoundaryLeft;
  ElemT m_xBoundaryRight;
};

IBoundaryPtr BoundaryFactory::makeStationaryBoundary(ElemT xBoundaryLeft, ElemT xBoundaryRight)
{
  return std::make_unique<StationaryBoundary>(xBoundaryLeft, xBoundaryRight);
}

} // namespace kae
