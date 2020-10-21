
#include "grid.h"

namespace kae {

class SimpleGrid : public IGrid
{
public:

  explicit SimpleGrid(std::size_t nPoints, ElemT xLeft = static_cast<ElemT>(-1), ElemT xRight = static_cast<ElemT>(1))
    : m_nPoints{ nPoints }, m_xLeft{ xLeft }, m_xRight{ xRight } {}

  std::size_t getNPoints()        const override { return m_nPoints; }
  ElemT       getH()              const override { return (m_xRight - m_xLeft) / static_cast<ElemT>(m_nPoints - 1); }
  ElemT       getX(std::size_t i) const override { return m_xLeft + i * getH(); }
  ElemT       getXLeft()          const override { return m_xLeft; }
  ElemT       getXRight()         const override { return m_xRight; }

private:
  const std::size_t m_nPoints;
  const ElemT m_xLeft;
  const ElemT m_xRight;
};

IGridPtr GridFactory::makeSimpleGrid(std::size_t nPoints, ElemT xLeft, ElemT xRight)
{
  return std::make_unique<SimpleGrid>(nPoints, xLeft, xRight);
}

}
