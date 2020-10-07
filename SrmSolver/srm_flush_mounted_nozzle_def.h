#pragma once

namespace kae {


template <class GpuGridT>
EBoundaryCondition SrmFlushMountedNozzle<GpuGridT>::getBoundaryCondition(ElemType x, ElemType y)
{
  if (std::fabs(x - xRight) < static_cast<ElemType>(0.1) * GpuGridT::hx)
  {
    return EBoundaryCondition::ePressureOutlet;
  }

  if (isPointOnGrain(x, y))
  {
    return EBoundaryCondition::eMassFlowInlet;
  }

  return EBoundaryCondition::eWall;
}

template <class GpuGridT>
auto SrmFlushMountedNozzle<GpuGridT>::getRadius(unsigned i, unsigned j) -> ElemType
{
  return j * GpuGridT::hy - yBottom;
}

template <class GpuGridT>
SrmFlushMountedNozzle<GpuGridT>::SrmFlushMountedNozzle()
  : m_distances{ GpuGridT::nx * GpuGridT::ny },
    m_linestring{}
{
  namespace bg = boost::geometry;

  for (unsigned i{}; i < nPoints; ++i)
  {
    bg::append(m_linestring, Point2d{ points[i][0U],  points[i][1U] });
  }

  std::for_each(std::begin(m_linestring), std::end(m_linestring), [](auto& point)
  {
    bg::set<0>(point, bg::get<0>(point) + xLeft);
    bg::set<1>(point, bg::get<1>(point) + yBottom);
  });

  Polygon2d polygon;
  std::copy(std::begin(m_linestring), std::end(m_linestring), std::back_inserter(polygon.outer()));

  for (unsigned i = 0U; i < GpuGridT::nx; ++i)
  {
    const auto x = i * GpuGridT::hx;
    for (unsigned j = 0U; j < GpuGridT::ny; ++j)
    {
      const auto y = j * GpuGridT::hy;
      const Point2d point{ x, y };
      const auto distance = static_cast<ElemType>(bg::distance(point, m_linestring));
      const auto isInside = bg::covered_by(point, polygon);

      const auto index = j * GpuGridT::nx + i;
      m_distances[index] = isInside ? -std::fabs(distance) : std::fabs(distance);
    }
  }
}

template <class GpuGridT>
constexpr auto SrmFlushMountedNozzle<GpuGridT>::getInitialSBurn() -> ElemType
{
  ElemType initialSBurn{};
  initialSBurn += initialSBurnPart<1U>();
  initialSBurn += initialSBurnPart<2U>();
  initialSBurn += initialSBurnPart<3U>();
  initialSBurn += initialSBurnPart<4U>();
  initialSBurn += initialSBurnPart<5U>();
  initialSBurn += initialSBurnPart<6U>();
  initialSBurn += initialSBurnPart<7U>();
  initialSBurn += initialSBurnPart<8U>();
  initialSBurn += initialSBurnPart<9U>();
  initialSBurn += initialSBurnPart<10U>();
  initialSBurn += initialSBurnPart<11U>();
  initialSBurn += initialSBurnPart<12U>();
  initialSBurn += initialSBurnPart<13U>();
  initialSBurn += initialSBurnPart<14U>();
  initialSBurn += initialSBurnPart<15U>();
  initialSBurn += initialSBurnPart<16U>();
  initialSBurn += initialSBurnPart<17U>();
  initialSBurn += initialSBurnPart<18U>();
  initialSBurn += initialSBurnPart<19U>();
  initialSBurn += initialSBurnPart<20U>();
  initialSBurn += initialSBurnPart<21U>();
  initialSBurn += initialSBurnPart<22U>();
  initialSBurn += initialSBurnPart<23U>();
  initialSBurn += initialSBurnPart<24U>();
  initialSBurn += initialSBurnPart<25U>();
  initialSBurn += initialSBurnPart<26U>();
  initialSBurn += initialSBurnPart<27U>();
  initialSBurn += initialSBurnPart<28U>();
  initialSBurn += initialSBurnPart<29U>();
  initialSBurn += initialSBurnPart<30U>();
  initialSBurn += initialSBurnPart<31U>();
  initialSBurn += initialSBurnPart<32U>();
  return initialSBurn;
}

template <class GpuGridT>
constexpr auto SrmFlushMountedNozzle<GpuGridT>::getFCritical() -> ElemType
{
  return static_cast<ElemType>(M_PI) * rkr * rkr;
}

template <class GpuGridT>
HOST_DEVICE auto SrmFlushMountedNozzle<GpuGridT>::isChamber(ElemType x, ElemType y) -> ElemType
{
  return (xEndPropellant - x >= static_cast<ElemType>(0.1) * GpuGridT::hx) &&
         (x - xLeft >= static_cast<ElemType>(0.1) * GpuGridT::hx);
}

template <class GpuGridT>
HOST_DEVICE auto SrmFlushMountedNozzle<GpuGridT>::isBurningSurface(ElemType x, ElemType y) -> ElemType
{
  constexpr auto threshold = static_cast<ElemType>(0.1) * GpuGridT::hx;
  const auto yIsLower = y - yBottom < yStartPropellant - threshold;
  const auto xIsLefter = x - xLeft <= 0;
  if (yIsLower || xIsLefter)
  {
    return false;
  }

  if (x - xLeft <= rRound)
  {
    const auto dxSqr = sqr(x - xLeft - rRound);
    const auto dySqr = sqr(y - yBottom - yEndPropellant + rRound);
    return (y - yBottom <= yEndPropellant - rRound) || (dxSqr + dySqr < rRound * rRound);
  }

  if (x - xLeft > rRound && x - xLeft < xEndPropellant - rRound)
  {
    return y - yBottom < yEndPropellant;
  }

  if (x - xLeft >= xEndPropellant - rRound && x - xLeft < xEndPropellant)
  {
    const auto dxSqr = sqr(x - xLeft - xEndPropellant + rRound);
    const auto dySqr = sqr(y - yBottom - yEndPropellant + rRound);
    const auto isOnRoundedCorner = dxSqr + dySqr < rRound * rRound;
    if (x - xLeft < xStartNozzle - 2 * GpuGridT::hx)
    {
      return (y - yBottom <= yEndPropellant - rRound) || isOnRoundedCorner;
    }

    return ((y - yBottom <= yEndPropellant - rRound) || isOnRoundedCorner) && 
            (y - yBottom > yMinNozzlePropellant + 2 * GpuGridT::hx);
  }

  return false;
}

template <class GpuGridT>
bool SrmFlushMountedNozzle<GpuGridT>::shouldApplyScheme(unsigned i, unsigned j)
{
  return (j * GpuGridT::hy - yBottom >= static_cast<ElemType>(0.5) * rkr) &&
         (i * GpuGridT::hx - xLeft <= xEndPropellant + 20 * GpuGridT::hx);
}

template <class GpuGridT>
bool SrmFlushMountedNozzle<GpuGridT>::isPointOnGrain(ElemType x, ElemType y)
{
  return isBurningSurface(x, y);
}

template <class GpuGridT>
auto SrmFlushMountedNozzle<GpuGridT>::values() const -> const thrust::host_vector<ElemType>&
{
  return m_distances;
}

} // namespace kae

