#pragma once

namespace kae {


template <class GpuGridT>
EBoundaryCondition SrmShapeNozzleLess<GpuGridT>::getBoundaryCondition(ElemType x, ElemType y, ElemType h)
{
  if (std::fabs(x - xRight) < static_cast<ElemType>(0.1) * h)
  {
    return EBoundaryCondition::ePressureOutlet;
  }

  if (isPointOnGrain(x, y, h))
  {
    return EBoundaryCondition::eMassFlowInlet;
  }

  return EBoundaryCondition::eWall;
}

template <class GpuGridT>
HOST_DEVICE auto SrmShapeNozzleLess<GpuGridT>::getRadius(ElemType x, ElemType y) -> ElemType
{
  return y - yBottom;
}

template <class GpuGridT>
SrmShapeNozzleLess<GpuGridT>::SrmShapeNozzleLess(unsigned nx, unsigned ny, ElemType hx, ElemType hy)
  : m_distances( nx * ny ),
    m_linestring{
      { points[0U][0U],  points[0U][1U] },
      { points[1U][0U],  points[1U][1U] },
      { points[2U][0U],  points[2U][1U] },
      { points[3U][0U],  points[3U][1U] },
      { points[4U][0U],  points[4U][1U] },
      { points[5U][0U],  points[5U][1U] },
      { points[6U][0U],  points[6U][1U] },
      { points[7U][0U],  points[7U][1U] },
      { points[8U][0U],  points[8U][1U] },
      { points[9U][0U],  points[9U][1U] },
      { points[10U][0U], points[10U][1U] },
      { points[11U][0U], points[11U][1U] },
      { points[12U][0U], points[12U][1U] },
      { points[13U][0U], points[13U][1U] },
      { points[14U][0U], points[14U][1U] },
      { points[15U][0U], points[15U][1U] }
    }
{
  namespace bg = boost::geometry;

  std::for_each(std::begin(m_linestring), std::end(m_linestring), [](auto & point)
  {
    bg::set<0>(point, bg::get<0>(point) + xLeft);
    bg::set<1>(point, bg::get<1>(point) + yBottom);
  });

  Polygon2d polygon;
  std::copy(std::begin(m_linestring), std::end(m_linestring), std::back_inserter(polygon.outer()));

  for (unsigned i = 0U; i < nx; ++i)
  {
    const auto x = i * hx;
    for (unsigned j = 0U; j < ny; ++j)
    {
      const auto y = j * hy;
      const Point2d point{ x, y };
      const auto distance = static_cast<ElemType>(bg::distance(point, m_linestring));
      const auto isInside = bg::covered_by(point, polygon);

      const auto index = j * nx + i;
      m_distances[index] = isInside ? -std::fabs(distance) : std::fabs(distance);
    }
  }
}

template <class GpuGridT>
constexpr auto SrmShapeNozzleLess<GpuGridT>::getInitialSBurn() -> ElemType
{
  ElemType initialSBurn{};
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
  return initialSBurn;
}

template <class GpuGridT>
constexpr auto SrmShapeNozzleLess<GpuGridT>::getFCritical() -> ElemType
{
  return static_cast<ElemType>(M_PI) * rkr * rkr;
}

template <class GpuGridT>
HOST_DEVICE auto SrmShapeNozzleLess<GpuGridT>::isChamber(ElemType x, ElemType y, ElemType h) -> ElemType
{
  return (xRight - x >= static_cast<ElemType>(0.1) * h) &&
         (x - xLeft >= static_cast<ElemType>(0.1) * h);
}

template <class GpuGridT>
HOST_DEVICE auto SrmShapeNozzleLess<GpuGridT>::isBurningSurface(ElemType x, ElemType y, ElemType h) -> ElemType
{
  return (xRight - x >= static_cast<ElemType>(0.1) * h) &&
         (x - xStartPropellant >= static_cast<ElemType>(0.1) * h) &&
         (isBurningPart<0U>(x - xLeft, y - yBottom) || isBurningPart<1U>(x - xLeft, y - yBottom)
       || isBurningPart<2U>(x - xLeft, y - yBottom) || isBurningPart<3U>(x - xLeft, y - yBottom)
       || isBurningPart<4U>(x - xLeft, y - yBottom) || isBurningPart<5U>(x - xLeft, y - yBottom)
       || isBurningPart<6U>(x - xLeft, y - yBottom) || isBurningPart<7U>(x - xLeft, y - yBottom)
       || isBurningPart<8U>(x - xLeft, y - yBottom) || isBurningPart<9U>(x - xLeft, y - yBottom)
       || isBurningPart<10U>(x - xLeft, y - yBottom));
}

template <class GpuGridT>
bool SrmShapeNozzleLess<GpuGridT>::shouldApplyScheme(unsigned i, unsigned j)
{
  return (j * GpuGridT::hy - yBottom >= static_cast<ElemType>(0.5) * rkr) &&
         (i * GpuGridT::hx - xLeft   >= static_cast<ElemType>(0.5) * delta);
}

template <class GpuGridT>
bool SrmShapeNozzleLess<GpuGridT>::isPointOnGrain(ElemType x, ElemType y, ElemType h)
{
  return isBurningSurface(x, y, h);
}

template <class GpuGridT>
auto SrmShapeNozzleLess<GpuGridT>::values() const -> const thrust::host_vector<ElemType> &
{
  return m_distances;
}

} // namespace kae
