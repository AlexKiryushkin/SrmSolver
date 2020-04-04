#pragma once

#pragma warning(push, 0)
#include <boost/geometry.hpp>
#pragma warning(pop)

namespace kae {

template <class GpuGridT>
EBoundaryCondition SrmShapeNozzleLess<GpuGridT>::getBoundaryCondition(ElemType x, ElemType y)
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
auto SrmShapeNozzleLess<GpuGridT>::getRadius(unsigned i, unsigned j) -> ElemType
{
  return j * GpuGridT::hy - yBottom;
}

template <class GpuGridT>
SrmShapeNozzleLess<GpuGridT>::SrmShapeNozzleLess()
  : m_distances{ GpuGridT::nx * GpuGridT::ny }
{
  namespace bg = boost::geometry;
  using Point2d      = bg::model::d2::point_xy<ElemType>;
  using Polygon2d    = bg::model::polygon<Point2d>;
  using Linestring2d = bg::model::linestring<Point2d>;

  Linestring2d linestring{
    { static_cast<ElemType>(0.0),          static_cast<ElemType>(0.0) },
    { static_cast<ElemType>(0.0),          Rk },
    { delta,                               Rk },
    { delta,                               static_cast<ElemType>(0.04) },
    { delta + static_cast<ElemType>(0.01), static_cast<ElemType>(0.031) },
    { static_cast<ElemType>(0.102),        static_cast<ElemType>(0.031) },
    { static_cast<ElemType>(0.105),        rkr },
    { static_cast<ElemType>(0.145),        rkr },
    { static_cast<ElemType>(0.691),        static_cast<ElemType>(0.026) },
    { static_cast<ElemType>(1.049),        static_cast<ElemType>(0.03) },
    { static_cast<ElemType>(1.087),        static_cast<ElemType>(0.03) },
    { static_cast<ElemType>(1.149),        static_cast<ElemType>(0.03) },
    { static_cast<ElemType>(1.194),        static_cast<ElemType>(0.044) },
    { static_cast<ElemType>(1.274),        static_cast<ElemType>(0.069) },
    { static_cast<ElemType>(1.274),        static_cast<ElemType>(0.0) },
    { static_cast<ElemType>(0.0),          static_cast<ElemType>(0.0) }
  };

  std::for_each(std::begin(linestring), std::end(linestring), [](auto & point)
  {
    bg::set<0>(point, bg::get<0>(point) + xLeft);
    bg::set<1>(point, bg::get<1>(point) + yBottom);
  });

  Polygon2d polygon;
  std::copy(std::begin(linestring), std::end(linestring), std::back_inserter(polygon.outer()));

  for (unsigned i = 0U; i < GpuGridT::nx; ++i)
  {
    const auto x = i * GpuGridT::hx;
    for (unsigned j = 0U; j < GpuGridT::ny; ++j)
    {
      const auto y = j * GpuGridT::hy;
      const Point2d point{ x, y };
      const auto distance = static_cast<ElemType>(bg::distance(point, linestring));
      const auto isInside = bg::covered_by(point, polygon);

      const auto index = j * GpuGridT::nx + i;
      m_distances[index] = isInside ? -std::fabs(distance) : std::fabs(distance);
    }
  }
}

template <class GpuGridT>
bool SrmShapeNozzleLess<GpuGridT>::shouldApplyScheme(unsigned i, unsigned j)
{
  return (j * GpuGridT::hy - yBottom >= static_cast<ElemType>(0.5) * rkr) &&
         (i * GpuGridT::hx - xLeft   >= static_cast<ElemType>(0.5) * delta);
}

template <class GpuGridT>
bool SrmShapeNozzleLess<GpuGridT>::isPointOnGrain(ElemType x, ElemType y)
{
  return (xRight - x >= static_cast<ElemType>(0.1) * GpuGridT::hx) &&
         (x - xStartPropellant >= static_cast<ElemType>(0.1) * GpuGridT::hx) &&
         (y - yBottom >= GpuGridT::hx) && 
         (y - yBottom <= Rk);
}

template <class GpuGridT>
auto SrmShapeNozzleLess<GpuGridT>::values() const -> const thrust::host_vector<ElemType> &
{
  return m_distances;
}

} // namespace kae
