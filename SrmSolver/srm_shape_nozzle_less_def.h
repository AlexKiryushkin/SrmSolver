#pragma once

#include <boost/geometry.hpp>

namespace kae {

template <class GpuGridT>
EBoundaryCondition SrmShapeNozzleLess<GpuGridT>::getBoundaryCondition(float x, float y)
{
  if (std::fabs(x - xRight) < 0.1f * GpuGridT::hx)
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
float SrmShapeNozzleLess<GpuGridT>::getRadius(unsigned i, unsigned j)
{
  return j * GpuGridT::hy - yBottom;
}

template <class GpuGridT>
SrmShapeNozzleLess<GpuGridT>::SrmShapeNozzleLess()
  : m_distances{ GpuGridT::nx * GpuGridT::ny }
{
  namespace bg = boost::geometry;
  using Point2d = bg::model::point<float, 2u, boost::geometry::cs::cartesian>;
  using Polygon2d = bg::model::polygon<Point2d>;
  using Linestring2d = bg::model::linestring<Point2d>;

  Linestring2d linestring{
    { 0.0f, 0.0f },   { 0.0f, Rk },  { delta, Rk },   { delta, 0.04f },
    { delta + 0.01f, 0.031f },   { 0.102f, 0.031f },  { 0.105f, rkr }, { 0.145f, rkr },
    { 0.691f, 0.026f },  { 1.049f, 0.03f }, { 1.087f, 0.03f }, { 1.149f, 0.03f }, { 1.194f, 0.044f },
    { 1.274f, 0.069f }, { 1.274f, 0.0f }, { 0.0f, 0.0f }
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
    float x = i * GpuGridT::hx;
    for (unsigned j = 0U; j < GpuGridT::ny; ++j)
    {
      float y = j * GpuGridT::hy;
      Point2d point{ x, y };
      auto distance = bg::distance(point, linestring);
      auto isInside = bg::covered_by(point, polygon);

      int index = j * GpuGridT::nx + i;
      m_distances[index] = isInside ? -std::fabs(distance) : std::fabs(distance);
    }
  }
}

template <class GpuGridT>
bool SrmShapeNozzleLess<GpuGridT>::shouldApplyScheme(unsigned i, unsigned j)
{
  return (j * GpuGridT::hy - yBottom >= 0.5f * rkr) && (i * GpuGridT::hx - xLeft >= 0.5f * delta);
}

template <class GpuGridT>
bool SrmShapeNozzleLess<GpuGridT>::isPointOnGrain(float x, float y)
{
  return (xRight - x >= 0.1f * GpuGridT::hx) && 
         (x - xStartPropellant >= 0.1f * GpuGridT::hx) &&
         (y - yBottom >= GpuGridT::hx) && 
         (y - yBottom) <= Rk;
}

template <class GpuGridT>
const thrust::host_vector<float> & SrmShapeNozzleLess<GpuGridT>::values() const
{
  return m_distances;
}

} // namespace kae
