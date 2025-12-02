#pragma once

#pragma warning(push, 0)
#include <boost/geometry.hpp>
#pragma warning(pop)

#include "cuda_includes.h"

#include "boundary_condition.h"

namespace kae {

template <class GpuGridT>
class SrmShapeNozzleLess
{
public:

  using ElemType = typename GpuGridT::ElemType;

  SrmShapeNozzleLess(unsigned nx, unsigned ny, ElemType hx, ElemType hy);

  HOST_DEVICE static bool shouldApplyScheme(unsigned i, unsigned j);

  HOST_DEVICE static bool isPointOnGrain(ElemType x, ElemType y, ElemType h);

  HOST_DEVICE static EBoundaryCondition getBoundaryCondition(ElemType x, ElemType y, ElemType h);

  HOST_DEVICE static ElemType getRadius(ElemType x, ElemType y);

  constexpr HOST_DEVICE static ElemType getInitialSBurn();

  constexpr HOST_DEVICE static ElemType getFCritical();

  constexpr HOST_DEVICE static ElemType getOutletCoordinate() { return xRight; }

  HOST_DEVICE static ElemType isChamber(ElemType x, ElemType y, ElemType h);

  HOST_DEVICE static ElemType isBurningSurface(ElemType x, ElemType y, ElemType h);
  
  const thrust::host_vector<ElemType> & values() const;

private:

  using Point2d      = boost::geometry::model::d2::point_xy<ElemType>;
  using Polygon2d    = boost::geometry::model::polygon<Point2d>;
  using Linestring2d = boost::geometry::model::linestring<Point2d>;

  constexpr static unsigned offsetPoints = 32;

  constexpr static ElemType xLeft            = (offsetPoints + static_cast<ElemType>(0.5)) * GpuGridT::hx;
  constexpr static ElemType delta            = static_cast<ElemType>(0.01);
  constexpr static ElemType xStartPropellant = xLeft + delta;
  constexpr static ElemType xRight           = xLeft + static_cast<ElemType>(1.274);
  constexpr static ElemType yBottom          = (offsetPoints + static_cast<ElemType>(0.5)) * GpuGridT::hy;
  constexpr static ElemType Rk               = static_cast<ElemType>(0.1);
  constexpr static ElemType rkr              = static_cast<ElemType>(0.0245);

  constexpr static unsigned nPoints              = 16U;
  constexpr static unsigned dim                  = 2U;
  constexpr static unsigned startPropellantPoint = 3U;
  constexpr static unsigned endPropellantPoint   = 13U;
  constexpr static ElemType points[nPoints][dim] = {
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

  constexpr static unsigned nEndPoints{ 12U };
  constexpr static ElemType endPoints[nEndPoints][dim] = {
    { static_cast<ElemType>(0.01),  static_cast<ElemType>(0.081)  },
    { static_cast<ElemType>(0.02),  static_cast<ElemType>(0.081)  },
    { static_cast<ElemType>(0.055), static_cast<ElemType>(0.0922) },
    { static_cast<ElemType>(0.102), static_cast<ElemType>(0.0922) },
    { static_cast<ElemType>(0.105), static_cast<ElemType>(0.0922) },
    { static_cast<ElemType>(0.145), static_cast<ElemType>(0.0922) },
    { static_cast<ElemType>(0.691), static_cast<ElemType>(0.0922) },
    { static_cast<ElemType>(1.045), static_cast<ElemType>(0.0922) },
    { static_cast<ElemType>(1.087), static_cast<ElemType>(0.081)  },
    { static_cast<ElemType>(1.139), static_cast<ElemType>(0.081)  },
    { static_cast<ElemType>(1.184), static_cast<ElemType>(0.0669) },
    { static_cast<ElemType>(1.274), static_cast<ElemType>(0.0872) }
  };

  template <unsigned idx>
  constexpr HOST_DEVICE static ElemType initialSBurnPart()
  {
    constexpr auto dx = points[idx + 1U][0U] - points[idx][0U];
    constexpr auto dy = points[idx + 1U][1U] - points[idx][1U];
    constexpr auto distance = gcem::sqrt(dx * dx + dy * dy);
    return static_cast<ElemType>(M_PI) * (points[idx + 1U][1U] + points[idx][1U]) * distance;
  }

  template <unsigned idx>
  constexpr HOST_DEVICE static bool isBurningPart(ElemType x, ElemType y)
  {
    constexpr auto xStart = endPoints[idx][0U];
    constexpr auto yStart = endPoints[idx][1U];

    constexpr auto threshold = static_cast<ElemType>(0.1) * GpuGridT::hx;
    constexpr auto xEnd = endPoints[idx + 1U][0U];
    constexpr auto yEnd = endPoints[idx + 1U][1U];
    if (((x < xStart)             && (idx != 0U)) || 
        ((x < xStart - threshold) && (idx == 0U)) ||
        ((x > xEnd)               && (idx != nEndPoints - 2U)) ||
        ((x > xEnd   + threshold) && (idx == nEndPoints - 2U)))
    {
      return false;
    }

    const auto yPropellant = yStart + (x - xStart) / (xEnd - xStart) * (yEnd - yStart);
    return y <= yPropellant && y >= rkr - GpuGridT::hx;
  }
private:

  thrust::host_vector<ElemType> m_distances;
  Linestring2d m_linestring;
};

} // namespace kae

#include "srm_shape_nozzle_less_def.h"
