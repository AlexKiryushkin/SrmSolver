#pragma once

#pragma warning(push, 0)
#include <boost/geometry.hpp>
#pragma warning(pop)

#include "cuda_includes.h"

#include "boundary_condition.h"

namespace kae {

template <class GpuGridT>
class SrmDualThrust
{
public:

  using ElemType = typename GpuGridT::ElemType;

  SrmDualThrust();

  HOST_DEVICE static bool shouldApplyScheme(unsigned i, unsigned j);

  HOST_DEVICE static bool isPointOnGrain(ElemType x, ElemType y);

  HOST_DEVICE static EBoundaryCondition getBoundaryCondition(ElemType x, ElemType y);

  HOST_DEVICE static ElemType getRadius(unsigned i, unsigned j);

  HOST_DEVICE static ElemType getRadius(ElemType x, ElemType y);

  constexpr HOST_DEVICE static ElemType getInitialSBurn();

  constexpr HOST_DEVICE static ElemType getFCritical();

  constexpr HOST_DEVICE static ElemType getOutletCoordinate() { return xRight; }

  HOST_DEVICE static ElemType isChamber(ElemType x, ElemType y);

  HOST_DEVICE static ElemType isBurningSurface(ElemType x, ElemType y);

  const thrust::host_vector<ElemType>& values() const;

private:

  using Point2d = boost::geometry::model::d2::point_xy<ElemType>;
  using Polygon2d = boost::geometry::model::polygon<Point2d>;
  using Linestring2d = boost::geometry::model::linestring<Point2d>;

  constexpr static unsigned offsetPoints = 32;

  constexpr static ElemType xLeft = (offsetPoints + static_cast<ElemType>(0.5)) * GpuGridT::hx;
  constexpr static ElemType xRight = xLeft + static_cast<ElemType>(0.32086);
  constexpr static ElemType propellantRight = xLeft + static_cast<ElemType>(0.213);
  constexpr static ElemType chamberRight = xLeft + static_cast<ElemType>(0.271);
  constexpr static ElemType yBottom = (offsetPoints + static_cast<ElemType>(0.5)) * GpuGridT::hy;
  constexpr static ElemType Rk = static_cast<ElemType>(0.054);
  constexpr static ElemType rkr = static_cast<ElemType>(0.0042);

  constexpr static unsigned nPoints = 38U;
  constexpr static unsigned dim = 2U;
  constexpr static unsigned startPropellantPoint = 0U;
  constexpr static unsigned endPropellantPoint = 21U;
  constexpr static ElemType points[nPoints][dim] = {
      { static_cast<ElemType>(0.15000), static_cast<ElemType>(0.00000) },
      { static_cast<ElemType>(0.15002), static_cast<ElemType>(0.00035) },
      { static_cast<ElemType>(0.15006), static_cast<ElemType>(0.00069) },
      { static_cast<ElemType>(0.15014), static_cast<ElemType>(0.00104) },
      { static_cast<ElemType>(0.15024), static_cast<ElemType>(0.00137) },
      { static_cast<ElemType>(0.15037), static_cast<ElemType>(0.00169) },
      { static_cast<ElemType>(0.15054), static_cast<ElemType>(0.00200) },
      { static_cast<ElemType>(0.15072), static_cast<ElemType>(0.00229) },
      { static_cast<ElemType>(0.15094), static_cast<ElemType>(0.00257) },
      { static_cast<ElemType>(0.15117), static_cast<ElemType>(0.00283) },
      { static_cast<ElemType>(0.15143), static_cast<ElemType>(0.00306) },
      { static_cast<ElemType>(0.15171), static_cast<ElemType>(0.00328) },
      { static_cast<ElemType>(0.15200), static_cast<ElemType>(0.00346) },
      { static_cast<ElemType>(0.15231), static_cast<ElemType>(0.00363) },
      { static_cast<ElemType>(0.15263), static_cast<ElemType>(0.00376) },
      { static_cast<ElemType>(0.15296), static_cast<ElemType>(0.00386) },
      { static_cast<ElemType>(0.15331), static_cast<ElemType>(0.00394) },
      { static_cast<ElemType>(0.15365), static_cast<ElemType>(0.00398) },
      { static_cast<ElemType>(0.15400), static_cast<ElemType>(0.00400) },
      { static_cast<ElemType>(0.21300), static_cast<ElemType>(0.00400) },
      { static_cast<ElemType>(0.21300), static_cast<ElemType>(0.05400) },
      { static_cast<ElemType>(0.17100), static_cast<ElemType>(0.05400) },
      { static_cast<ElemType>(0.17100), static_cast<ElemType>(0.05900) },
      { static_cast<ElemType>(0.27100), static_cast<ElemType>(0.05900) },
      { static_cast<ElemType>(0.27100), static_cast<ElemType>(0.01098) },
      { static_cast<ElemType>(0.30113), static_cast<ElemType>(0.00440) },
      { static_cast<ElemType>(0.30152), static_cast<ElemType>(0.00435) },
      { static_cast<ElemType>(0.30204), static_cast<ElemType>(0.00427) },
      { static_cast<ElemType>(0.30256), static_cast<ElemType>(0.00422) },
      { static_cast<ElemType>(0.30309), static_cast<ElemType>(0.00420) },
      { static_cast<ElemType>(0.30361), static_cast<ElemType>(0.00421) },
      { static_cast<ElemType>(0.30413), static_cast<ElemType>(0.00424) },
      { static_cast<ElemType>(0.30465), static_cast<ElemType>(0.00430) },
      { static_cast<ElemType>(0.30517), static_cast<ElemType>(0.00438) },
      { static_cast<ElemType>(0.30539), static_cast<ElemType>(0.00443) },
      { static_cast<ElemType>(0.32086), static_cast<ElemType>(0.00779) },
      { static_cast<ElemType>(0.32086), static_cast<ElemType>(0.00000) },
      { static_cast<ElemType>(0.15000), static_cast<ElemType>(0.00000) }
  };

  template <unsigned idx>
  constexpr HOST_DEVICE static ElemType initialSBurnPart()
  {
    constexpr auto dx = points[idx + 1U][0U] - points[idx][0U];
    constexpr auto dy = points[idx + 1U][1U] - points[idx][1U];
    constexpr auto distance = gcem::sqrt(dx * dx + dy * dy);
    return static_cast<ElemType>(M_PI) * (points[idx + 1U][1U] + points[idx][1U]) * distance;
  }

private:

  thrust::host_vector<ElemType> m_distances;
  Linestring2d m_linestring;
};

} // namespace kae

#include "srm_dual_thrust_def.h"

