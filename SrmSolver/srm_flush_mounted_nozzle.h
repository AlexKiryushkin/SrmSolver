#pragma once

#pragma warning(push, 0)
#include <boost/geometry.hpp>
#pragma warning(pop)

#include "cuda_includes.h"

#include "boundary_condition.h"

namespace kae {

template <class GpuGridT>
class SrmFlushMountedNozzle
{
public:

  using ElemType = typename GpuGridT::ElemType;

  SrmFlushMountedNozzle();

  HOST_DEVICE static bool shouldApplyScheme(unsigned i, unsigned j);

  HOST_DEVICE static bool isPointOnGrain(ElemType x, ElemType y);

  HOST_DEVICE static EBoundaryCondition getBoundaryCondition(ElemType x, ElemType y);

  HOST_DEVICE static ElemType getRadius(unsigned i, unsigned j);

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
  constexpr static ElemType xRight = xLeft + static_cast<ElemType>(1.77);
  constexpr static ElemType yBottom = (offsetPoints + static_cast<ElemType>(0.5)) * GpuGridT::hy;

  constexpr static ElemType xEndPropellant = static_cast<ElemType>(1.64);
  constexpr static ElemType xStartNozzle = static_cast<ElemType>(1.5008);
  constexpr static ElemType yMinNozzlePropellant = static_cast<ElemType>(0.1936);
  constexpr static ElemType yStartPropellant = static_cast<ElemType>(0.152);
  constexpr static ElemType yEndPropellant = static_cast<ElemType>(0.55);
  constexpr static ElemType rRound = static_cast<ElemType>(0.25);
  constexpr static ElemType rkr = static_cast<ElemType>(0.1049);
  constexpr static ElemType R0 = static_cast<ElemType>(0.152);

  constexpr static unsigned dim = 2U;
  constexpr static unsigned startPropellantPoint = 1U;
  constexpr static unsigned endPropellantPoint = 33U;
  constexpr static ElemType points[][dim] = {
    { static_cast<ElemType>(0),      static_cast<ElemType>(0) },
    { static_cast<ElemType>(0),      static_cast<ElemType>(0.152) },
    { static_cast<ElemType>(0.325),  static_cast<ElemType>(0.152) },

    { static_cast<ElemType>(0.2192), static_cast<ElemType>(0.4428) },
    { static_cast<ElemType>(0.2183), static_cast<ElemType>(0.4461) },
    { static_cast<ElemType>(0.218),  static_cast<ElemType>(0.4496) },
    { static_cast<ElemType>(0.2183), static_cast<ElemType>(0.4531) },
    { static_cast<ElemType>(0.2192), static_cast<ElemType>(0.4565) },
    { static_cast<ElemType>(0.2206), static_cast<ElemType>(0.4596) },
    { static_cast<ElemType>(0.2226), static_cast<ElemType>(0.4625) },
    { static_cast<ElemType>(0.2251), static_cast<ElemType>(0.4649) },
    { static_cast<ElemType>(0.228),  static_cast<ElemType>(0.4669) },
    { static_cast<ElemType>(0.2311), static_cast<ElemType>(0.4684) },
    { static_cast<ElemType>(0.2345), static_cast<ElemType>(0.4693) },
    { static_cast<ElemType>(0.2379), static_cast<ElemType>(0.4696) },

    { static_cast<ElemType>(0.2466), static_cast<ElemType>(0.4696) },
    { static_cast<ElemType>(0.2501), static_cast<ElemType>(0.4693) },
    { static_cast<ElemType>(0.2535), static_cast<ElemType>(0.4684) },
    { static_cast<ElemType>(0.2567), static_cast<ElemType>(0.4669) },
    { static_cast<ElemType>(0.2595), static_cast<ElemType>(0.4649) },
    { static_cast<ElemType>(0.262),  static_cast<ElemType>(0.4625) },
    { static_cast<ElemType>(0.264),  static_cast<ElemType>(0.4596) },
    { static_cast<ElemType>(0.2648), static_cast<ElemType>(0.4581) },

    { static_cast<ElemType>(0.394),  static_cast<ElemType>(0.1809) },
    { static_cast<ElemType>(0.3961), static_cast<ElemType>(0.177) },
    { static_cast<ElemType>(0.4011), static_cast<ElemType>(0.1699) },
    { static_cast<ElemType>(0.4072), static_cast<ElemType>(0.1637) },
    { static_cast<ElemType>(0.4144), static_cast<ElemType>(0.1587) },
    { static_cast<ElemType>(0.4223), static_cast<ElemType>(0.155) },
    { static_cast<ElemType>(0.4307), static_cast<ElemType>(0.1528) },
    { static_cast<ElemType>(0.4393), static_cast<ElemType>(0.152) },

    { static_cast<ElemType>(1.4274), static_cast<ElemType>(0.152) },
    { static_cast<ElemType>(1.5078), static_cast<ElemType>(0.2195) },
    { static_cast<ElemType>(1.6399), static_cast<ElemType>(0.2195) },

    { static_cast<ElemType>(1.6399), static_cast<ElemType>(0.1936) },
    { static_cast<ElemType>(1.6398), static_cast<ElemType>(0.1919) },
    { static_cast<ElemType>(1.6394), static_cast<ElemType>(0.1902) },
    { static_cast<ElemType>(1.6387), static_cast<ElemType>(0.1887) },
    { static_cast<ElemType>(1.6377), static_cast<ElemType>(0.1872) },
    { static_cast<ElemType>(1.6364), static_cast<ElemType>(0.186) },
    { static_cast<ElemType>(1.635),  static_cast<ElemType>(0.185) },
    { static_cast<ElemType>(1.6334), static_cast<ElemType>(0.1843) },
    { static_cast<ElemType>(1.6319), static_cast<ElemType>(0.1838) },

    { static_cast<ElemType>(1.5831), static_cast<ElemType>(0.1738) },

    { static_cast<ElemType>(1.5208), static_cast<ElemType>(0.1738) },
    { static_cast<ElemType>(1.5174), static_cast<ElemType>(0.1735) },
    { static_cast<ElemType>(1.514),  static_cast<ElemType>(0.1726) },
    { static_cast<ElemType>(1.5109), static_cast<ElemType>(0.1711) },
    { static_cast<ElemType>(1.508),  static_cast<ElemType>(0.1691) },
    { static_cast<ElemType>(1.5056), static_cast<ElemType>(0.1667) },
    { static_cast<ElemType>(1.5036), static_cast<ElemType>(0.1638) },
    { static_cast<ElemType>(1.5021), static_cast<ElemType>(0.1607) },
    { static_cast<ElemType>(1.5012), static_cast<ElemType>(0.1573) },
    { static_cast<ElemType>(1.5009), static_cast<ElemType>(0.1538) },
    { static_cast<ElemType>(1.5012), static_cast<ElemType>(0.1503) },
    { static_cast<ElemType>(1.5021), static_cast<ElemType>(0.147) },
    { static_cast<ElemType>(1.5036), static_cast<ElemType>(0.1438) },
    { static_cast<ElemType>(1.5055), static_cast<ElemType>(0.1409) },

    { static_cast<ElemType>(1.5241), static_cast<ElemType>(0.119) },
    { static_cast<ElemType>(1.529),  static_cast<ElemType>(0.1142)},
    { static_cast<ElemType>(1.5347), static_cast<ElemType>(0.1102)},
    { static_cast<ElemType>(1.541),  static_cast<ElemType>(0.1073)},
    { static_cast<ElemType>(1.5477), static_cast<ElemType>(0.1055)},
    { static_cast<ElemType>(1.5547), static_cast<ElemType>(0.1049)},
    { static_cast<ElemType>(1.5616), static_cast<ElemType>(0.1055)},
    { static_cast<ElemType>(1.5633), static_cast<ElemType>(0.1058) },

    { static_cast<ElemType>(1.77),   static_cast<ElemType>(0.152) },
    { static_cast<ElemType>(1.77),   static_cast<ElemType>(0) },
    { static_cast<ElemType>(0),   static_cast<ElemType>(0) }
  };
  constexpr static unsigned nPoints = sizeof(points) / dim / sizeof(ElemType);

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

#include "srm_flush_mounted_nozzle_def.h"
