#pragma once

#pragma warning(push, 0)
#include <Eigen/Core>
#pragma warning(pop)

namespace kae {

template <class ElemT>
struct GhostPointData
{
  constexpr static unsigned maxPoints{ 9U };

  Eigen::Matrix<ElemT, maxPoints, maxPoints> lhsMatrix;
  unsigned closestIndex;
};

} // namespace kae
