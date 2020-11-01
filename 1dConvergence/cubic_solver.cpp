
#include "cubic_solver.h"

#include <cmath>

ElemT kae::cubicSolve(const std::vector<ElemT> & data, std::size_t idx, ElemT h)
{
  constexpr auto pi = static_cast<ElemT>(M_PI);
  constexpr auto inf = std::numeric_limits<ElemT>::infinity();
  ElemT a = (data.at(idx + 2U) - 3 * data.at(idx + 1U) + 3 * data.at(idx) - data.at(idx - 1U)) / 6 / h / h / h;
  ElemT b = (data.at(idx + 1U) - 2 * data.at(idx) + data.at(idx - 1U)) / 2 / h / h;
  ElemT c = (-data.at(idx + 2U) + 6 * data.at(idx + 1U) - 3 * data.at(idx) - 2 * data.at(idx - 1U)) / 6 / h;
  ElemT d = data.at(idx);

  if ( std::fabs(d) < std::numeric_limits<ElemT>::min() )
  {
    return idx * h;
  }
  else if (std::fabs(a) < std::numeric_limits<ElemT>::min())
  {
    if (std::fabs(b) < std::numeric_limits<ElemT>::min())
    {
      return -d / b;
    }

    ElemT disc = std::sqrt(c * c - 4 * b * d);
    ElemT root1 = (-c + disc) / 2 / b;
    ElemT root2 = (-c - disc) / 2 / b;
    return ( root1 <= h ) ? root1 : ( ( root2 <= h ) ? root2 : inf );
  }

  b /= a;
  c /= a;
  d /= a;

  ElemT q = (3 * c - (b * b)) / 9;
  ElemT r = (-(27 * d) + b * (9 * c - 2 * (b * b))) / 54;
  ElemT disc = q * q * q + r * r;
  ElemT term1 = (b / 3);

  if (disc > 0) 
  {
    ElemT s = r + std::sqrt(disc);
    s = ( ( s < 0 ) ? -std::cbrt( -s ) : std::cbrt( s ) );
    ElemT t = r - std::sqrt(disc);
    t = ( ( t < 0 ) ? -std::cbrt( -t ) : std::cbrt( t ) );
    return -term1 + s + t;
  }

  if (disc == 0) 
  {
    ElemT r13 = ( ( r < 0 ) ? -std::cbrt( -r ) : std::cbrt( r ) );
    ElemT root1 = -term1 + 2 * r13;
    ElemT root2 = -(r13 + term1);
    return ( root1 <= h ) ? root1 : ( ( root2 <= h ) ? root2 : inf );
  }

  q = -q;
  ElemT dum1 = q * q * q;
  dum1 = std::acos(r / std::sqrt(dum1));
  ElemT r13 = 2 * std::sqrt(q);
  ElemT root1 = -term1 + r13 * std::cos(dum1 / 3);
  ElemT root2 = -term1 + r13 * std::cos((dum1 + 2 * pi) / 3);
  ElemT root3 = -term1 + r13 * std::cos((dum1 + 4 * pi) / 3);
  return ( root1 <= h ) ? root1 : ( ( root2 <= h ) ? root2 : ( ( root3 <= h ) ? root3 : inf ) );
}
