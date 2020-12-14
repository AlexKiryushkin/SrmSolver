
#include "quartic_solve.h"

#include <cmath>
#include <numbers>

#include "level_set_derivatives.h"

namespace {

template <class FloatT>
constexpr auto epsilon = std::numeric_limits<FloatT>::epsilon();

template <class FloatT>
FloatT cubicSolveImpl(FloatT a, FloatT b, FloatT c, FloatT d, FloatT h)
{
  constexpr auto pi = std::numbers::pi_v<FloatT>;
  constexpr auto nan = std::numeric_limits<FloatT>::quiet_NaN();
  const auto eps = h * h * h * h;
  constexpr auto zero = static_cast<FloatT>(0.0);

  if (std::fabs(d) < eps)
  {
    return static_cast<FloatT>(0.0);
  }
  else if (std::fabs(a) < eps)
  {
    if (std::fabs(b) < eps)
    {
      return - d / c;
    }

    const FloatT disc = std::sqrt(c * c - 4 * b * d);
    const FloatT root1 = (-c + disc) / 2 / b;
    const FloatT root2 = (-c - disc) / 2 / b;
    if ((root1 >= zero) && (root1 <= h + eps))
    {
      return root1;
    }
    else if ((root2 >= zero) && (root2 <= h + eps))
    {
      return root2;
    }
    else
    {
      return nan;
    }
  }

  b /= a;
  c /= a;
  d /= a;

  FloatT q = (3 * c - (b * b)) / 9;
  const FloatT r = (-(27 * d) + b * (9 * c - 2 * (b * b))) / 54;
  const FloatT disc = q * q * q + r * r;
  const FloatT term1 = (b / 3);

  if (disc > 0)
  {
    FloatT s = r + std::sqrt(disc);
    s = ((s < 0) ? -std::cbrt(-s) : std::cbrt(s));
    FloatT t = r - std::sqrt(disc);
    t = ((t < 0) ? -std::cbrt(-t) : std::cbrt(t));
    return - term1 + s + t;
  }

  if (disc == 0)
  {
    FloatT r13 = ((r < 0) ? -std::cbrt(-r) : std::cbrt(r));
    FloatT root1 = -term1 + 2 * r13;
    FloatT root2 = -(r13 + term1);
    if ((root1 >= zero) && (root1 <= h + eps))
    {
      return root1;
    }
    else if ((root2 >= zero) && (root2 <= h + eps))
    {
      return root2;
    }
    else
    {
      return nan;
    }
  }

  q = -q;
  FloatT dum1 = q * q * q;
  dum1 = std::acos(r / std::sqrt(dum1));
  FloatT r13 = 2 * std::sqrt(q);
  FloatT root1 = -term1 + r13 * std::cos(dum1 / 3);
  FloatT root2 = -term1 + r13 * std::cos((dum1 + 2 * pi) / 3);
  FloatT root3 = -term1 + r13 * std::cos((dum1 + 4 * pi) / 3);
  if ((root1 >= zero) && (root1 <= h + eps))
  {
    return root1;
  }
  else if ((root2 >= zero) && (root2 <= h + eps))
  {
    return root2;
  }
  else if ((root3 >= zero) && (root3 <= h + eps))
  {
    return root3;
  }
  else
  {
    return nan;
  }
}

} // namespace

namespace kae {

template <class FloatT>
FloatT quarticSolve(const std::vector<FloatT>& data, std::size_t idx, FloatT h, std::size_t step)
{
  const auto eps = epsilon<FloatT> / h;
  const auto order = h * h * h * h;

  const FloatT a1 = (data.at(idx + step) - 2 * data.at(idx) + 2 * data.at(idx - 2U * step) - data.at(idx - 3U * step));
  const FloatT a2 = (data.at(idx + 2U * step) - 2 * data.at(idx + step) + 2 * data.at(idx - step) - data.at(idx - 2U * step));
  const FloatT a3 = (data.at(idx + 3U * step) - 2 * data.at(idx + 2U * step) + 2 * data.at(idx) - data.at(idx - step));

  FloatT a = FloatT{};// minmod(a1, minmod(a2, a3));
  a = std::fabs(a) < eps ? 0 : a / 12 / h / h / h;

  const FloatT b1 = data.at(idx + step) - 2 * data.at(idx) + data.at(idx - step);
  const FloatT b2 = data.at(idx + 2U * step) - 2 * data.at(idx + step) + data.at(idx);
  FloatT b = minmod(b1, b2);
  b = std::fabs(b) < eps ? 0 : b / 2 / h / h;

  const FloatT c1 = (data.at(idx + step) - data.at(idx - step)) / 2;
  const FloatT c2 = data.at(idx + step) - data.at(idx);
  FloatT c = std::fabs(c2 - c1) > h * std::sqrt(h) ?  c2 : c1;
  c = std::fabs(c) < eps ? 0 : c / h;

  FloatT d = data.at(idx);

  if (std::fabs(d) < order)
  {
    return static_cast<FloatT>(0.0);
  }
  else
  {
    return cubicSolveImpl(a, b, c, d, h);
  }
}

template float quarticSolve<float>(const std::vector<float>& data, std::size_t idx, float h, std::size_t step);
template double quarticSolve<double>(const std::vector<double>& data, std::size_t idx, double h, std::size_t step);

} // namespace kae
