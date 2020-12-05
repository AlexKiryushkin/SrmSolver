
#include "quartic_solve.h"

#include <cmath>
#include <complex>
#include <numbers>

namespace {

template <class FloatT>
constexpr auto epsilon = std::numeric_limits<FloatT>::epsilon();

template <class FloatT>
FloatT cubicSolveImpl(FloatT a, FloatT b, FloatT c, FloatT d, std::size_t idx, FloatT h)
{
  constexpr auto pi = std::numbers::pi_v<FloatT>;
  constexpr auto inf = std::numeric_limits<FloatT>::infinity();
  const auto eps = h * h * h * h * h;
  constexpr auto zero = static_cast<FloatT>(0.0);

  if (std::fabs(d) < eps)
  {
    return idx * h;
  }
  else if (std::fabs(a) < eps)
  {
    if (std::fabs(b) < eps)
    {
      return idx * h - d / c;
    }

    const FloatT disc = std::sqrt(c * c - 4 * b * d);
    const FloatT root1 = (-c + disc) / 2 / b;
    const FloatT root2 = (-c - disc) / 2 / b;
    if ((root1 >= zero) && (root1 <= h))
    {
      return idx * h + root1;
    }
    else if ((root2 >= zero) && (root2 <= h))
    {
      return idx * h + root2;
    }
    else
    {
      return inf;
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
    return idx * h - term1 + s + t;
  }

  if (disc == 0)
  {
    FloatT r13 = ((r < 0) ? -std::cbrt(-r) : std::cbrt(r));
    FloatT root1 = -term1 + 2 * r13;
    FloatT root2 = -(r13 + term1);
    if ((root1 >= zero) && (root1 <= h))
    {
      return idx * h + root1;
    }
    else if ((root2 >= zero) && (root2 <= h))
    {
      return idx * h + root2;
    }
    else
    {
      return inf;
    }
  }

  q = -q;
  FloatT dum1 = q * q * q;
  dum1 = std::acos(r / std::sqrt(dum1));
  FloatT r13 = 2 * std::sqrt(q);
  FloatT root1 = -term1 + r13 * std::cos(dum1 / 3);
  FloatT root2 = -term1 + r13 * std::cos((dum1 + 2 * pi) / 3);
  FloatT root3 = -term1 + r13 * std::cos((dum1 + 4 * pi) / 3);
  if ((root1 >= zero) && (root1 <= h))
  {
    return idx * h + root1;
  }
  else if ((root2 >= zero) && (root2 <= h))
  {
    return idx * h + root2;
  }
  else if ((root3 >= zero) && (root3 <= h))
  {
    return idx * h + root3;
  }
  else
  {
    return inf;
  }
}

template <class FloatT>
FloatT quarticSolveImpl(FloatT a, FloatT b1, FloatT c1, FloatT d1, FloatT e1, std::size_t idx, FloatT h)
{
  constexpr auto inf = std::numeric_limits<FloatT>::infinity();
  constexpr auto zero = static_cast<FloatT>(0.0);

  const FloatT b = b1 / a;
  const FloatT c = c1 / a;
  const FloatT d = d1 / a;
  const FloatT e = e1 / a;

  const FloatT q1 = c * c - 3 * b * d + 12 * e;
  const FloatT q2 = 2 * c * c * c - 9 * b * c * d + 27 * d * d + 27 * b * b * e - 72 * c * e;
  const FloatT q3 = 8 * b * c - 16 * d - 2 * b * b * b;
  const FloatT q4 = 3 * b * b - 8 * c;

  const FloatT q5 = std::cbrt(q2 / 2 + std::sqrt(q2 * q2 / 4 - q1 * q1 * q1));
  const FloatT q6 = (q1 / q5 + q5) / 3;
  const FloatT q7 = 2 * std::sqrt(q4 / 12 + q6);

  const FloatT firstDiscriminant = 4 * q4 / 6 - 4 * q6 - q3 / q7;
  if (firstDiscriminant >= zero)
  {
    const FloatT root1 = (-b - q7 - std::sqrt(firstDiscriminant)) / 4;
    const FloatT root2 = (-b - q7 + std::sqrt(firstDiscriminant)) / 4;
    if ((root1 >= zero) && (root1 <= h))
    {
      return idx * h + root1;
    }
    else if ((root2 >= zero) && (root2 <= h))
    {
      return idx * h + root2;
    }
  }

  const FloatT secondDiscriminant = 4 * q4 / 6 - 4 * q6 + q3 / q7;
  if (secondDiscriminant >= zero)
  {
    const FloatT root1 = (-b + q7 - std::sqrt(secondDiscriminant)) / 4;
    const FloatT root2 = (-b + q7 + std::sqrt(secondDiscriminant)) / 4;
    if ((root1 >= zero) && (root1 <= h))
    {
      return idx * h + root1;
    }
    else if ((root2 >= zero) && (root2 <= h))
    {
      return idx * h + root2;
    }
  }

  return inf;
}

} // namespace

namespace kae {

template <class FloatT>
FloatT quarticSolve(const std::vector<FloatT>& data, std::size_t idx, FloatT h)
{
  const auto eps = epsilon<FloatT> / h;
  const auto order = h * h * h * h * h;

  FloatT a = (data.at(idx + 2U) - 4 * data.at(idx + 1U) + 6 * data.at(idx) - 4 * data.at(idx - 1U) + data.at(idx - 2U));
  a = std::fabs(a) < eps ? 0 : a / 24 / h / h / h / h;

  FloatT b = (data.at(idx + 2U) - 2 * data.at(idx + 1U) + 2 * data.at(idx - 1U) - data.at(idx - 2U));
  b = std::fabs(b) < eps ? 0 : b / 12 / h / h / h;

  FloatT c = (-data.at(idx + 2U) + 16 * data.at(idx + 1U) - 30 * data.at(idx) + 16 * data.at(idx - 1U) - data.at(idx - 2U));
  c = std::fabs(c) < eps ? 0 : c / 24 / h / h;

  FloatT d = (-data.at(idx + 2U) + 8 * data.at(idx + 1U) - 8 * data.at(idx - 1U) + data.at(idx - 2U));
  d = std::fabs(d) < eps ? 0 : d / 12 / h;

  FloatT e = data.at(idx);

  if (std::fabs(e) < order)
  {
    return idx * h;
  }
  else if (std::fabs(a) < order)
  {
    return cubicSolveImpl(b, c, d, e, idx, h);
  }
  else
  {
    return quarticSolveImpl(a, b, c, d, e, idx, h);
  }
}

template float quarticSolve<float>(const std::vector<float>& data, std::size_t idx, float h);
template double quarticSolve<double>(const std::vector<double>& data, std::size_t idx, double h);

} // namespace kae
