#pragma once

#include "std_includes.h"
#include "cuda_includes.h"

namespace kae {

namespace detail {

template <class ElemT>
__host__ __device__ ElemT phi(ElemT x)
{
  constexpr auto pi = static_cast<ElemT>(M_PI);
  return std::sqrt(pi / 9) * std::exp(-pi * pi * x * x / 9);
}

template <class ElemT>
__host__ __device__ ElemT deltaDiracFunction(ElemT x, ElemT h, ElemT m = 2)
{
  const auto epsilon = m * h;
  return 1 / epsilon * phi(x / epsilon);
}

} // namespace detail

} // namespace kae
