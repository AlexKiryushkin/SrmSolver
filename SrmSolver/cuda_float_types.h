#pragma once

#include "std_includes.h"
#include "cuda_includes.h"

namespace kae {

template <std::size_t nElems, class ElemT>
struct CudaFloat;

template <std::size_t nElems, class ElemT>
using CudaFloatT = typename CudaFloat<nElems, ElemT>::type;

template <>
struct CudaFloat<2U, float>
{
  using type = float2;
};

template <>
struct CudaFloat<2U, double>
{
  using type = double2;
};

template <>
struct CudaFloat<3U, float>
{
  using type = float3;
};

template <>
struct CudaFloat<3U, double>
{
  using type = double3;
};

template <>
struct CudaFloat<4U, float>
{
  using type = float4;
};

template <>
struct CudaFloat<4U, double>
{
  using type = double4;
};

} // namespace kae
