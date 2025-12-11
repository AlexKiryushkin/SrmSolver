#pragma once

#include "std_includes.h"

namespace kae {

namespace detail {

template <class T, class ElemT>
struct ToFloat;

template <class T, class ElemT>
constexpr ElemT ToFloatV = ToFloat<T, ElemT>::value;

template <std::intmax_t Num, std::intmax_t Denom, class ElemT>
struct ToFloat<std::ratio<Num, Denom>, ElemT>
{
  constexpr static ElemT value = static_cast<ElemT>(Num) / static_cast<ElemT>(Denom);
};

} // namespace detail

} // namespace kae
