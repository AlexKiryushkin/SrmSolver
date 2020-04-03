#pragma once

#include <cstdint>
#include <ratio>

namespace kae {

namespace detail {

template <class T, class ElemT>
struct ToFloat;

template <class T, class ElemT = float>
constexpr ElemT ToFloatV = ToFloat<T, ElemT>::value;

template <std::intmax_t Num, std::intmax_t Denom, class ElemT>
struct ToFloat<std::ratio<Num, Denom>, ElemT>
{
  constexpr static ElemT value = static_cast<ElemT>(Num) / static_cast<ElemT>(Denom);
};

} // namespace detail

} // namespace kae
