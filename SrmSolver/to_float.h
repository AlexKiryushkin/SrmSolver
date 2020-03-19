#pragma once

#include <cstdint>
#include <ratio>

namespace kae {

namespace detail {

template <class T>
struct ToFloat;

template <class T>
constexpr float ToFloatV = ToFloat<T>::value;

template <std::intmax_t Num, std::intmax_t Denom>
struct ToFloat<std::ratio<Num, Denom>>
{
  constexpr static float value = static_cast<float>(Num) / static_cast<float>(Denom);
};

} // namespace detail

} // namespace kae
