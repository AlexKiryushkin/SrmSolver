#pragma once

namespace kae {

namespace detail {

struct EmptyCallback
{
  template <class... T>
  void operator()(T&& ...) {}
};

} // namespace detail

} // namespace kae
