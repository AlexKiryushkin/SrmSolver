#pragma once

namespace kae {

template <class Derived>
class WrapperBase
{
protected:

  template <class T>
  T cast() const noexcept
  {
    T matrix;
    for (unsigned i{}; i < T::rows; ++i)
    {
      for (unsigned j{}; j < T::cols; ++j)
      {
        matrix(i, j) = self()(i, j);
      }
    }
    return matrix;
  }
private:
  Derived& self()
  {
    return *static_cast<Derived*>(this);
  }
  const Derived& self() const
  {
    return *static_cast<const Derived*>(this);
  }
};

} // namespace kae
