#pragma once

#include "std_includes.h"
#include "cuda_includes.h"

namespace kae
{

template <class T>
class GpuMatrix
{
public:

  using Type        = T;

  GpuMatrix(unsigned nx, unsigned ny, Type value = {});
  GpuMatrix(unsigned nx, unsigned ny, thrust::host_vector<T> values);
  template <class ShapeT, class = std::void_t<decltype(std::declval<ShapeT>().values())>, class = void>
  GpuMatrix(unsigned nx, unsigned ny, ShapeT shape);

  const thrust::device_vector<Type> & values() const { return m_devValues; }
  thrust::device_vector<Type> & values() { return m_devValues; }

  unsigned nx() const { return m_nx; }
  unsigned ny() const { return m_ny; }

private:
    unsigned m_nx;
    unsigned m_ny;
  thrust::device_vector<Type> m_devValues;
};

template <class T>
thrust::device_ptr<const T> getConstDevicePtr(const GpuMatrix<T> & matrix)
{
  return matrix.values().data();
}

template <class T>
thrust::device_ptr<const T> getDevicePtr(const GpuMatrix<T> & matrix)
{
  return matrix.values().data();
}

template <class T>
thrust::device_ptr<T> getDevicePtr(GpuMatrix<T> & matrix)
{
  return matrix.values().data();
}

template <class T>
GpuMatrix<T>::GpuMatrix(unsigned nx, unsigned ny, T value)
    : m_nx{ nx }, m_ny{ ny }, m_devValues(m_nx * m_ny, value)
{
}

template <class T>
GpuMatrix<T>::GpuMatrix(unsigned nx, unsigned ny, thrust::host_vector<T> values)
    : m_nx{ nx }, m_ny{ ny }, m_devValues(m_nx* m_ny)
{
    thrust::copy(std::begin(values), std::end(values), std::begin(m_devValues));
}

template <class T>
template <class ShapeT, class, class>
GpuMatrix<T>::GpuMatrix(unsigned nx, unsigned ny, ShapeT shape)
  : m_nx{ nx }, m_ny{ ny }, m_devValues(m_nx* m_ny)
{
  thrust::copy(std::begin(shape.values()), std::end(shape.values()), std::begin(m_devValues));
}

} // namespace kae
