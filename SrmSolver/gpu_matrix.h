#pragma once

#include <type_traits>

#include <thrust/device_vector.h>

#include "cuda_runtime.h"

namespace kae
{

template <class GpuGridT, class T>
class GpuMatrix
{
public:

  using Type        = T;
  using GpuGridType = GpuGridT;

  GpuMatrix(Type value = {});
  template <class ShapeT, class = std::void_t<decltype(std::declval<ShapeT>()(1U, 2U))>>
  GpuMatrix(ShapeT shape);
  template <class ShapeT, class = std::void_t<decltype(std::declval<ShapeT>().values())>, class = void>
  GpuMatrix(ShapeT shape);

  const thrust::device_vector<Type> & values() const { return m_devValues; }
  thrust::device_vector<Type> & values() { return m_devValues; }

private:

  thrust::device_vector<Type> m_devValues;
};

template <class GpuGridT, class T>
thrust::device_ptr<const T> getConstDevicePtr(const GpuMatrix<GpuGridT, T> & matrix)
{
  return matrix.values().data();
}

template <class GpuGridT, class T>
thrust::device_ptr<const T> getDevicePtr(const GpuMatrix<GpuGridT, T> & matrix)
{
  return matrix.values().data();
}

template <class GpuGridT, class T>
thrust::device_ptr<T> getDevicePtr(GpuMatrix<GpuGridT, T> & matrix)
{
  return matrix.values().data();
}

template <class GpuGridT, class ShapeT, class ElemT>
__global__ void initializeGpuMatrix(thrust::device_ptr<ElemT> pValues, ShapeT shape)
{
  const unsigned i = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned j = threadIdx.y + blockDim.y * blockIdx.y;
  if ((i >= GpuGridT::nx) || (j >= GpuGridT::ny))
  {
    return;
  }

  pValues[j * GpuGridT::nx + i] = shape(i, j);
}

template <class GpuGridT, class T>
GpuMatrix<GpuGridT, T>::GpuMatrix(T value)
  : m_devValues(GpuGridT::n, value)
{
}

template <class GpuGridT, class T>
template <class ShapeT, class>
GpuMatrix<GpuGridT, T>::GpuMatrix(ShapeT shape)
  : m_devValues(GpuGridT::n)
{
  initializeGpuMatrix<GpuGridT><<<GpuGridT::gridSize, GpuGridT::blockSize>>>(m_devValues.data(), shape);
  cudaDeviceSynchronize();
}

template <class GpuGridT, class T>
template <class ShapeT, class, class>
GpuMatrix<GpuGridT, T>::GpuMatrix(ShapeT shape)
  : m_devValues(GpuGridT::n)
{
  thrust::copy(std::begin(shape.values()), std::end(shape.values()), std::begin(m_devValues));
}

} // namespace kae
