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
  template <class ShapeT, class = std::void_t<decltype(std::declval<ShapeT>()(1U, 2U))>>
  GpuMatrix(unsigned nx, unsigned ny, ShapeT shape);
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

template <class ShapeT, class ElemT>
__global__ void initializeGpuMatrix(thrust::device_ptr<ElemT> pValues, ShapeT shape, unsigned nx, unsigned ny)
{
  const unsigned i = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned j = threadIdx.y + blockDim.y * blockIdx.y;
  if ((i >= nx) || (j >= ny))
  {
    return;
  }

  pValues[j * nx + i] = shape(i, j);
}

template <class T>
GpuMatrix<T>::GpuMatrix(unsigned nx, unsigned ny, T value)
    : m_nx{ nx }, m_ny{ ny }, m_devValues(m_nx * m_ny, value)
{
}

template <class T>
template <class ShapeT, class>
GpuMatrix<T>::GpuMatrix(unsigned nx, unsigned ny, ShapeT shape)
    :m_nx{ nx }, m_ny{ ny }, m_devValues(m_nx* m_ny)
{
    const dim3 blockSize{ 32U, 8U };
    const dim3 gridSize{ (m_nx + blockSize.x - 1) / blockSize.x, (m_ny + blockSize.y - 1) / blockSize.y };
    initializeGpuMatrix << <gridSize, blockSize >> > (m_devValues.data(), shape, m_nx, m_ny);
    cudaDeviceSynchronize();
}

template <class T>
template <class ShapeT, class, class>
GpuMatrix<T>::GpuMatrix(unsigned nx, unsigned ny, ShapeT shape)
  : m_nx{ nx }, m_ny{ ny }, m_devValues(m_nx* m_ny)
{
  thrust::copy(std::begin(shape.values()), std::end(shape.values()), std::begin(m_devValues));
}

} // namespace kae
