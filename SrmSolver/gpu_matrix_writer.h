#pragma once

#include <cassert>
#include <fstream>
#include <string>

#include "gas_state.h"
#include "gpu_matrix.h"

namespace kae {

template <class GpuGridT>
void writeMatrixToFile(const GpuMatrix<GpuGridT, float> & matrix, const std::string & path)
{
  std::ofstream fOut(path);
  assert(!!fOut);

  std::vector<float> hostValues(GpuGridT::n);
  auto && deviceValues = matrix.values();
  thrust::copy(std::begin(deviceValues), std::end(deviceValues), std::begin(hostValues));

  for (unsigned i = 0; i < GpuGridT::nx; ++i)
    for (unsigned j = 0; j < GpuGridT::ny; ++j)
    {
      float x = i * GpuGridT::hx;
      float y = j * GpuGridT::hy;

      fOut << x << ';' << y << ';' << hostValues[j*GpuGridT::nx + i] << '\n';
    }
}

template <class GpuGridT, class KappaT, class CpT>
void writeMatrixToFile(const GpuMatrix<GpuGridT, GasState<KappaT, CpT>> & matrix,
                       const std::string & pPath,
                       const std::string & uxPath,
                       const std::string & uyPath,
                       const std::string & machPath,
                       const std::string & tPath)
{
  std::ofstream pFile(pPath);
  std::ofstream uxFile(uxPath);
  std::ofstream uyFile(uyPath);
  std::ofstream machFile(machPath);
  std::ofstream tFile(tPath);
  assert(pFile && uxFile && uyFile && machFile && tFile);

  std::vector<GasState<KappaT, CpT>> hostValues(GpuGridT::n);
  auto && deviceValues = matrix.values();
  thrust::copy(std::begin(deviceValues), std::end(deviceValues), std::begin(hostValues));

  for (unsigned i = 0; i < GpuGridT::nx; ++i)
    for (unsigned j = 0; j < GpuGridT::ny; ++j)
    {
      float x = i * GpuGridT::hx;
      float y = j * GpuGridT::hy;

      auto && gasState = hostValues[j*GpuGridT::nx + i];
      pFile << x << ';' << y << ';' << gasState.p << '\n';
      uxFile << x << ';' << y << ';' << gasState.ux << '\n';
      uyFile << x << ';' << y << ';' << gasState.uy << '\n';
      machFile << x << ';' << y << ';' <<  Mach::get(gasState) << '\n';
      tFile << x << ';' << y << ';' << Temperature::get(gasState) << '\n';
    }
}

} // namespace kae
