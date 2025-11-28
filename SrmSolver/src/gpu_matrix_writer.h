#pragma once

#include "std_includes.h"

#include "gas_state.h"
#include "gpu_matrix.h"

namespace kae {

template <class ElemT, class = std::enable_if_t<std::is_floating_point<ElemT>::value>>
void writeMatrixToFile(const GpuMatrix<ElemT> & matrix, ElemT hx, ElemT hy, const std::string & path)
{
  std::ofstream fOut(path);
  assert(!!fOut);

  std::vector<ElemT> hostValues(matrix.nx() * matrix.ny());
  auto && deviceValues = matrix.values();
  thrust::copy(std::begin(deviceValues), std::end(deviceValues), std::begin(hostValues));

  for (unsigned i = 0; i < matrix.nx(); ++i)
  {
    for (unsigned j = 0; j < matrix.ny(); ++j)
    {
      ElemT x = i * hx;
      ElemT y = j * hy;

      fOut << x << ';' << y << ';' << hostValues[j * matrix.nx() + i] << '\n';
    }
  }
}

template <class ElemT>
void writeMatrixToFile(const GpuMatrix<GasState<ElemT>> & matrix, const GasParameters<ElemT> &gasParameters, ElemT hx, ElemT hy,
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

  std::vector<GasState<ElemT>> hostValues(matrix.nx() * matrix.ny());
  auto && deviceValues = matrix.values();
  thrust::copy(std::begin(deviceValues), std::end(deviceValues), std::begin(hostValues));

  for (unsigned i = 0; i < matrix.nx(); ++i)
  {
    for (unsigned j = 0; j < matrix.ny(); ++j)
    {
      ElemT x = i * hx;
      ElemT y = j * hy;

      auto&& gasState = hostValues[j * matrix.nx() + i];
      pFile    << x << ';' << y << ';' << gasState.p                 << '\n';
      uxFile   << x << ';' << y << ';' << gasState.ux                << '\n';
      uyFile   << x << ';' << y << ';' << gasState.uy                << '\n';
      machFile << x << ';' << y << ';' << Mach::get(gasState, gasParameters)        << '\n';
      tFile    << x << ';' << y << ';' << Temperature::get(gasState, gasParameters) << '\n';
    }
  }
}

} // namespace kae
