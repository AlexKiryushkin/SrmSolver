#pragma once

#include "std_includes.h"

#include "cuda_float_types.h"
#include "filesystem.h"
#include "gas_state.h"
#include "gnu_plot_wrapper.h"
#include "gpu_matrix.h"
#include "gpu_matrix_writer.h"

namespace kae {

class WriteToFolderCallback
{
public:
  WriteToFolderCallback(std::wstring folderPath)
    : m_gnuPlotWrapper("\"C:\\Program Files\\gnuplot\\bin\\gnuplot.exe\""),
      m_folderPath(std::move(folderPath))
  {
    kae::remove_all(m_folderPath);
    kae::create_directories(m_folderPath);
    kae::current_path(m_folderPath);
  }

  template <class GpuGridType, class GasStateType, class ElemT = typename GasStateType::ElemType>
  void operator()(const GpuMatrix<GpuGridType, GasStateType> & gasValues,
                  const GpuMatrix<GpuGridType, ElemT> & currPhi,
                  unsigned i, ElemT t, CudaFloatT<4U, ElemT> maxDerivatives)
  {
    std::cout << "Iteration: " << i << ". Time: " << t << '\n';
    std::cout << "Max derivatives: d(rho)/dt = " << maxDerivatives.x
              << "; d(rho * ux)/dt = "           << maxDerivatives.y
              << "; d(rho * uy)/dt = "           << maxDerivatives.z
              << "; d(rho * E)/dt = "            << maxDerivatives.w << "\n\n";
    const std::wstring tString = std::to_wstring(t);
    const std::wstring newCurrentPath = kae::append(m_folderPath, tString);
    kae::create_directories(newCurrentPath);
    kae::current_path(newCurrentPath);

    writeMatrixToFile(gasValues, "p.dat", "ux.dat", "uy.dat", "mach.dat", "T.dat");
    writeMatrixToFile(currPhi, "sgd.dat");
  }

  template <class GpuGridType, class GasStateType, class ElemT = typename GasStateType::ElemType>
  void operator()(const GpuMatrix<GpuGridType, GasStateType> & gasValues)
  {
    static std::future<void> future;
    if (future.valid())
    {
      const auto status = future.wait_for(std::chrono::milliseconds(10));
      if (status != std::future_status::ready)
      {
        return;
      }
    }
    auto drawTemperature = [this](thrust::host_vector<GasStateType> hostGasStateValues)
    {
      std::vector<std::vector<ElemT>> gridTemperatureValues;
      for (unsigned j{ 0U }; j < GpuGridType::ny; ++j)
      {
        const auto offset = j * GpuGridType::nx;
        std::vector<ElemT> rowTemperatureValues(GpuGridType::nx);
        std::transform(std::next(std::begin(hostGasStateValues), offset),
          std::next(std::begin(hostGasStateValues), offset + GpuGridType::nx),
          std::begin(rowTemperatureValues),
          Temperature{});
        gridTemperatureValues.push_back(std::move(rowTemperatureValues));
      }
      m_gnuPlotWrapper.display2dPlot(gridTemperatureValues);
    };
    auto && values = gasValues.values();
    thrust::host_vector<GasStateType> hostGasStateValues( values );
    future = std::async(std::launch::async, drawTemperature, std::move(hostGasStateValues));
  }

private:
  GnuPlotWrapper m_gnuPlotWrapper;
  std::wstring m_folderPath;
};

} // namespace kae
