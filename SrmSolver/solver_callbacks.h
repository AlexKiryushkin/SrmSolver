#pragma once

#include "std_includes.h"

#include "cuda_float_types.h"
#include "filesystem.h"
#include "gas_state.h"
#include "gnu_plot_wrapper.h"
#include "gpu_matrix.h"
#include "gpu_matrix_writer.h"
#include "solver_reduction_functions.h"

namespace kae {

class WriteToFolderCallback
{
public:
  WriteToFolderCallback(std::wstring folderPath)
    : m_gnuPlotTemperature("\"C:\\Program Files\\gnuplot\\bin\\gnuplot.exe\""),
      m_folderPath(std::move(folderPath))
  {
    kae::remove_all(m_folderPath);
    kae::create_directories(m_folderPath);
    kae::current_path(m_folderPath);
  }

  template <class GpuGridType,
            class GasStateType,
            class ShapeT,
            class ElemT = typename GasStateType::ElemType>
  void operator()(const GpuMatrix<GpuGridType, GasStateType> & gasValues,
                  const GpuMatrix<GpuGridType, ElemT> & currPhi,
                  unsigned i, ElemT t, CudaFloatT<4U, ElemT> maxDerivatives, ShapeT)
  {
    static thread_local std::vector<std::pair<ElemT, ElemT>> meanPressureValues;
    const auto meanPressure =
      detail::getCalculatedBoriPressure<GpuGridType, ShapeT>(gasValues.values(), currPhi.values());
    meanPressureValues.emplace_back(t, meanPressure);
    const auto writeToFile = [this](std::vector<std::pair<ElemT, ElemT>> meanPressureValues,
      GpuMatrix<GpuGridType, GasStateType> gasValues,
      GpuMatrix<GpuGridType, ElemT> currPhi,
      unsigned i, ElemT t, CudaFloatT<4U, ElemT> maxDerivatives)
    {
      std::cout << "Iteration: " << i << ". Time: " << t << '\n';
      std::cout << "Mean chamber pressure: " << meanPressureValues.back().second << '\n';
      std::cout << "Max derivatives: d(rho)/dt = " << maxDerivatives.x
        << "; d(rho * ux)/dt = " << maxDerivatives.y
        << "; d(rho * uy)/dt = " << maxDerivatives.z
        << "; d(rho * E)/dt = " << maxDerivatives.w << "\n\n";
      const std::wstring tString = std::to_wstring(t);
      const std::wstring newCurrentPath = kae::append(m_folderPath, tString);
      kae::create_directories(newCurrentPath);
      kae::current_path(newCurrentPath);

      std::ofstream meanPressureFile{ "mean_pressure_values.dat" };
      for (const auto& elem : meanPressureValues)
      {
        meanPressureFile << elem.first << ';' << elem.second << '\n';
      }
      writeMatrixToFile(gasValues, "p.dat", "ux.dat", "uy.dat", "mach.dat", "T.dat");
      writeMatrixToFile(currPhi, "sgd.dat");
    };
    std::thread writeAsync{ writeToFile, meanPressureValues, gasValues, currPhi, i, t, maxDerivatives };
    writeAsync.detach();
  }

  template <class GpuGridT, class GasStateT, class ElemT = typename GasStateT::ElemType>
  void operator()(const GpuMatrix<GpuGridT, GasStateT> & gasValues)
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
    auto drawTemperature = [this](thrust::host_vector<GasStateT> hostGasStateValues)
    {
      std::vector<std::vector<ElemT>> gridTemperatureValues;
      for (unsigned j{ 0U }; j < GpuGridT::ny; ++j)
      {
        const auto offset = j * GpuGridT::nx;
        std::vector<ElemT> rowTemperatureValues(GpuGridT::nx);
        std::transform(std::next(std::begin(hostGasStateValues), offset),
          std::next(std::begin(hostGasStateValues), offset + GpuGridT::nx),
          std::begin(rowTemperatureValues),
          Temperature{});
        gridTemperatureValues.push_back(std::move(rowTemperatureValues));
      }
      m_gnuPlotTemperature.display2dPlot(gridTemperatureValues);
    };
    auto && values = gasValues.values();
    thrust::host_vector<GasStateT> hostGasStateValues( values );
    future = std::async(std::launch::async, drawTemperature, std::move(hostGasStateValues));
  }

private:
  GnuPlotWrapper m_gnuPlotTemperature;
  std::wstring m_folderPath;
};

} // namespace kae
