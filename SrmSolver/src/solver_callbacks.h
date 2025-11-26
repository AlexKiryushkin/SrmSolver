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

namespace detail {

class ThreadWorker
{
public:

  ThreadWorker()
    : m_thread([this]()
      {
        while (m_bContinue)
        {
          std::unique_lock<std::mutex> locker(m_mutex);
          m_conditionVariable.wait(locker, [this]() { return m_bReady; });
          m_function();
          m_bReady = false;
        }
      })
  {}

  ~ThreadWorker()
  {
    m_bContinue = false;
    m_bReady = true;
    m_function = []() {};
    m_conditionVariable.notify_one();
  }

  template <class FunctorT>
  bool trySubmit(FunctorT && functor)
  {
    if (m_mutex.try_lock())
    {
      m_function = functor;
      m_bReady = true;
      m_mutex.unlock();
      m_conditionVariable.notify_one();
      return true;
    }

    return false;
  }

private:

  std::thread m_thread;
  std::function<void()> m_function;
  std::mutex m_mutex;
  std::condition_variable m_conditionVariable;
  bool m_bReady = false;
  bool m_bContinue = true;

};

} // namespace detail

template <class ElemT>
class WriteToFolderCallback
{
public:

  WriteToFolderCallback(std::wstring folderPath, 
                        std::string pathToGnuPlot = "\"C:\\Program Files\\gnuplot\\bin\\gnuplot.exe\"")
    : m_folderPath(std::move(folderPath)),
      m_gnuPlotTemperature(pathToGnuPlot)
  {
    kae::remove_all(m_folderPath);
    kae::create_directories(m_folderPath);
    kae::current_path(m_folderPath);
  }

  template <class GasStateT,
            class ShapeT>
  void operator()(const GpuMatrix<GasStateT> & gasValues,
                  const GpuMatrix<ElemT> & currPhi,
                  unsigned i, ElemT t, CudaFloat4T<ElemT> maxDerivatives, ElemT sBurn, ShapeT, unsigned nx, unsigned ny, ElemT hx, ElemT hy)
  {
    const auto meanPressure   = detail::getCalculatedBoriPressure<ShapeT>(gasValues.values(), currPhi.values(), nx, ny, hx, hy);
    const auto maxPressure    = detail::getMaxChamberPressure<ShapeT>(gasValues.values(), currPhi.values(), nx, ny, hx, hy);
    const auto thrustData     = detail::getMotorThrust<ShapeT>(gasValues.values(), currPhi.values(), nx, ny, hx, hy);
    const auto massFlowRate   = thrust::get<2U>(thrustData);
    const auto velocity       = thrust::get<1U>(thrustData) / thrust::get<0U>(thrustData);
    const auto thrust         = massFlowRate * velocity + thrust::get<3U>(thrustData);
    const auto specificThrust = thrust / massFlowRate;
    m_meanPressureValues.emplace_back(t, meanPressure, maxPressure, sBurn, thrust, specificThrust, velocity);

    const auto writeToFile = [this, hx, hy](std::vector<IntegralDataT> meanPressureValues,
      GpuMatrix<GasStateT> gasValues,
      GpuMatrix<ElemT> currPhi,
      unsigned i, ElemT t, CudaFloat4T<ElemT> maxDerivatives)
    {
      std::cout << "Iteration: " << i << ". Time: " << t << '\n';
      std::cout << "Mean chamber pressure: " << std::get<1U>(meanPressureValues.back()) << '\n';
      std::cout << "Max derivatives: d(rho)/dt = " << maxDerivatives.x
        << "; d(rho * ux)/dt = " << maxDerivatives.y
        << "; d(rho * uy)/dt = " << maxDerivatives.z
        << "; d(rho * E)/dt = " << maxDerivatives.w << "\n\n";

      const std::wstring tString = std::to_wstring(t);
      const std::wstring newCurrentPath = kae::append(m_folderPath, tString);
      kae::create_directories(newCurrentPath);
      kae::current_path(newCurrentPath);

      std::ofstream meanPressureFile{ "mean_pressure_values.dat" };
      meanPressureFile << "t;P_av;P_max;S;Thrust;specThrust;velocity\n";
      for (const auto& elem : meanPressureValues)
      {
        meanPressureFile << std::get<0U>(elem) << ';' << std::get<1U>(elem) << ';'
                         << std::get<2U>(elem) << ';' << std::get<3U>(elem) << ';'
                         << std::get<4U>(elem) << ';' << std::get<5U>(elem) << ';'
                         << std::get<6U>(elem) <<'\n';
      }
      writeMatrixToFile(currPhi, hx, hy, "sgd.dat");
      writeMatrixToFile(gasValues, hx, hy, "p.dat", "ux.dat", "uy.dat", "mach.dat", "T.dat");
    };
    std::thread writeAsync{ writeToFile, m_meanPressureValues, gasValues, currPhi, i, t, maxDerivatives };
    writeAsync.detach();
  }

  template <class GasStateT>
  void operator()(const GpuMatrix<GasStateT> & gasValues,
    const GpuMatrix<ElemT> & phiValues, ElemT hx, ElemT hy)
  {
    const auto func = [=]()
    {
      thrust::host_vector<GasStateT> hostGasStateValues(gasValues.values());
      thrust::host_vector<ElemT> hostPhiValues(phiValues.values());
      drawTemperature(std::move(hostGasStateValues), std::move(hostPhiValues), gasValues.nx(), gasValues.ny(), hx, hy);
    };
    m_threadWorker.trySubmit(func);
  }

private:

  using IntegralDataT = std::tuple<ElemT, ElemT, ElemT, ElemT, ElemT, ElemT, ElemT>;

  template <class GasStateT, class ElemT = typename GasStateT::ElemType>
  void drawTemperature(thrust::host_vector<GasStateT> hostGasStateValues,
                       thrust::host_vector<ElemT> hostPhiValues, unsigned nx, unsigned ny, ElemT hx, ElemT hy)
  {
    std::vector<std::vector<std::tuple<ElemT, ElemT, ElemT>>> gridTemperatureValues;
    for (unsigned j{ 0U }; j < ny; ++j)
    {
      std::vector<std::tuple<ElemT, ElemT, ElemT>> rowTemperatureValues(nx);
      for (unsigned i{}; i < nx; ++i)
      {
        const auto x = i * hx;
        const auto y = j * hy;
        rowTemperatureValues[i] = std::make_tuple(x, y, P::get(hostGasStateValues[j * nx + i]));
      };
      gridTemperatureValues.push_back(std::move(rowTemperatureValues));
    }
    m_gnuPlotTemperature.display2dPlot(gridTemperatureValues);
  }

private:
  std::wstring               m_folderPath;
  GnuPlotWrapper             m_gnuPlotTemperature;
  std::vector<IntegralDataT> m_meanPressureValues;
  detail::ThreadWorker       m_threadWorker;
};

} // namespace kae
