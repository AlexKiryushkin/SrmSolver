#pragma once

#include "std_includes.h"

#include "cuda_float_types.h"
#include "filesystem.h"
#include "gpu_matrix.h"
#include "gpu_matrix_writer.h"

namespace kae {

class WriteToFolderCallback
{
public:
  WriteToFolderCallback(std::wstring folderPath) : m_folderPath(std::move(folderPath))
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

private:
  std::wstring m_folderPath;
};

} // namespace kae
