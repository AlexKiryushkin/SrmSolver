
#include <gtest/gtest.h>

#include <SrmSolver/get_coordinates_matrix.h>
#include <SrmSolver/get_stencil_indices.h>
#include <SrmSolver/gpu_grid.h>
#include <SrmSolver/linear_system_solver.h>  

#include "aliases.h"

namespace tests {

template <class T>
class extrapolate_polynomial_tests : public ::testing::Test
{
public:

  constexpr static unsigned order{ 3U };
  using ElemType        = T;
  using Real2Type       = kae::CudaFloat2T<T>;
  using IndexMatrixType = Eigen::Matrix<unsigned, order, order>;

  constexpr static unsigned nx{ 201U };
  constexpr static unsigned ny{ 101U };
  constexpr static unsigned smExtension{ 3U };
  using LxToType    = std::ratio<2, 1>;
  using LyToType    = std::ratio<1, 1>;
  using GpuGridType = kae::GpuGrid<nx, ny, LxToType, LyToType, smExtension, ElemType>;

  using KappaType    = std::ratio<12, 10>;
  using RType        = std::ratio<6, 1>;
  using GasStateType = GasStateType<KappaType, RType, ElemType>;

  constexpr static Real2Type surfacePoint{ static_cast<ElemType>(0.45642), static_cast<ElemType>(0.33522) };
  constexpr static Real2Type normal{ static_cast<ElemType>(0.8), static_cast<ElemType>(0.6) };

  constexpr static std::array<ElemType, 6U> rhoCoeffs{ 1.0, 2.0, 3.0, -4.0, 5.0, 6.0 };
  constexpr static std::array<ElemType, 6U> unCoeffs{ 3.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  constexpr static std::array<ElemType, 6U> utauCoeffs{ 4.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  constexpr static std::array<ElemType, 6U> pCoeffs{ 2.0, -1.0, 2.5, 3.0, 0.5, 1.5 };

  static std::vector<GasStateType> generateGasStates()
  {
    std::vector<GasStateType> gasStates(GpuGridType::nx * GpuGridType::ny, GasStateType{});
    for (std::size_t i{}; i < GpuGridType::nx; ++i)
    {
      for (std::size_t j{}; j < GpuGridType::ny; ++j)
      {
        const auto index = j * GpuGridType::nx + i;
        const auto dx    = i * GpuGridType::hx - surfacePoint.x;
        const auto dy    = j * GpuGridType::hy - surfacePoint.y;

        const auto dn    = dx * normal.x + dy * normal.y;
        const auto dtau  = -dx * normal.y + dy * normal.x;

        const auto rho = rhoCoeffs[0] + rhoCoeffs[1] * dn + rhoCoeffs[2] * dtau +
          rhoCoeffs[3] * dn * dn + rhoCoeffs[4] * dn * dtau + rhoCoeffs[5] * dtau * dtau;
        const auto un = unCoeffs[0] + unCoeffs[1] * dn + unCoeffs[2] * dtau +
          unCoeffs[3] * dn * dn + unCoeffs[4] * dn * dtau + unCoeffs[5] * dtau * dtau;
        const auto utau = utauCoeffs[0] + utauCoeffs[1] * dn + utauCoeffs[2] * dtau +
          utauCoeffs[3] * dn * dn + utauCoeffs[4] * dn * dtau + utauCoeffs[5] * dtau * dtau;
        const auto p = pCoeffs[0] + pCoeffs[1] * dn + pCoeffs[2] * dtau +
          pCoeffs[3] * dn * dn + pCoeffs[4] * dn * dtau + pCoeffs[5] * dtau * dtau;

        gasStates.at(index) = GasStateType{ rho, un, utau, p };
      }
    }
    return gasStates;
  }
}; 

using TypeParams = ::testing::Types<float, double>;
TYPED_TEST_SUITE(extrapolate_polynomial_tests, TypeParams);

TYPED_TEST(extrapolate_polynomial_tests, extrapolate_polynomial_tests_1)
{
  using kae::detail::getStencilIndices;
  using kae::detail::getCoordinatesMatrix;
  using kae::detail::getRightHandSideMatrix;

  using tf              = TestFixture;
  using ElemType        = typename tf::ElemType;
  using Real2Type       = typename tf::Real2Type;
  using IndexMatrixType = typename tf::IndexMatrixType;
  using GpuGridType     = typename tf::GpuGridType;
  using InitList        = std::initializer_list<unsigned>;

  const auto negativeValue = static_cast<ElemType>(-0.1);
  std::vector<ElemType> phiValues(GpuGridType::nx * GpuGridType::ny, negativeValue);
  const auto gasStates = this->generateGasStates();
  const auto pGasValues = gasStates.data();

  const auto indexMatrix = getStencilIndices<GpuGridType, tf::order>(phiValues.data(), tf::surfacePoint, tf::normal);
  const auto lhsMatrix   = getCoordinatesMatrix<GpuGridType, tf::order>(tf::surfacePoint, tf::normal, indexMatrix);
  const auto rhsMatrix   = getRightHandSideMatrix<GpuGridType, tf::order>(tf::normal, pGasValues, indexMatrix);
  const auto A = lhsMatrix.transpose() * lhsMatrix;
  const auto b = lhsMatrix.transpose() * rhsMatrix;
  const auto x = kae::detail::choleskySolve(A, b);

  std::cout << x << '\n';
}

} // namespace tests
