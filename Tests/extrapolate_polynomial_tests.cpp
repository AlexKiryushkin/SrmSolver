
#include <SrmSolver/std_includes.h>

#include <gtest/gtest.h>

#include <SrmSolver/get_coordinates_matrix.h>
#include <SrmSolver/get_polynomial.h>
#include <SrmSolver/get_stencil_indices.h>
#include <SrmSolver/gpu_grid.h>
#include <SrmSolver/linear_system_solver.h>  
#include <SrmSolver/matrix.h>
#include <SrmSolver/matrix_operations.h>

#include "aliases.h"

namespace kae_tests {

template <class T>
class extrapolate_polynomial_tests : public ::testing::Test
{
public:

  void test()
  {
    using kae::detail::getStencilIndices;
    using kae::detail::getCoordinatesMatrix;
    using kae::detail::getRightHandSideMatrix;

    using tf = extrapolate_polynomial_tests<T>;
    using ElemType = typename tf::ElemType;
    using Real2Type = typename tf::Real2Type;
    using IndexMatrixType = typename tf::IndexMatrixType;
    using GpuGridType = typename tf::GpuGridType;
    using InitList = std::initializer_list<unsigned>;

    const auto negativeValue = static_cast<ElemType>(-0.1);
    std::vector<ElemType> phiValues(GpuGridType::nx * GpuGridType::ny, negativeValue);
    const auto gasStates = this->generateGasStates();
    const auto pGasValues = gasStates.data();

    std::cout.precision(9);
    std::cout << std::setw(25);
    const auto indexMatrix = getStencilIndices<GpuGridType, tf::order>(phiValues.data(), tf::surfacePoint, tf::normal);
    const auto coefficients = kae::detail::getWenoPolynomial<GpuGridType>(tf::surfacePoint,
      tf::normal,
      pGasValues,
      indexMatrix);

    const auto goldCoefficients = tf::goldCoefficients();
    const auto thresholdMatrix = cwiseAbs(goldCoefficients - coefficients);
    const auto goldThresholdMatrix = tf::thresholdMatrix();
    const auto diffMatrix = thresholdMatrix - goldThresholdMatrix;
    const auto maxDiff = maxCoeff(diffMatrix);
    EXPECT_LE(maxDiff, 0) << coefficients << "\n\n" << goldCoefficients << "\n\n"
      << cwiseAbs(goldCoefficients - coefficients) << "\n\n"
      << diffMatrix << "\n\n";
  }

private:

  constexpr static unsigned order{ 2U };
  constexpr static unsigned nCoefficients{ order * (order + 1U) / 2U };
  using ElemType        = T;
  using Real2Type       = kae::CudaFloat2T<T>;
  using IndexMatrixType = kae::Matrix<unsigned, order, order>;

  constexpr static unsigned nx{ 2001U };
  constexpr static unsigned ny{ 1001U };
  constexpr static unsigned smExtension{ 3U };
  using LxToType    = std::ratio<2, 1>;
  using LyToType    = std::ratio<1, 1>;
  using GpuGridType = kae::GpuGrid<nx, ny, LxToType, LyToType, smExtension, ElemType>;

  constexpr static ElemType hx = GpuGridType::hx;
  constexpr static ElemType hy = GpuGridType::hy;

  using KappaType    = std::ratio<12, 10>;
  using RType        = std::ratio<6, 1>;
  using GasStateType = GasStateType<KappaType, RType, ElemType>;

  constexpr static Real2Type surfacePoint{ static_cast<ElemType>(0.45642), static_cast<ElemType>(0.33522) };
  constexpr static Real2Type normal{ static_cast<ElemType>(0.8), static_cast<ElemType>(0.6) };

  constexpr static std::array<ElemType, nCoefficients> rhoCoeffs{
    static_cast<ElemType>(1.0), 
    static_cast<ElemType>(2.0), 
    static_cast<ElemType>(3.0)};
  constexpr static std::array<ElemType, nCoefficients> uCoeffs{
    static_cast<ElemType>(3.0),
    static_cast<ElemType>(1.0),
    static_cast<ElemType>(-2.4) };
  constexpr static std::array<ElemType, nCoefficients> vCoeffs{
    static_cast<ElemType>(4.0),
    static_cast<ElemType>(3.0),
    static_cast<ElemType>(1.5)};
  constexpr static std::array<ElemType, nCoefficients> pCoeffs{
    static_cast<ElemType>(2.0),
    static_cast<ElemType>(-1.0),
    static_cast<ElemType>(2.5) };

  constexpr static std::array<ElemType, nCoefficients> unCoeffs{
    std::get<0U>(uCoeffs) * normal.x + std::get<0U>(vCoeffs) * normal.y,
    std::get<1U>(uCoeffs) * normal.x + std::get<1U>(vCoeffs) * normal.y,
    std::get<2U>(uCoeffs) * normal.x + std::get<2U>(vCoeffs) * normal.y
  };

  constexpr static std::array<ElemType, nCoefficients> utauCoeffs{
    -std::get<0U>(uCoeffs) * normal.y + std::get<0U>(vCoeffs) * normal.x,
    -std::get<1U>(uCoeffs) * normal.y + std::get<1U>(vCoeffs) * normal.x,
    -std::get<2U>(uCoeffs) * normal.y + std::get<2U>(vCoeffs) * normal.x
  };

  constexpr static ElemType multiplier{ static_cast<ElemType>(5.0) };
  constexpr static std::array<ElemType, nCoefficients> thresholdVector{
    multiplier * hx * hx, multiplier * hx, multiplier * hy
  };

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

        const auto rho = rhoCoeffs[0] + rhoCoeffs[1] * dn + rhoCoeffs[2] * dtau;
        const auto u   = uCoeffs[0]   + uCoeffs[1]   * dn + uCoeffs[2]   * dtau;
        const auto v   = vCoeffs[0]   + vCoeffs[1]   * dn + vCoeffs[2]   * dtau;
        const auto p   = pCoeffs[0]   + pCoeffs[1]   * dn + pCoeffs[2]   * dtau;

        gasStates.at(index) = GasStateType{ rho, u, v, p };
      }
    }
    return gasStates;
  }

  static kae::Matrix<ElemType, nCoefficients, 4U> goldCoefficients()
  {
    kae::Matrix<ElemType, nCoefficients, 4U> coefficients{};
    kae::setCol(coefficients, 0U, kae::Matrix<ElemType, nCoefficients, 1U>(rhoCoeffs.data()));
    kae::setCol(coefficients, 1U, kae::Matrix<ElemType, nCoefficients, 1U>(unCoeffs.data()));
    kae::setCol(coefficients, 2U, kae::Matrix<ElemType, nCoefficients, 1U>(utauCoeffs.data()));
    kae::setCol(coefficients, 3U, kae::Matrix<ElemType, nCoefficients, 1U>(pCoeffs.data()));

    return coefficients;
  }

  static kae::Matrix<ElemType, nCoefficients, 4U> thresholdMatrix()
  {
    kae::Matrix<ElemType, nCoefficients, 4U> thresholds{};
    kae::setCol(thresholds, 0U, kae::Matrix<ElemType, nCoefficients, 1U>(thresholdVector.data()));
    kae::setCol(thresholds, 1U, kae::Matrix<ElemType, nCoefficients, 1U>(thresholdVector.data()));
    kae::setCol(thresholds, 2U, kae::Matrix<ElemType, nCoefficients, 1U>(thresholdVector.data()));
    kae::setCol(thresholds, 3U, kae::Matrix<ElemType, nCoefficients, 1U>(thresholdVector.data()));
    return thresholds;
  }
}; 

using TypeParams = ::testing::Types<float, double>;
TYPED_TEST_SUITE(extrapolate_polynomial_tests, TypeParams);

TYPED_TEST(extrapolate_polynomial_tests, extrapolate_polynomial_tests_1)
{
  this->test();
}

} // namespace kae_tests
