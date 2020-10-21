
#include <ratio>
#include <vector>

#include <gtest/gtest.h>

#include <SrmSolver/gas_dynamic_flux.h>
#include <SrmSolver/gas_state.h>

#include "aliases.h"
#include "comparators.h"

namespace kae_tests {

template <class T>
class gas_dynamic_flux : public ::testing::Test
{
public:
  using ElemType  = T;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = GasStateType<KappaT, CpT, ElemType>;

  constexpr static unsigned bias{ 5U };
  constexpr static ElemType startPoint{ static_cast<ElemType>(0.5) };
  constexpr static ElemType endPoint{ static_cast<ElemType>(1.0) };
  constexpr static unsigned pointsCount{ 201U };
  constexpr static ElemType step{ (endPoint - startPoint) / static_cast<ElemType>(pointsCount - 1U) };
  constexpr static ElemType hx{ step };
  constexpr static ElemType lambda{ static_cast<ElemType>(2.5) };

  static ElemType xCoordinate(unsigned pointNumber) { return startPoint + step * pointNumber; }
  static ElemType xCoordinatePlus(unsigned pointNumber) { return startPoint + step * (static_cast<ElemType>(pointNumber) + static_cast<ElemType>(0.5)); }
  static ElemType yCoordinate(unsigned pointNumber) { return startPoint + step * pointNumber; }
  static ElemType yCoordinatePlus(unsigned pointNumber) { return startPoint + step * (static_cast<ElemType>(pointNumber) + static_cast<ElemType>(0.5)); }

  static ElemType rho(ElemType x, ElemType y) { return static_cast<ElemType>(0.1) + static_cast<ElemType>(0.1) * (x + y); }
  static ElemType ux(ElemType x, ElemType y) { return static_cast<ElemType>(0.2) * (x + y); }
  static ElemType uy(ElemType x, ElemType y) { return static_cast<ElemType>(-0.2) * (x + y); }
  static ElemType p(ElemType x, ElemType y) { return static_cast<ElemType>(0.2) + static_cast<ElemType>(0.1) * (x + y); }

  static GasStateT gasState(ElemType x, ElemType y) { return GasStateT{ rho(x, y), ux(x, y), uy(x, y), p(x, y) }; }

  gas_dynamic_flux()
  {
    for (unsigned i{ 0U }; i < pointsCount; ++i)
    {
      auto x = xCoordinate(i);
      for (unsigned j{ 0U }; j < pointsCount; ++j)
      {
        auto y = yCoordinate(j);
        gasStates.push_back(gasState(x, y));
      }
    }
  }

  std::vector<GasStateT> gasStates;
};

using TypeParams = ::testing::Types<float, double>;
TYPED_TEST_SUITE(gas_dynamic_flux, TypeParams);

TYPED_TEST(gas_dynamic_flux, gas_dynamic_flux_mass_flux)
{
  using tf = TestFixture;
  using ElemType = typename tf::ElemType;

  auto pGasState = tf::gasStates.data();
  for (unsigned i{ tf::bias }; i < tf::pointsCount - tf::bias; ++i)
  {
    for (unsigned j{ tf::bias }; j < tf::pointsCount - tf::bias; ++j)
    {
      auto index = j * tf::pointsCount + i;

      const auto calculatedFluxX = kae::getXFluxes<1U, tf>(pGasState, index, tf::lambda);
      const auto calculatedFluxY = kae::getYFluxes<tf::pointsCount, tf>(pGasState, index, tf::lambda);

      const auto goldFluxX = kae::XFluxes::get(tf::gasState(tf::xCoordinatePlus(i), tf::yCoordinate(j)));
      const auto goldFluxY = kae::YFluxes::get(tf::gasState(tf::xCoordinate(i), tf::yCoordinatePlus(j)));

      constexpr ElemType threshold1 = 10 * tf::step * tf::step * tf::step;
      constexpr ElemType threshold2{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(4e-7) :
                                                                           static_cast<ElemType>(4e-11) };
      constexpr ElemType maxThreshold{ std::max(threshold1, threshold2) };

      EXPECT_FLOAT4_NEAR(calculatedFluxX, goldFluxX, maxThreshold);
      EXPECT_FLOAT4_NEAR(calculatedFluxY, goldFluxY, maxThreshold);
    }
  }
}

} // namespace kae_tests