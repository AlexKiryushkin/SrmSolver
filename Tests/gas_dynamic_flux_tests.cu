
#include <ratio>
#include <vector>

#include <gtest/gtest.h>

#include <SrmSolver/gas_dynamic_flux.h>
#include <SrmSolver/gas_state.h>

#include "comparators.h"

namespace tests {

class gas_dynamic_flux : public ::testing::Test
{
public:
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  constexpr static unsigned bias{ 5U };
  constexpr static unsigned startPoint{ 0U };
  constexpr static unsigned endPoint{ 1U };
  constexpr static unsigned pointsCount{ 101U };
  constexpr static float step{ static_cast<float>(endPoint - startPoint) / static_cast<float>(pointsCount - 1U) };
  constexpr static float lambda{ 2.5f };

  static float xCoordinate(unsigned pointNumber) { return step * pointNumber; }
  static float xCoordinatePlus(unsigned pointNumber) { return step * (static_cast<float>(pointNumber) + 0.5f); }
  static float yCoordinate(unsigned pointNumber) { return step * pointNumber; }
  static float yCoordinatePlus(unsigned pointNumber) { return step * (static_cast<float>(pointNumber) + 0.5f); }

  static float rho(float x, float y) { return 0.1f + 0.1f * (x + y); }
  static float ux(float x, float y) { return 0.2f * (x + y); }
  static float uy(float x, float y) { return -0.2f * (x + y); }
  static float p(float x, float y) { return 0.2f + 0.1f * (x + y); }

  static GasStateT gasState(float x, float y) { return GasStateT{ rho(x, y), ux(x, y), uy(x, y), p(x, y) }; }

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

TEST_F(gas_dynamic_flux, gas_dynamic_flux_mass_flux)
{
  auto pGasState = gasStates.data();
  for (unsigned i{ bias }; i < pointsCount - bias; ++i)
  {
    for (unsigned j{ bias }; j < pointsCount - bias; ++j)
    {
      auto index = j * pointsCount + i;

      const auto calculatedFluxX = kae::getXFluxes<1U>(pGasState, index, lambda);
      const auto calculatedFluxY = kae::getYFluxes<pointsCount>(pGasState, index, lambda);


      const auto goldFluxX = kae::XFluxes::get(gasState(xCoordinatePlus(i), yCoordinate(j)));
      const auto goldFluxY = kae::YFluxes::get(gasState(xCoordinate(i), yCoordinatePlus(j)));

      constexpr float threshold{ 1e-5f };
      EXPECT_FLOAT4_NEAR(calculatedFluxX, goldFluxX, threshold);
      EXPECT_FLOAT4_NEAR(calculatedFluxY, goldFluxY, threshold);
    }
  }
}

} // namespace tests