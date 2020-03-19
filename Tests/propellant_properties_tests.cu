
#include <ratio>

#include <gtest/gtest.h>

#include <SrmSolver/gas_state.h>
#include <SrmSolver/propellant_properties.h>

namespace tests {

TEST(propellant_properties, propellant_properties_fields)
{
  using NuType                   = std::ratio<5, 10>;
  using MtType                   = std::ratio<3, 1000>;
  using TBurnType                = std::ratio<1, 1>;
  using RhoPType                 = std::ratio<300, 1>;
  using KappaType                = std::ratio<12, 10>;
  using CpType                   = std::ratio<6, 1>;
  using P0Type                   = std::ratio<1, 1000>;
  using PropellantPropertiesType = kae::PropellantProperties<NuType, MtType, TBurnType, RhoPType, P0Type>;
  using GasStateType             = kae::GasState<KappaType, CpType>;

  constexpr float goldNu{ 0.5f };
  constexpr float goldMt{ 0.003f };
  constexpr float goldTBurn{ 1.0f };
  constexpr float goldRhoP{ 300.0f };
  constexpr float goldH0{ 6.0f };
  constexpr float goldP0{ 0.001f };

  constexpr float threshold{ 1e-6f };
  EXPECT_NEAR(PropellantPropertiesType::nu,               goldNu,    threshold);
  EXPECT_NEAR(PropellantPropertiesType::mt,               goldMt,    threshold);
  EXPECT_NEAR(PropellantPropertiesType::TBurn,            goldTBurn, threshold);
  EXPECT_NEAR(PropellantPropertiesType::rhoP,             goldRhoP,  threshold);
  EXPECT_NEAR(PropellantPropertiesType::H0<GasStateType>, goldH0,    threshold);
  EXPECT_NEAR(PropellantPropertiesType::P0,               goldP0,    threshold);
}

} // namespace tests