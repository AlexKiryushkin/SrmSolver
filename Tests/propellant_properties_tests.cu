
#include <ratio>

#include <gtest/gtest.h>

#include <SrmSolver/gas_state.h>
#include <SrmSolver/propellant_properties.h>

namespace tests {

template <class T>
class propellant_properties : public ::testing::Test {};

using TypeParams = ::testing::Types<float, double>;
TYPED_TEST_CASE(propellant_properties, TypeParams);

TYPED_TEST(propellant_properties, propellant_properties_fields)
{
  using ElemType                 = TypeParam;
  using NuType                   = std::ratio<5, 10>;
  using MtType                   = std::ratio<3, 1000>;
  using TBurnType                = std::ratio<1, 1>;
  using RhoPType                 = std::ratio<300, 1>;
  using KappaType                = std::ratio<12, 10>;
  using CpType                   = std::ratio<6, 1>;
  using P0Type                   = std::ratio<1, 1000>;
  using PropellantPropertiesType = kae::PropellantProperties<NuType, MtType, TBurnType, RhoPType, P0Type, ElemType>;
  using GasStateType             = kae::GasState<KappaType, CpType, ElemType>;

  constexpr ElemType goldNu{ static_cast<ElemType>(0.5) };
  constexpr ElemType goldMt{ static_cast<ElemType>(0.003) };
  constexpr ElemType goldTBurn{ static_cast<ElemType>(1.0) };
  constexpr ElemType goldRhoP{ static_cast<ElemType>(300.0) };
  constexpr ElemType goldH0{ static_cast<ElemType>(6.0) };
  constexpr ElemType goldP0{ static_cast<ElemType>(0.001) };

  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_NEAR(PropellantPropertiesType::nu,                        goldNu,    threshold);
  EXPECT_NEAR(PropellantPropertiesType::mt,                        goldMt,    threshold);
  EXPECT_NEAR(PropellantPropertiesType::TBurn,                     goldTBurn, threshold);
  EXPECT_NEAR(PropellantPropertiesType::rhoP,                      goldRhoP,  threshold);
  EXPECT_NEAR(PropellantPropertiesType::template H0<GasStateType>, goldH0,    threshold);
  EXPECT_NEAR(PropellantPropertiesType::P0,                        goldP0,    threshold);
}

} // namespace tests