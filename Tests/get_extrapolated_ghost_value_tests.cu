
#include <ratio>

#include <gtest/gtest.h>

#include <SrmSolver/gas_state.h>
#include <SrmSolver/get_extrapolated_ghost_value.h>
#include <SrmSolver/propellant_properties.h>

#include "aliases.h"
#include "comparators.h"

namespace tests {

template <class T>
class get_extrapolated_ghost_value : public ::testing::Test
{
public:
  using ElemType     = T;
  using KappaType    = std::ratio<12, 10>;
  using CpType       = std::ratio<6045, 1000>;
  using GasStateType = GasStateType<KappaType, CpType, ElemType>;

  using NuType                   = std::ratio<5, 10>;
  using MtType                   = std::ratio<-3, 10>;
  using TBurnType                = std::ratio<1, 1>;
  using RhoPType                 = std::ratio<300, 1>;
  using P0Type                   = std::ratio<144, 1000>;
  using PropellantPropertiesType = 
      PhysicalProperties<NuType, MtType, TBurnType, RhoPType, P0Type, KappaType, CpType, ElemType>;
};

using TypeParams = ::testing::Types<float, double>;
TYPED_TEST_CASE(get_extrapolated_ghost_value, TypeParams);

TYPED_TEST(get_extrapolated_ghost_value, get_extrapolated_ghost_value_wall)
{
  using ElemType                 = typename TestFixture::ElemType;
  using GasStateType             = typename TestFixture::GasStateType;
  using PropellantPropertiesType = typename TestFixture::PropellantPropertiesType;

  const GasStateType gasState{ static_cast<ElemType>(1.0),
                               static_cast<ElemType>(1.5),
                               static_cast<ElemType>(0.5),
                               static_cast<ElemType>(1.2) };
  const auto extrapolatedState = kae::detail::getFirstOrderExtrapolatedGhostValue<PropellantPropertiesType>(gasState, kae::EBoundaryCondition::eWall);

  const GasStateType goldExtrapolatedState{ static_cast<ElemType>(2.25),
                                            static_cast<ElemType>(0.0),
                                            static_cast<ElemType>(0.5),
                                            static_cast<ElemType>(3.0) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_GAS_STATE_NEAR(extrapolatedState, goldExtrapolatedState, threshold);
}

TYPED_TEST(get_extrapolated_ghost_value, get_extrapolated_ghost_value_pressure_outlet_supersonic)
{
  using ElemType                 = typename TestFixture::ElemType;
  using GasStateType             = typename TestFixture::GasStateType;
  using PropellantPropertiesType = typename TestFixture::PropellantPropertiesType;

  const GasStateType gasState{ static_cast<ElemType>(1.0),
                               static_cast<ElemType>(1.5),
                               static_cast<ElemType>(0.5),
                               static_cast<ElemType>(1.2) };
  const auto extrapolatedState = kae::detail::getFirstOrderExtrapolatedGhostValue<PropellantPropertiesType>(gasState, kae::EBoundaryCondition::ePressureOutlet);

  const GasStateType goldExtrapolatedState{ static_cast<ElemType>(1.0),
                                            static_cast<ElemType>(1.5),
                                            static_cast<ElemType>(0.5),
                                            static_cast<ElemType>(1.2) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_GAS_STATE_NEAR(extrapolatedState, goldExtrapolatedState, threshold);
}

TYPED_TEST(get_extrapolated_ghost_value, get_extrapolated_ghost_value_pressure_outlet_subsonic)
{
  using ElemType                 = typename TestFixture::ElemType;
  using GasStateType             = typename TestFixture::GasStateType;
  using PropellantPropertiesType = typename TestFixture::PropellantPropertiesType;

  const GasStateType gasState{ static_cast<ElemType>(1.2),
                               static_cast<ElemType>(0.2),
                               static_cast<ElemType>(0.5),
                               static_cast<ElemType>(1.44) };
  const auto extrapolatedState = kae::detail::getFirstOrderExtrapolatedGhostValue<PropellantPropertiesType>(gasState, kae::EBoundaryCondition::ePressureOutlet);

  const GasStateType goldExtrapolatedState{ static_cast<ElemType>(0.3),
                                            static_cast<ElemType>(1.1),
                                            static_cast<ElemType>(0.5),
                                            PropellantPropertiesType::P0 };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_GAS_STATE_NEAR(extrapolatedState, goldExtrapolatedState, threshold);
}

TYPED_TEST(get_extrapolated_ghost_value, get_extrapolated_ghost_value_massflow_inlet)
{
  using ElemType = typename TestFixture::ElemType;
  using GasStateType = typename TestFixture::GasStateType;
  using PropellantPropertiesType = typename TestFixture::PropellantPropertiesType;

  const GasStateType gasState{ static_cast<ElemType>(1.0),
                               static_cast<ElemType>(-0.3),
                               static_cast<ElemType>(0.1),
                               static_cast<ElemType>(1.0) };
  const auto extrapolatedState = kae::detail::getFirstOrderExtrapolatedGhostValue<PropellantPropertiesType>(gasState, kae::EBoundaryCondition::eMassFlowInlet);

  const GasStateType goldExtrapolatedState{ static_cast<ElemType>(1.0),
                                            static_cast<ElemType>(-0.3),
                                            static_cast<ElemType>(0.0),
                                            static_cast<ElemType>(1.0) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_GAS_STATE_NEAR(extrapolatedState, goldExtrapolatedState, threshold);
}

} // namespace tests
