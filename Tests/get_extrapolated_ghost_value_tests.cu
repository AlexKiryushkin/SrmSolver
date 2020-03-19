
#include <ratio>

#include <gtest/gtest.h>

#include <SrmSolver/gas_state.h>
#include <SrmSolver/get_extrapolated_ghost_value.h>
#include <SrmSolver/propellant_properties.h>

#include "comparators.h"

namespace tests {

class get_extrapolated_ghost_value : public ::testing::Test
{
public:
  using KappaType = std::ratio<12, 10>;
  using CpType = std::ratio<6045, 1000>;
  using GasStateType = kae::GasState<KappaType, CpType>;

  using NuType = std::ratio<5, 10>;
  using MtType = std::ratio<-3, 10>;
  using TBurnType = std::ratio<1, 1>;
  using RhoPType = std::ratio<300, 1>;
  using P0Type = std::ratio<144, 1000>;
  using PropellantPropertiesType = kae::PropellantProperties<NuType, MtType, TBurnType, RhoPType, P0Type>;
};

TEST_F(get_extrapolated_ghost_value, get_extrapolated_ghost_value_wall)
{
  const GasStateType gasState{ 1.0f, 1.5f, 0.5f, 1.2f };
  const auto extrapolatedState = kae::detail::getFirstOrderExtrapolatedGhostValue<PropellantPropertiesType>(gasState, kae::EBoundaryCondition::eWall);

  const GasStateType goldExtrapolatedState{ 2.25f, 0.0f, 0.5f, 3.0f };
  constexpr float threshold{ 1e-6f };
  EXPECT_GAS_STATE_NEAR(extrapolatedState, goldExtrapolatedState, threshold);
}

TEST_F(get_extrapolated_ghost_value, get_extrapolated_ghost_value_pressure_outlet_supersonic)
{
  const GasStateType gasState{ 1.0f, 1.5f, 0.5f, 1.2f };
  const auto extrapolatedState = kae::detail::getFirstOrderExtrapolatedGhostValue<PropellantPropertiesType>(gasState, kae::EBoundaryCondition::ePressureOutlet);

  const GasStateType goldExtrapolatedState{ 1.0f, 1.5f, 0.5f, 1.2f };
  constexpr float threshold{ 1e-6f };
  EXPECT_GAS_STATE_NEAR(extrapolatedState, goldExtrapolatedState, threshold);
}

TEST_F(get_extrapolated_ghost_value, get_extrapolated_ghost_value_pressure_outlet_subsonic)
{
  const GasStateType gasState{ 1.2f, 0.2f, 0.5f, 1.44f };
  const auto extrapolatedState = kae::detail::getFirstOrderExtrapolatedGhostValue<PropellantPropertiesType>(gasState, kae::EBoundaryCondition::ePressureOutlet);

  const GasStateType goldExtrapolatedState{ 0.3f, 1.1f, 0.5f, PropellantPropertiesType::P0 };
  constexpr float threshold{ 1e-6f };
  EXPECT_GAS_STATE_NEAR(extrapolatedState, goldExtrapolatedState, threshold);
}

TEST_F(get_extrapolated_ghost_value, get_extrapolated_ghost_value_massflow_inlet)
{
  const GasStateType gasState{ 1.0f, -0.3f, 0.1f, 1.0f };
  const auto extrapolatedState = kae::detail::getFirstOrderExtrapolatedGhostValue<PropellantPropertiesType>(gasState, kae::EBoundaryCondition::eMassFlowInlet);

  const GasStateType goldExtrapolatedState{ 1.0f, -0.3f, 0.0f, 1.0f };
  constexpr float threshold{ 1e-6f };
  EXPECT_GAS_STATE_NEAR(extrapolatedState, goldExtrapolatedState, threshold);
}

} // namespace tests
