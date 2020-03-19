
#include <ratio>

#include <gtest/gtest.h>

#include <SrmSolver/gas_state.h>

#include "comparators.h"

namespace tests {

TEST(gas_state, gas_state_fields)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 1.0f, 2.0f, 3.0f, 4.0f };

  constexpr float goldKappa{ 1.2f };
  constexpr float goldCp{ 6.0f };
  constexpr float goldR{ 1.0f };
  constexpr float goldRho{ 1.0f };
  constexpr float goldUx{ 2.0f };
  constexpr float goldUy{ 3.0f };
  constexpr float goldP{ 4.0f };

  constexpr float threshold{ 1e-6f };
  EXPECT_NEAR(GasStateT::kappa, goldKappa, threshold);
  EXPECT_NEAR(GasStateT::Cp, goldCp, threshold);
  EXPECT_NEAR(GasStateT::R, goldR, threshold);
  EXPECT_EQ(gasState.rho, goldRho);
  EXPECT_EQ(gasState.ux, goldUx);
  EXPECT_EQ(gasState.uy, goldUy);
  EXPECT_EQ(gasState.p, goldP);
}

TEST(gas_state, gas_state_rho)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 1.0f, 3.0f, 4.0f, 2.0f };
  const float rho = kae::Rho::get(gasState);

  constexpr float goldRho{ 1.0f };
  constexpr float threshold{ 1e-6f };
  EXPECT_NEAR(rho, goldRho, threshold);
}

TEST(gas_state, gas_state_p)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 1.0f, 3.0f, 4.0f, 2.0f };
  const float p = kae::P::get(gasState);

  constexpr float goldP{ 2.0f };
  constexpr float threshold{ 1e-6f };
  EXPECT_NEAR(p, goldP, threshold);
}

TEST(gas_state, gas_state_velocity_squared)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 1.0f, 3.0f, 4.0f, 2.0f };
  const float velocitySquared = kae::VelocitySquared::get(gasState);

  constexpr float goldVelocitySquared{ 25.0f };
  constexpr float threshold{ 1e-6f };
  EXPECT_NEAR(velocitySquared, goldVelocitySquared, threshold);
}

TEST(gas_state, gas_state_velocity)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 1.0f, 3.0f, 4.0f, 2.0f };
  const float velocity = kae::Velocity::get(gasState);

  constexpr float goldVelocity{ 5.0f };
  constexpr float threshold{ 1e-6f };
  EXPECT_NEAR(velocity, goldVelocity, threshold);
}

TEST(gas_state, gas_state_mass_flux_x)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 5.0f, 3.0f, 4.0f, 2.0f };
  const float massFluxX = kae::MassFluxX::get(gasState);

  const float goldMassFluxX{ 15.0f };
  const float threshold{ 1e-6f };
  EXPECT_NEAR(massFluxX, goldMassFluxX, threshold);
}

TEST(gas_state, gas_state_mass_flux_y)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 5.0f, 3.0f, 4.0f, 2.0f };
  const float massFluxY = kae::MassFluxY::get(gasState);

  const float goldMassFluxY{ 20.0f };
  const float threshold{ 1e-6f };
  EXPECT_NEAR(massFluxY, goldMassFluxY, threshold);
}

TEST(gas_state, gas_state_momentum_flux_xx)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 5.0f, 3.0f, 4.0f, 2.0f };
  const float momentumFluxXx = kae::MomentumFluxXx::get(gasState);

  const float goldMomentumFluxXx{ 47.0f };
  const float threshold{ 1e-6f };
  EXPECT_NEAR(momentumFluxXx, goldMomentumFluxXx, threshold);
}

TEST(gas_state, gas_state_momentum_flux_xy)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 5.0f, 3.0f, 4.0f, 2.0f };
  const float momentumFluxXy = kae::MomentumFluxXy::get(gasState);

  const float goldMomentumFluxXy{ 60.0f };
  const float threshold{ 1e-6f };
  EXPECT_NEAR(momentumFluxXy, goldMomentumFluxXy, threshold);
}

TEST(gas_state, gas_state_momentum_flux_yy)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 5.0f, 3.0f, 4.0f, 2.0f };
  const float momentumFluxYy = kae::MomentumFluxYy::get(gasState);

  const float goldMomentumFluxYy{ 82.0f };
  const float threshold{ 1e-6f };
  EXPECT_NEAR(momentumFluxYy, goldMomentumFluxYy, threshold);
}

TEST(gas_state, gas_state_rho_energy)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 5.0f, 3.0f, 4.0f, 2.0f };
  const float rhoEnergy = kae::RhoEnergy::get(gasState);

  const float goldRhoEnergy{ 72.5f };
  const float threshold{ 1e-6f };
  EXPECT_NEAR(rhoEnergy, goldRhoEnergy, threshold);
}

TEST(gas_state, gas_state_energy)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 5.0f, 3.0f, 4.0f, 2.0f };
  const float energy = kae::Energy::get(gasState);

  const float goldEnergy{ 14.5f };
  const float threshold{ 1e-6f };
  EXPECT_NEAR(energy, goldEnergy, threshold);
}

TEST(gas_state, gas_state_enthalpy_flux_x)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 5.0f, 3.0f, 4.0f, 2.0f };
  const float enthalpyFluxX = kae::EnthalpyFluxX::get(gasState);

  const float goldEnthalpyFluxX{ 223.5f };
  const float threshold{ 1e-6f };
  EXPECT_NEAR(enthalpyFluxX, goldEnthalpyFluxX, threshold);
}

TEST(gas_state, gas_state_enthalpy_flux_y)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 5.0f, 3.0f, 4.0f, 2.0f };
  const float enthalpyFluxY = kae::EnthalpyFluxY::get(gasState);

  const float goldEnthalpyFluxY{ 298.0f };
  const float threshold{ 1e-6f };
  EXPECT_NEAR(enthalpyFluxY, goldEnthalpyFluxY, threshold);
}

TEST(gas_state, gas_state_sonic_speed_squared)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 2.0f, 3.0f, 4.0f, 2.4f };
  const float sonicSpeedSquared = kae::SonicSpeedSquared::get(gasState);

  const float goldSonicSpeedSquared{ 1.44f };
  const float threshold{ 1e-6f };
  EXPECT_NEAR(sonicSpeedSquared, goldSonicSpeedSquared, threshold);
}

TEST(gas_state, gas_state_sonic_speed)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 2.0f, 3.0f, 4.0f, 2.4f };
  const float sonicSpeed = kae::SonicSpeed::get(gasState);

  const float goldSonicSpeed{ 1.2f };
  const float threshold{ 1e-6f };
  EXPECT_NEAR(sonicSpeed, goldSonicSpeed, threshold);
}

TEST(gas_state, gas_state_mach)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 2.0f, 3.0f, 4.0f, 2.4f };
  const float mach = kae::Mach::get(gasState);

  const float goldMach{ 5.0f / 1.2f };
  const float threshold{ 1e-6f };
  EXPECT_NEAR(mach, goldMach, threshold);
}

TEST(gas_state, gas_state_temperature)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<12, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 2.0f, 3.0f, 4.0f, 6.0f };
  const float temperature = kae::Temperature::get(gasState);

  const float goldTemperature{ 1.5f };
  const float threshold{ 1e-6f };
  EXPECT_NEAR(temperature, goldTemperature, threshold);
}

TEST(gas_state, gas_state_rotate_1_quadrant)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 5.0f, 5.0f, 12.0f, 2.0f };
  constexpr float nx{ 0.8f };
  constexpr float ny{ 0.6f };
  GasStateT rotatedGasState = kae::Rotate::get(gasState, nx, ny);

  GasStateT goldRotatedGasState{ 5.0f, 11.2f, 6.6f, 2.0f };

  constexpr float threshold{ 1e-6f };
  EXPECT_GAS_STATE_NEAR(rotatedGasState, goldRotatedGasState, threshold);
}

TEST(gas_state, gas_state_rotate_2_quadrant)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 5.0f, 5.0f, 12.0f, 2.0f };
  constexpr float nx{ -0.8f };
  constexpr float ny{ 0.6f };
  GasStateT rotatedGasState = kae::Rotate::get(gasState, nx, ny);

  GasStateT goldRotatedGasState{ 5.0f, 3.2f, -12.6f, 2.0f };

  constexpr float threshold{ 1e-6f };
  EXPECT_GAS_STATE_NEAR(rotatedGasState, goldRotatedGasState, threshold);
}

TEST(gas_state, gas_state_rotate_3_quadrant)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 5.0f, 5.0f, 12.0f, 2.0f };
  constexpr float nx{ -0.8f };
  constexpr float ny{ -0.6f };
  GasStateT rotatedGasState = kae::Rotate::get(gasState, nx, ny);

  GasStateT goldRotatedGasState{ 5.0f, -11.2f, -6.6f, 2.0f };

  constexpr float threshold{ 1e-6f };
  EXPECT_GAS_STATE_NEAR(rotatedGasState, goldRotatedGasState, threshold);
}

TEST(gas_state, gas_state_rotate_4_quadrant)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 5.0f, 5.0f, 12.0f, 2.0f };
  constexpr float nx{ 0.8f };
  constexpr float ny{ -0.6f };
  GasStateT rotatedGasState = kae::Rotate::get(gasState, nx, ny);

  GasStateT goldRotatedGasState{ 5.0f, -3.2f, 12.6f, 2.0f };

  constexpr float threshold{ 1e-6f };
  EXPECT_GAS_STATE_NEAR(rotatedGasState, goldRotatedGasState, threshold);
}

TEST(gas_state, gas_state_reverse_rotate_1_quadrant)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 5.0f, 11.2f, 6.6f, 2.0f };
  constexpr float nx{ 0.8f };
  constexpr float ny{ 0.6f };
  GasStateT rotatedGasState = kae::ReverseRotate::get(gasState, nx, ny);

  GasStateT goldRotatedGasState{ 5.0f, 5.0f, 12.0f, 2.0f };

  constexpr float threshold{ 1e-6f };
  EXPECT_GAS_STATE_NEAR(rotatedGasState, goldRotatedGasState, threshold);
}

TEST(gas_state, gas_state_reverse_rotate_2_quadrant)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 5.0f, 3.2f, -12.6f, 2.0f };
  constexpr float nx{ -0.8f };
  constexpr float ny{ 0.6f };
  GasStateT rotatedGasState = kae::ReverseRotate::get(gasState, nx, ny);

  GasStateT goldRotatedGasState{ 5.0f, 5.0f, 12.0f, 2.0f };

  constexpr float threshold{ 1e-6f };
  EXPECT_GAS_STATE_NEAR(rotatedGasState, goldRotatedGasState, threshold);
}

TEST(gas_state, gas_state_reverse_rotate_3_quadrant)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 5.0f, -11.2f, -6.6f, 2.0f };
  constexpr float nx{ -0.8f };
  constexpr float ny{ -0.6f };
  GasStateT rotatedGasState = kae::ReverseRotate::get(gasState, nx, ny);

  GasStateT goldRotatedGasState{ 5.0f, 5.0f, 12.0f, 2.0f };

  constexpr float threshold{ 1e-6f };
  EXPECT_GAS_STATE_NEAR(rotatedGasState, goldRotatedGasState, threshold);
}

TEST(gas_state, gas_state_reverse_rotate_4_quadrant)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 5.0f, -3.2f, 12.6f, 2.0f };
  constexpr float nx{ 0.8f };
  constexpr float ny{ -0.6f };
  GasStateT rotatedGasState = kae::ReverseRotate::get(gasState, nx, ny);

  GasStateT goldRotatedGasState{ 5.0f, 5.0f, 12.0f, 2.0f };

  constexpr float threshold{ 1e-6f };
  EXPECT_GAS_STATE_NEAR(rotatedGasState, goldRotatedGasState, threshold);
}

TEST(gas_state, gas_state_wave_speed_x)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 2.0f, 3.0f, 4.0f, 2.4f };
  const float sonicSpeed = kae::WaveSpeedX::get(gasState);

  const float goldSonicSpeed{ 4.2f };
  const float threshold{ 1e-6f };
  EXPECT_NEAR(sonicSpeed, goldSonicSpeed, threshold);
}

TEST(gas_state, gas_state_wave_speed_y)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 2.0f, 3.0f, 4.0f, 2.4f };
  const float sonicSpeed = kae::WaveSpeedY::get(gasState);

  const float goldSonicSpeed{ 5.2f };
  const float threshold{ 1e-6f };
  EXPECT_NEAR(sonicSpeed, goldSonicSpeed, threshold);
}

TEST(gas_state, gas_state_wave_speed)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 2.0f, 3.0f, 4.0f, 2.4f };
  const float sonicSpeed = kae::WaveSpeed::get(gasState);

  const float goldSonicSpeed{ 6.2f };
  const float threshold{ 1e-6f };
  EXPECT_NEAR(sonicSpeed, goldSonicSpeed, threshold);
}

TEST(gas_state, gas_state_mirror_state)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  const GasStateT gasState{ 5.0f, 5.0f, 12.0f, 2.0f };
  const GasStateT mirrorGasState = kae::MirrorState::get(gasState);

  const GasStateT goldMirrorGasState{ 5.0f, -5.0f, 12.0f, 2.0f };

  constexpr float threshold{ 1e-6f };
  EXPECT_GAS_STATE_NEAR(mirrorGasState, goldMirrorGasState, threshold);
}

TEST(gas_state, gas_state_conservative_variables)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 5.0f, 3.0f, 4.0f, 2.0f };
  const float4 conservativeVariables = kae::ConservativeVariables::get(gasState);

  const float4 goldConservativeVariables{ 5.0f, 15.0f, 20.0f, 72.5f };
  const float threshold{ 1e-6f };
  EXPECT_FLOAT4_NEAR(conservativeVariables, goldConservativeVariables, threshold);
}

TEST(gas_state, gas_state_x_fluxes)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 5.0f, 3.0f, 4.0f, 2.0f };
  const float4 xFluxes = kae::XFluxes::get(gasState);

  const float4 goldXFluxes{ 15.0f, 47.0f, 60.0f, 223.5f };
  const float threshold{ 1e-6f };
  EXPECT_FLOAT4_NEAR(xFluxes, goldXFluxes, threshold);
}

TEST(gas_state, gas_state_y_fluxes)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 5.0f, 3.0f, 4.0f, 2.0f };
  const float4 yFluxes = kae::YFluxes::get(gasState);

  const float4 goldYFluxes{ 20.0f, 60.0f, 82.0f, 298.0f };
  const float threshold{ 1e-6f };
  EXPECT_FLOAT4_NEAR(yFluxes, goldYFluxes, threshold);
}

TEST(gas_state, gas_state_source_term)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  GasStateT gasState{ 5.0f, 3.0f, 4.0f, 2.0f };
  const float4 sourceTerm = kae::SourceTerm::get(gasState);

  const float4 goldSourceTerm{ 20.0f, 60.0f, 80.0f, 298.0f };
  const float threshold{ 1e-6f };
  EXPECT_FLOAT4_NEAR(sourceTerm, goldSourceTerm, threshold);
}

TEST(gas_state, gas_state_conservative_to_gas_state)
{
  using KappaT = std::ratio<12, 10>;
  using CpT = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT>;

  constexpr float4 conservativeVariables{ 5.0f, 15.0f, 20.0f, 72.5f };
  const GasStateT gasState = kae::ConservativeToGasState::get<GasStateT>(conservativeVariables);

  const GasStateT goldGasState{ 5.0f, 3.0f, 4.0f, 2.0f };
  const float threshold{ 1e-6f };
  EXPECT_GAS_STATE_NEAR(gasState, goldGasState, threshold);
}

} // namespace tests
