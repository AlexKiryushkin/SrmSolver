
#include <ratio>

#include <gtest/gtest.h>

#include <SrmSolver/cuda_float_types.h>
#include <SrmSolver/gas_state.h>

#include "comparators.h"

namespace tests {

template <class T>
class gas_state : public ::testing::Test {};

using TypeParams = ::testing::Types<float, double>;
TYPED_TEST_CASE(gas_state, TypeParams);

TYPED_TEST(gas_state, gas_state_fields)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(1.0),
                            static_cast<ElemType>(2.0),
                            static_cast<ElemType>(3.0),
                            static_cast<ElemType>(4.0) };

  constexpr ElemType goldKappa{ static_cast<ElemType>(1.2) };
  constexpr ElemType goldCp{    static_cast<ElemType>(6.0) };
  constexpr ElemType goldR{     static_cast<ElemType>(1.0) };
  constexpr ElemType goldRho{   static_cast<ElemType>(1.0) };
  constexpr ElemType goldUx{    static_cast<ElemType>(2.0) };
  constexpr ElemType goldUy{    static_cast<ElemType>(3.0) };
  constexpr ElemType goldP{     static_cast<ElemType>(4.0) };

  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_NEAR(GasStateT::kappa, goldKappa, threshold);
  EXPECT_NEAR(GasStateT::Cp, goldCp, threshold);
  EXPECT_NEAR(GasStateT::R, goldR, threshold);
  EXPECT_EQ(gasState.rho, goldRho);
  EXPECT_EQ(gasState.ux, goldUx);
  EXPECT_EQ(gasState.uy, goldUy);
  EXPECT_EQ(gasState.p, goldP);
}

TYPED_TEST(gas_state, gas_state_rho)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(1.0),
                            static_cast<ElemType>(3.0),
                            static_cast<ElemType>(4.0),
                            static_cast<ElemType>(2.0) };
  const ElemType rho = kae::Rho::get(gasState);

  constexpr ElemType goldRho{ static_cast<ElemType>(1.0) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_NEAR(rho, goldRho, threshold);
}

TYPED_TEST(gas_state, gas_state_p)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(1.0),
                            static_cast<ElemType>(3.0),
                            static_cast<ElemType>(4.0),
                            static_cast<ElemType>(2.0) };
  const ElemType p = kae::P::get(gasState);

  constexpr ElemType goldP{ static_cast<ElemType>(2.0) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_NEAR(p, goldP, threshold);
}

TYPED_TEST(gas_state, gas_state_velocity_squared)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(1.0),
                            static_cast<ElemType>(3.0),
                            static_cast<ElemType>(4.0),
                            static_cast<ElemType>(2.0) };
  const ElemType velocitySquared = kae::VelocitySquared::get(gasState);

  constexpr ElemType goldVelocitySquared{ static_cast<ElemType>(25.0) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_NEAR(velocitySquared, goldVelocitySquared, threshold);
}

TYPED_TEST(gas_state, gas_state_velocity)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(1.0),
                            static_cast<ElemType>(3.0),
                            static_cast<ElemType>(4.0),
                            static_cast<ElemType>(2.0) };
  const ElemType velocity = kae::Velocity::get(gasState);

  constexpr ElemType goldVelocity{ static_cast<ElemType>(5.0) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_NEAR(velocity, goldVelocity, threshold);
}

TYPED_TEST(gas_state, gas_state_mass_flux_x)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(5.0),
                            static_cast<ElemType>(3.0),
                            static_cast<ElemType>(4.0),
                            static_cast<ElemType>(2.0) };
  const ElemType massFluxX = kae::MassFluxX::get(gasState);

  constexpr ElemType goldMassFluxX{ static_cast<ElemType>(15.0) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_NEAR(massFluxX, goldMassFluxX, threshold);
}

TYPED_TEST(gas_state, gas_state_mass_flux_y)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(5.0),
                            static_cast<ElemType>(3.0),
                            static_cast<ElemType>(4.0),
                            static_cast<ElemType>(2.0) };
  const ElemType massFluxY = kae::MassFluxY::get(gasState);

  constexpr ElemType goldMassFluxY{ static_cast<ElemType>(20.0) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_NEAR(massFluxY, goldMassFluxY, threshold);
}

TYPED_TEST(gas_state, gas_state_momentum_flux_xx)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(5.0), 
                            static_cast<ElemType>(3.0),
                            static_cast<ElemType>(4.0),
                            static_cast<ElemType>(2.0) };
  const ElemType momentumFluxXx = kae::MomentumFluxXx::get(gasState);

  constexpr ElemType goldMomentumFluxXx{ static_cast<ElemType>(47.0) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_NEAR(momentumFluxXx, goldMomentumFluxXx, threshold);
}

TYPED_TEST(gas_state, gas_state_momentum_flux_xy)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(5.0),
                            static_cast<ElemType>(3.0),
                            static_cast<ElemType>(4.0),
                            static_cast<ElemType>(2.0) };
  const ElemType momentumFluxXy = kae::MomentumFluxXy::get(gasState);

  constexpr ElemType goldMomentumFluxXy{ static_cast<ElemType>(60.0) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_NEAR(momentumFluxXy, goldMomentumFluxXy, threshold);
}

TYPED_TEST(gas_state, gas_state_momentum_flux_yy)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(5.0),
                            static_cast<ElemType>(3.0),
                            static_cast<ElemType>(4.0),
                            static_cast<ElemType>(2.0) };
  const ElemType momentumFluxYy = kae::MomentumFluxYy::get(gasState);

  constexpr ElemType goldMomentumFluxYy{ static_cast<ElemType>(82.0) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_NEAR(momentumFluxYy, goldMomentumFluxYy, threshold);
}

TYPED_TEST(gas_state, gas_state_rho_energy)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(5.0),
                            static_cast<ElemType>(3.0),
                            static_cast<ElemType>(4.0),
                            static_cast<ElemType>(2.0) };
  const ElemType rhoEnergy = kae::RhoEnergy::get(gasState);

  constexpr ElemType goldRhoEnergy{ static_cast<ElemType>(72.5) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_NEAR(rhoEnergy, goldRhoEnergy, threshold);
}

TYPED_TEST(gas_state, gas_state_energy)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(5.0),
                            static_cast<ElemType>(3.0),
                            static_cast<ElemType>(4.0),
                            static_cast<ElemType>(2.0) };
  const ElemType energy = kae::Energy::get(gasState);

  constexpr ElemType goldEnergy{ static_cast<ElemType>(14.5) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_NEAR(energy, goldEnergy, threshold);
}

TYPED_TEST(gas_state, gas_state_enthalpy_flux_x)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(5.0),
                            static_cast<ElemType>(3.0),
                            static_cast<ElemType>(4.0),
                            static_cast<ElemType>(2.0) };
  const ElemType enthalpyFluxX = kae::EnthalpyFluxX::get(gasState);

  constexpr ElemType goldEnthalpyFluxX{ static_cast<ElemType>(223.5) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_NEAR(enthalpyFluxX, goldEnthalpyFluxX, threshold);
}

TYPED_TEST(gas_state, gas_state_enthalpy_flux_y)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(5.0),
                            static_cast<ElemType>(3.0),
                            static_cast<ElemType>(4.0),
                            static_cast<ElemType>(2.0) };
  const ElemType enthalpyFluxY = kae::EnthalpyFluxY::get(gasState);

  constexpr ElemType goldEnthalpyFluxY{ static_cast<ElemType>(298.0) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_NEAR(enthalpyFluxY, goldEnthalpyFluxY, threshold);
}

TYPED_TEST(gas_state, gas_state_sonic_speed_squared)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  GasStateT gasState{ static_cast<ElemType>(2.0),
                      static_cast<ElemType>(3.0),
                      static_cast<ElemType>(4.0),
                      static_cast<ElemType>(2.4) };
  const ElemType sonicSpeedSquared = kae::SonicSpeedSquared::get(gasState);

  constexpr ElemType goldSonicSpeedSquared{ static_cast<ElemType>(1.44) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_NEAR(sonicSpeedSquared, goldSonicSpeedSquared, threshold);
}

TYPED_TEST(gas_state, gas_state_sonic_speed)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(2.0),
                            static_cast<ElemType>(3.0),
                            static_cast<ElemType>(4.0),
                            static_cast<ElemType>(2.4) };
  const ElemType sonicSpeed = kae::SonicSpeed::get(gasState);

  constexpr ElemType goldSonicSpeed{ static_cast<ElemType>(1.2) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_NEAR(sonicSpeed, goldSonicSpeed, threshold);
}

TYPED_TEST(gas_state, gas_state_mach)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(2.0),
                            static_cast<ElemType>(3.0),
                            static_cast<ElemType>(4.0),
                            static_cast<ElemType>(2.4) };
  const ElemType mach = kae::Mach::get(gasState);

  constexpr ElemType goldMach{ static_cast<ElemType>(5.0 / 1.2) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) : static_cast<ElemType>(1e-14) };
  EXPECT_NEAR(mach, goldMach, threshold);
}

TYPED_TEST(gas_state, gas_state_temperature)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<12, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(2.0),
                            static_cast<ElemType>(3.0),
                            static_cast<ElemType>(4.0),
                            static_cast<ElemType>(6.0) };
  const ElemType temperature = kae::Temperature::get(gasState);

  constexpr ElemType goldTemperature{ static_cast<ElemType>(1.5) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_NEAR(temperature, goldTemperature, threshold);
}

TYPED_TEST(gas_state, gas_state_rotate_1_quadrant)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(5.0),
                            static_cast<ElemType>(5.0),
                            static_cast<ElemType>(12.0),
                            static_cast<ElemType>(2.0) };
  constexpr ElemType nx{ static_cast<ElemType>(0.8) };
  constexpr ElemType ny{ static_cast<ElemType>(0.6) };
  const GasStateT rotatedGasState = kae::Rotate::get(gasState, nx, ny);

  const GasStateT goldRotatedGasState{ static_cast<ElemType>(5.0),
                                       static_cast<ElemType>(11.2),
                                       static_cast<ElemType>(6.6),
                                       static_cast<ElemType>(2.0) };

  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_GAS_STATE_NEAR(rotatedGasState, goldRotatedGasState, threshold);
}

TYPED_TEST(gas_state, gas_state_rotate_2_quadrant)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(5.0),
                            static_cast<ElemType>(5.0),
                            static_cast<ElemType>(12.0),
                            static_cast<ElemType>(2.0) };
  constexpr ElemType nx{ static_cast<ElemType>(-0.8) };
  constexpr ElemType ny{ static_cast<ElemType>(0.6)};
  const GasStateT rotatedGasState = kae::Rotate::get(gasState, nx, ny);

  const GasStateT goldRotatedGasState{ static_cast<ElemType>(5.0),
                                       static_cast<ElemType>(3.2),
                                       static_cast<ElemType>(-12.6),
                                       static_cast<ElemType>(2.0) };

  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_GAS_STATE_NEAR(rotatedGasState, goldRotatedGasState, threshold);
}

TYPED_TEST(gas_state, gas_state_rotate_3_quadrant)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(5.0),
                            static_cast<ElemType>(5.0),
                            static_cast<ElemType>(12.0),
                            static_cast<ElemType>(2.0) };
  constexpr ElemType nx{ static_cast<ElemType>(-0.8) };
  constexpr ElemType ny{ static_cast<ElemType>(-0.6) };
  const GasStateT rotatedGasState = kae::Rotate::get(gasState, nx, ny);

  const GasStateT goldRotatedGasState{ static_cast<ElemType>(5.0),
                                       static_cast<ElemType>(-11.2),
                                       static_cast<ElemType>(-6.6),
                                       static_cast<ElemType>(2.0) };

  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_GAS_STATE_NEAR(rotatedGasState, goldRotatedGasState, threshold);
}

TYPED_TEST(gas_state, gas_state_rotate_4_quadrant)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(5.0),
                            static_cast<ElemType>(5.0),
                            static_cast<ElemType>(12.0),
                            static_cast<ElemType>(2.0) };
  constexpr ElemType nx{ static_cast<ElemType>(0.8) };
  constexpr ElemType ny{ static_cast<ElemType>(-0.6) };
  const GasStateT rotatedGasState = kae::Rotate::get(gasState, nx, ny);

  const GasStateT goldRotatedGasState{ static_cast<ElemType>(5.0),
                                       static_cast<ElemType>(-3.2),
                                       static_cast<ElemType>(12.6),
                                       static_cast<ElemType>(2.0) };

  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_GAS_STATE_NEAR(rotatedGasState, goldRotatedGasState, threshold);
}

TYPED_TEST(gas_state, gas_state_reverse_rotate_1_quadrant)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(5.0),
                            static_cast<ElemType>(11.2),
                            static_cast<ElemType>(6.6),
                            static_cast<ElemType>(2.0) };
  constexpr ElemType nx{ static_cast<ElemType>(0.8) };
  constexpr ElemType ny{ static_cast<ElemType>(0.6) };
  const GasStateT rotatedGasState = kae::ReverseRotate::get(gasState, nx, ny);

  const GasStateT goldRotatedGasState{ static_cast<ElemType>(5.0),
                                       static_cast<ElemType>(5.0),
                                       static_cast<ElemType>(12.0), 
                                       static_cast<ElemType>(2.0) };

  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_GAS_STATE_NEAR(rotatedGasState, goldRotatedGasState, threshold);
}

TYPED_TEST(gas_state, gas_state_reverse_rotate_2_quadrant)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(5.0),
                            static_cast<ElemType>(3.2),
                            static_cast<ElemType>(-12.6),
                            static_cast<ElemType>(2.0) };
  constexpr ElemType nx{ static_cast<ElemType>(-0.8) };
  constexpr ElemType ny{ static_cast<ElemType>(0.6) };
  const GasStateT rotatedGasState = kae::ReverseRotate::get(gasState, nx, ny);

  const GasStateT goldRotatedGasState{ static_cast<ElemType>(5.0),
                                       static_cast<ElemType>(5.0),
                                       static_cast<ElemType>(12.0),
                                       static_cast<ElemType>(2.0) };

  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_GAS_STATE_NEAR(rotatedGasState, goldRotatedGasState, threshold);
}

TYPED_TEST(gas_state, gas_state_reverse_rotate_3_quadrant)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(5.0),
                            static_cast<ElemType>(-11.2),
                            static_cast<ElemType>(-6.6),
                            static_cast<ElemType>(2.0) };
  constexpr ElemType nx{ static_cast<ElemType>(-0.8) };
  constexpr ElemType ny{ static_cast<ElemType>(-0.6) };
  const GasStateT rotatedGasState = kae::ReverseRotate::get(gasState, nx, ny);

  const GasStateT goldRotatedGasState{ static_cast<ElemType>(5.0),
                                       static_cast<ElemType>(5.0),
                                       static_cast<ElemType>(12.0),
                                       static_cast<ElemType>(2.0) };

  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_GAS_STATE_NEAR(rotatedGasState, goldRotatedGasState, threshold);
}

TYPED_TEST(gas_state, gas_state_reverse_rotate_4_quadrant)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(5.0),
                            static_cast<ElemType>(-3.2),
                            static_cast<ElemType>(12.6),
                            static_cast<ElemType>(2.0) };
  constexpr ElemType nx{ static_cast<ElemType>(0.8) };
  constexpr ElemType ny{ static_cast<ElemType>(-0.6) };
  const GasStateT rotatedGasState = kae::ReverseRotate::get(gasState, nx, ny);

  const GasStateT goldRotatedGasState{ static_cast<ElemType>(5.0),
                                       static_cast<ElemType>(5.0),
                                       static_cast<ElemType>(12.0),
                                       static_cast<ElemType>(2.0) };

  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_GAS_STATE_NEAR(rotatedGasState, goldRotatedGasState, threshold);
}

TYPED_TEST(gas_state, gas_state_wave_speed_x)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(2.0),
                            static_cast<ElemType>(3.0),
                            static_cast<ElemType>(4.0),
                            static_cast<ElemType>(2.4) };
  const ElemType sonicSpeed = kae::WaveSpeedX::get(gasState);

  constexpr ElemType goldSonicSpeed{ static_cast<ElemType>(4.2) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_NEAR(sonicSpeed, goldSonicSpeed, threshold);
}

TYPED_TEST(gas_state, gas_state_wave_speed_y)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(2.0),
                            static_cast<ElemType>(3.0),
                            static_cast<ElemType>(4.0),
                            static_cast<ElemType>(2.4) };
  const ElemType sonicSpeed = kae::WaveSpeedY::get(gasState);

  constexpr ElemType goldSonicSpeed{ static_cast<ElemType>(5.2) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_NEAR(sonicSpeed, goldSonicSpeed, threshold);
}

TYPED_TEST(gas_state, gas_state_wave_speed)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(2.0),
                            static_cast<ElemType>(3.0),
                            static_cast<ElemType>(4.0),
                            static_cast<ElemType>(2.4) };
  const ElemType sonicSpeed = kae::WaveSpeed::get(gasState);

  constexpr ElemType goldSonicSpeed{ static_cast<ElemType>(6.2) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_NEAR(sonicSpeed, goldSonicSpeed, threshold);
}

TYPED_TEST(gas_state, gas_state_mirror_state)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(5.0),
                            static_cast<ElemType>(5.0), 
                            static_cast<ElemType>(12.0),
                            static_cast<ElemType>(2.0) };
  const GasStateT mirrorGasState = kae::MirrorState::get(gasState);

  const GasStateT goldMirrorGasState{ static_cast<ElemType>(5.0),
                                      static_cast<ElemType>(-5.0),
                                      static_cast<ElemType>(12.0),
                                      static_cast<ElemType>(2.0) };

  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_GAS_STATE_NEAR(mirrorGasState, goldMirrorGasState, threshold);
}

TYPED_TEST(gas_state, gas_state_conservative_variables)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(5.0),
                            static_cast<ElemType>(3.0),
                            static_cast<ElemType>(4.0),
                            static_cast<ElemType>(2.0) };
  const kae::CudaFloatT<4U, ElemType> conservativeVariables = kae::ConservativeVariables::get(gasState);

  constexpr kae::CudaFloatT<4U, ElemType> goldConservativeVariables{ static_cast<ElemType>(5.0),
                                                                     static_cast<ElemType>(15.0),
                                                                     static_cast<ElemType>(20.0),
                                                                     static_cast<ElemType>(72.5) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_FLOAT4_NEAR(conservativeVariables, goldConservativeVariables, threshold);
}

TYPED_TEST(gas_state, gas_state_x_fluxes)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(5.0),
                            static_cast<ElemType>(3.0),
                            static_cast<ElemType>(4.0),
                            static_cast<ElemType>(2.0) };
  const kae::CudaFloatT<4U, ElemType> xFluxes = kae::XFluxes::get(gasState);

  constexpr kae::CudaFloatT<4U, ElemType> goldXFluxes{ static_cast<ElemType>(15.0),
                                                       static_cast<ElemType>(47.0),
                                                       static_cast<ElemType>(60.0),
                                                       static_cast<ElemType>(223.5) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_FLOAT4_NEAR(xFluxes, goldXFluxes, threshold);
}

TYPED_TEST(gas_state, gas_state_y_fluxes)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(5.0),
                            static_cast<ElemType>(3.0),
                            static_cast<ElemType>(4.0),
                            static_cast<ElemType>(2.0) };
  const kae::CudaFloatT<4U, ElemType> yFluxes = kae::YFluxes::get(gasState);

  constexpr kae::CudaFloatT<4U, ElemType> goldYFluxes{ static_cast<ElemType>(20.0),
                                                       static_cast<ElemType>(60.0),
                                                       static_cast<ElemType>(82.0),
                                                       static_cast<ElemType>(298.0) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_FLOAT4_NEAR(yFluxes, goldYFluxes, threshold);
}

TYPED_TEST(gas_state, gas_state_source_term)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(5.0),
                            static_cast<ElemType>(3.0),
                            static_cast<ElemType>(4.0),
                            static_cast<ElemType>(2.0) };
  const kae::CudaFloatT<4U, ElemType> sourceTerm = kae::SourceTerm::get(gasState);

  constexpr kae::CudaFloatT<4U, ElemType> goldSourceTerm{ static_cast<ElemType>(20.0),
                                                          static_cast<ElemType>(60.0),
                                                          static_cast<ElemType>(80.0),
                                                          static_cast<ElemType>(298.0) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_FLOAT4_NEAR(sourceTerm, goldSourceTerm, threshold);
}

TYPED_TEST(gas_state, gas_state_conservative_to_gas_state)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  constexpr kae::CudaFloatT<4U, ElemType> conservativeVariables{ static_cast<ElemType>(5.0),
                                                                 static_cast<ElemType>(15.0),
                                                                 static_cast<ElemType>(20.0),
                                                                 static_cast<ElemType>(72.5) };
  const GasStateT gasState = kae::ConservativeToGasState::get<GasStateT>(conservativeVariables);

  const GasStateT goldGasState{ static_cast<ElemType>(5.0),
                                static_cast<ElemType>(3.0),
                                static_cast<ElemType>(4.0),
                                static_cast<ElemType>(2.0) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_GAS_STATE_NEAR(gasState, goldGasState, threshold);
}

TYPED_TEST(gas_state, gas_state_eigen_values_x)
{
  using ElemType  = TypeParam;
  using KappaT    = std::ratio<12, 10>;
  using CpT       = std::ratio<6, 1>;
  using GasStateT = kae::GasState<KappaT, CpT, ElemType>;

  const GasStateT gasState{ static_cast<ElemType>(2.0),
                            static_cast<ElemType>(3.0),
                            static_cast<ElemType>(4.0),
                            static_cast<ElemType>(2.4) };
  const kae::CudaFloatT<4U, ElemType> eigenValuesX = kae::EigenValuesX::get(gasState);

  constexpr kae::CudaFloatT<4U, ElemType> goldEigenValuesX{ static_cast<ElemType>(1.8),
                                                            static_cast<ElemType>(3.0),
                                                            static_cast<ElemType>(3.0),
                                                            static_cast<ElemType>(4.2) };
  constexpr ElemType threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-6) :
                                                                       static_cast<ElemType>(1e-14) };
  EXPECT_FLOAT4_NEAR(eigenValuesX, goldEigenValuesX, threshold);
}

} // namespace tests
