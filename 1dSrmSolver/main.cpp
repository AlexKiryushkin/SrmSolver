
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <numeric>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

struct GasState
{
  constexpr static double kappa = 1.23;

  double rho;
  double u;
  double p;
  double r;
};

struct MirrorState
{
  static GasState get(const GasState& gasState)
  {
    return GasState{ gasState.rho, -gasState.u, gasState.p, gasState.r };
  }
};

struct SonicSpeed
{
  static double get(const GasState& state)
  {
    return std::sqrt(GasState::kappa * state.p / state.rho);
  }
};

struct U
{
  static double get(const GasState& gasState)
  {
    return gasState.u;
  }
};

struct P
{
  static double get(const GasState& gasState)
  {
    return gasState.p;
  }
};

struct WaveSpeed
{
  auto operator()(const GasState& state) const { return get(state); }
  static double get(const GasState& state)
  {
    return SonicSpeed::get(state) + std::fabs(U::get(state));
  }
};

struct S
{
  static double get(const GasState& state)
  {
    return state.r * state.r;
  }
};

struct RhoS
{
  static double get(const GasState& gasState)
  {
    return gasState.rho * S::get(gasState);
  }
};

struct MassFlux
{
  static double get(const GasState& gasState)
  {
    return gasState.rho * gasState.u;
  }
};

struct RhoEnergyFlux
{
  static double get(const GasState& gasState)
  {
    constexpr double multiplier = 1.0 / (GasState::kappa - 1.0);
    return multiplier * gasState.p + 0.5 * gasState.rho * gasState.u * gasState.u;
  }
};

struct RhoEnthalpyFlux
{
  static double get(const GasState& gasState)
  {
    return gasState.u * (RhoEnergyFlux::get(gasState) + gasState.p);
  }
};

struct MomentumFlux
{
  static double get(const GasState& gasState)
  {
    return gasState.rho * gasState.u * gasState.u + gasState.p;
  }
};

struct MassFluxS
{
  static double get(const GasState& gasState)
  {
    return MassFlux::get(gasState) * S::get(gasState);
  }
};

struct RhoEnergyFluxS
{
  static double get(const GasState& gasState)
  {
    return RhoEnergyFlux::get(gasState) * S::get(gasState);
  }
};

struct MomentumFluxS
{
  static double get(const GasState& gasState)
  {
    return MomentumFlux::get(gasState) * S::get(gasState);
  }
};

struct RhoEnthalpyFluxS
{
  static double get(const GasState& gasState)
  {
    return RhoEnthalpyFlux::get(gasState) * S::get(gasState);
  }
};

struct Mach
{
  static double get(const GasState& state)
  {
    return std::fabs(state.u) / SonicSpeed::get(state);
  }
};

template <class U, class F, unsigned Step = 1U>
double getFlux(const GasState* pState, std::size_t idx, double lambda)
{
  constexpr double half = 0.5;

  const double fPlus = half * (F::get(pState[idx]) + lambda * U::get(pState[idx]));
  const double fMinus = half * (F::get(pState[idx + Step]) - lambda * U::get(pState[idx + Step]));

  return fPlus + fMinus;
}

double getMaxRadius(double x)
{
  constexpr auto pointSize = 12U;
  constexpr static double points[pointSize][2] = {
    { (0.01),  (0.081)  },
    { (0.02),  (0.081)  },
    { (0.055), (0.0922) },
    { (0.102), (0.0922) },
    { (0.105), (0.0922) },
    { (0.145), (0.0922) },
    { (0.691), (0.0922) },
    { (1.045), (0.0922) },
    { (1.087), (0.081)  },
    { (1.139), (0.081)  },
    { (1.184), (0.0669) },
    { (1.274), (0.0872) }
  };

  for (std::size_t pointIdx{}; pointIdx < pointSize - 1; ++pointIdx)
  {
    const auto leftX = points[pointIdx][0];
    const auto rightX = points[pointIdx + 1][0];
    if (x >= leftX && x <= rightX)
    {
      const auto leftY = points[pointIdx][1];
      const auto rightY = points[pointIdx + 1][1];

      return leftY + (x - leftX) / (rightX - leftX) * (rightY - leftY);
    }
  }
}

std::vector<double> getRadii(std::size_t nPoints, double h)
{
  constexpr auto pointSize = 11U;
  constexpr static double points[pointSize][2] = {
    { 0.0,   0.04 },
    { 0.01,  0.031 },
    { 0.092, 0.031 },
    { 0.095, 0.0245 },
    { 0.135, 0.0245 },
    { 0.681, 0.026 },
    { 1.039, 0.03 },
    { 1.077, 0.03 },
    { 1.139, 0.03 },
    { 1.184, 0.044 },
    { 1.264, 0.069 }
  };

  std::vector<double> radiiValues(nPoints + 3);
  for (std::size_t idx{ 1U }; idx <= nPoints + 1; ++idx)
  {
    const auto x = (idx - 1U) * h;
    for (std::size_t pointIdx{}; pointIdx < pointSize - 1; ++pointIdx)
    {
      const auto leftX = points[pointIdx][0];
      const auto rightX = points[pointIdx + 1][0];
      if (x >= leftX && x <= rightX)
      {
        const auto leftY = points[pointIdx][1];
        const auto rightY = points[pointIdx + 1][1];

        radiiValues[idx] = leftY + (x - leftX) / (rightX - leftX) * (rightY - leftY);
        break;
      }
    }
  }
  return radiiValues;
}

std::vector<std::tuple<double, double, double, double>> solve(std::size_t nPoints)
{
  constexpr auto length        = 1.264;
  const auto h = length / static_cast<double>(nPoints);

  constexpr auto nu            = 0.41;
  constexpr auto mt            = 0.0052381;
  constexpr auto P0            = 0.00601628;
  constexpr auto H0            = 5.34783;
  constexpr auto rhoP          = 100.225;

  std::vector<double> prevRadii = getRadii(nPoints, h);

  std::vector<GasState> prevGasValues(nPoints + 2, { 0.0153947, 0.0, 0.0153947, 1.0 });
  std::vector<GasState> currGasValues(nPoints + 2, { 0.0153947, 0.0, 0.0153947, 1.0 });
  for (std::size_t idx{ 1U }; idx <= nPoints; ++idx)
  {
    const auto r = (prevRadii.at(idx) + prevRadii.at(idx + 1)) / 2;
    prevGasValues.at(idx).r = r;
    currGasValues.at(idx).r = r;
  }

  std::vector<std::tuple<double, double, double, double>> integralValues;
  double maxT{ 2600.0 };
  double writeDeltaT{ 5.0 };
  double writeT{ writeDeltaT };
  double t{};
  while (t < maxT)
  {
    std::swap(prevGasValues, currGasValues);

    prevGasValues.at(0U) = MirrorState::get(prevGasValues.at(1U));
    const auto& closestGasState = prevGasValues.at(nPoints);
    const auto c = SonicSpeed::get(closestGasState);
    if (closestGasState.u >= c)
    {
      prevGasValues.at(nPoints + 1U) = closestGasState;
    }
    else
    {
      prevGasValues.at(nPoints + 1U) = GasState{
        closestGasState.rho - 1 / c / c * (closestGasState.p - P0),
        closestGasState.u + 1 / closestGasState.rho / c * (closestGasState.p - P0),
        P0,
        closestGasState.r };
    }

    constexpr auto courant = 0.8;
    const auto lambda = std::accumulate(std::begin(prevGasValues), std::end(prevGasValues), 0.0, 
      [](double currLambda, const GasState & gasState)
      {
        return std::max(currLambda, WaveSpeed::get(gasState));
      });
    const auto dt = courant * h / lambda;

    for (std::size_t idx{ 1U }; idx <= nPoints; ++idx)
    {
      const auto x = (idx - 0.5) * h;
      const auto & prevGasValue = prevGasValues.at(idx);

      const auto massFluxLeft = getFlux<RhoS, MassFluxS>(prevGasValues.data(), idx - size_t(1U), lambda);
      const auto massFluxRight = getFlux<RhoS, MassFluxS>(prevGasValues.data(), idx, lambda);

      const auto momentumFluxLeft = getFlux<MassFluxS, MomentumFluxS>(prevGasValues.data(), idx - size_t(1U), lambda);
      const auto momentumFluxRight = getFlux<MassFluxS, MomentumFluxS>(prevGasValues.data(), idx, lambda);

      const auto enthalpyFluxLeft = getFlux<RhoEnergyFluxS, RhoEnthalpyFluxS>(prevGasValues.data(), idx - size_t(1U), lambda);
      const auto enthalpyFluxRight = getFlux<RhoEnergyFluxS, RhoEnthalpyFluxS>(prevGasValues.data(), idx, lambda);

      const auto prevRPrime = (prevGasValues.at(idx + 1).r - prevGasValues.at(idx - 1).r) / 2 / h;
      const auto maxR = getMaxRadius(x);
      const auto prevR = prevGasValue.r;
      const auto mtpnu = (prevR > maxR) ? 0.0 : mt * std::pow(P::get(prevGasValue), nu);
      const auto newR = prevR + dt * mtpnu / rhoP;
      const auto newRho =
        (RhoS::get(prevGasValue) -
        dt / h * (massFluxRight - massFluxLeft) +
        dt * 2  * prevR * mtpnu) / newR / newR;
      const auto newRhoU = 
        (MassFluxS::get(prevGasValue) - 
        dt / h * (momentumFluxRight - momentumFluxLeft) +
        dt * 2 * prevRPrime * prevR * P::get(prevGasValue)) / newR / newR;
      const auto newRhoE = 
        (RhoEnergyFluxS::get(prevGasValue) - 
        dt / h * (enthalpyFluxRight - enthalpyFluxLeft) +
        dt * 2 * prevR * mtpnu * H0) / newR / newR;
      const auto newU = newRhoU / newRho;
      const auto newP = (GasState::kappa - 1) * (newRhoE - newRho * newU * newU / 2);
      currGasValues.at(idx) = GasState{ newRho, newU, newP, newR };
    }
    t += dt;

    if (t > writeT)
    {
      writeT += writeDeltaT;

      double sBurn{};
      double maxP{};
      for (std::size_t idx{ 1U }; idx <= nPoints; ++idx)
      {
        auto&& gasState = currGasValues[idx];

        maxP = std::max(maxP, gasState.p);
        sBurn += 2 * M_PI * gasState.r * h;
      }
      double thrust = M_PI * MomentumFluxS::get(currGasValues[nPoints]);
      integralValues.emplace_back(t, maxP, sBurn, thrust);
    }
  }

  return integralValues;
}

int main()
{
  constexpr auto nPoints{ 400U };
  const auto start = std::chrono::high_resolution_clock::now();
  const auto values = solve(nPoints);
  const auto end = std::chrono::high_resolution_clock::now();
  std::cout << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "\n";

  std::ofstream outputFile{ "integral_data.dat" };
  outputFile << "t;maxP;sBurn;thrust\n";
  for (auto & tuple : values)
  {
    outputFile << std::get<0U>(tuple) << ";"
               << std::get<1U>(tuple) << ";"
               << std::get<2U>(tuple) << ";"
               << std::get<3U>(tuple) << "\n";
  }
}