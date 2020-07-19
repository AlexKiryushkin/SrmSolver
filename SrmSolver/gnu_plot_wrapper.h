#pragma once

#include "std_includes.h"

namespace gnuplotio {

class Gnuplot;

} // namespace gnuplotio

namespace kae {

class GnuPlotWrapper
{
public:
  explicit GnuPlotWrapper(const std::string & pathToGnuPlotExe);
  ~GnuPlotWrapper();

  void display2dPlot(const std::vector<std::vector<float>> & values);
  void display2dPlot(const std::vector<std::vector<double>>& values);
private:

  std::unique_ptr<gnuplotio::Gnuplot> m_pImpl;
};

} // namespace kae
