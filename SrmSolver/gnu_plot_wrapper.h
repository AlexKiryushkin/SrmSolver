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

  void display2dPlot(const std::vector<std::vector<float>> & values);
private:

  std::shared_ptr<gnuplotio::Gnuplot> m_pImpl;
};

} // namespace kae
