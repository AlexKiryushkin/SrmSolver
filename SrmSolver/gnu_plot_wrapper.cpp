#include "gnu_plot_wrapper.h"

#include "gnuplot-iostream.h"

namespace kae {

GnuPlotWrapper::GnuPlotWrapper(const std::string & pathToGnuPlotExe)
  : m_pImpl(std::make_unique<Gnuplot>(pathToGnuPlotExe))
{
}

GnuPlotWrapper::~GnuPlotWrapper()
{
}

void GnuPlotWrapper::display2dPlot(const std::vector<std::vector<float>> & values)
{
  auto && gp = *m_pImpl;

  gp << "plot '-' binary" << gp.binFmt2d(values, "array") << "with image\n";
  gp.sendBinary2d(values);
}

void GnuPlotWrapper::display2dPlot(const std::vector<std::vector<double>>& values)
{
  auto&& gp = *m_pImpl;

  gp << "plot '-' binary" << gp.binFmt2d(values, "array") << "with image\n";
  gp.sendBinary2d(values);
}

} // namespace kae 
