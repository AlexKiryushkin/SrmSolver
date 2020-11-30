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

void GnuPlotWrapper::display2dPlot(const std::vector<std::vector<std::tuple<float, float, float>>>& values)
{
  auto && gp = *m_pImpl;

  gp << "set terminal wxt size 1280, 660\n";
  gp << "set title \"Pressure values\"\n";
  gp << "plot '-' binary" << gp.binFmt2d(values, "record")
    << " with image\n";

  gp.sendBinary2d(values);
}

void GnuPlotWrapper::display2dPlot(const std::vector<std::vector<std::tuple<double, double, double>>>& values)
{
  auto&& gp = *m_pImpl;
  auto plots = gnuplotio::Gnuplot::splotGroup();
  plots.add_plot2d(values, "with lines title 'vec of vec of std::tuple'");
  gp << plots;
  //gp << "plot '-' binary" << gp.binFmt2d(values, "array") << "with image\n";
  //gp.sendBinary2d(values);
}

} // namespace kae 
