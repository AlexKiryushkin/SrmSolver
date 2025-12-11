
#pragma warning(push, 0)

#include <iostream>

#include <boost/foreach.hpp>
#include <boost/geometry.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#pragma warning(pop)

#include "filesystem.h"
#include "gpu_level_set_solver.h"
#include "gpu_srm_solver.h"
#include "shape/shape_analyzer.h"
#include "solver_callbacks.h"

namespace pt = boost::property_tree;

using ElemType = float;
using LevelSetSolverType = kae::GpuLevelSetSolver<ElemType>;
using SrmSolverType = kae::GpuSrmSolver<kae::GasState<ElemType>>;

using Point2d = boost::geometry::model::d2::point_xy<ElemType>;
using Polygon2d = boost::geometry::model::polygon<Point2d>;
using Linestring2d = boost::geometry::model::linestring<Point2d>;

Linestring2d readLinestring(const pt::ptree& root, const std::string& childName)
{
    Linestring2d shapePoints;
    for (const auto& point : root.get_child(childName)) {
        ElemType x = point.second.get<ElemType>("x");
        ElemType y = point.second.get<ElemType>("y");
        shapePoints.emplace_back(x, y);
    }

    return shapePoints;
}

int main()
{
    try
    {
        pt::ptree root;
        try
        {
            std::cout << "******************************************\n";
            std::cout << "Reading config.json...\n";
            pt::read_json("config.json", root);
            std::cout << "Reading is done\n";
        }
        catch (std::exception& e)
        {
            std::cerr << "Failed loading config.json. Reason: " << e.what();
            std::cin.get();
            return 1;
        }
        catch (...)
        {
            std::cerr << "Failed loading config.json. Unknown reason.";
            std::cin.get();
            return 1;
        }

        std::cout << "******************************************\n";
        std::cout << "Parsing physical parameters...\n";
        const auto& physical = root.get_child("physical_parameters");
        ElemType nu = physical.get<ElemType>("nu");
        ElemType mt = physical.get<ElemType>("mt");
        ElemType TBurn = physical.get<ElemType>("TBurn");
        ElemType rhoP = physical.get<ElemType>("rhoP");
        ElemType P0 = physical.get<ElemType>("P0");
        ElemType kappa = physical.get<ElemType>("kappa");
        ElemType cp = physical.get<ElemType>("cp");

        std::cout << "Parsing numerical parameters...\n";
        const auto& numerical = root.get_child("numerical_parameters");
        ElemType h = numerical.get<ElemType>("h");

        std::cout << "Parsing geometry...\n";
        Linestring2d initialShapePoints = readLinestring(root, "initial_srm_shape");
        Linestring2d propellantPoints = readLinestring(root, "propellant_shape");

        // {
        //     std::ofstream svg_file("output.svg");
        //     boost::geometry::svg_mapper<Point2d> mapper(svg_file, 400, 400, 1.5); // 400x400 SVG canvas
        //     mapper.add(initialShapePoints);
        //     mapper.map(initialShapePoints, "fill:lightgreen;stroke:lightgreen;stroke-width:1;");
        // 
        //     mapper.add(propellantPoints);
        //     mapper.map(propellantPoints, "fill:darkgreen;stroke:darkgreen;stroke-width:2;");
        // }

        kae::ShapeAnalyzer shapeAnalyzer{ initialShapePoints, propellantPoints };

        std::cout << "Parsing post-processing parameters...\n";
        const auto &postProcessing = root.get_child("post_processing");
        const auto writeDt = postProcessing.get<ElemType>("write_dt");
        const auto outputFolder = postProcessing.get<std::string>("output_folder");
        const auto integrationType = postProcessing.get<std::string>("integration_type");

        std::cout << "Parsing is done\n";

        std::cout << "******************************************\n";
        std::cout << "Initialize solvers...\n";
        std::cout << "Calculating grid...\n";
        const auto grid = shapeAnalyzer.calculateGrid(h);

        std::cout << "Building GPU shape...\n";
        const auto newShape = shapeAnalyzer.buildShape(grid);

        std::cout << "Creating level set solver...\n";
        LevelSetSolverType levelSetSolver{ grid, shapeAnalyzer.getSignedDistances(grid), newShape };

        std::cout << "Creating results writer...\n";
        const std::wstring currentPath = kae::append(kae::current_path(), outputFolder);
        kae::WriteToFolderCallback<ElemType> callback{ currentPath };

        std::cout << "Calculating physcial properties...\n";
        kae::PhysicalPropertiesData<ElemType> physicalProperties(nu, mt, TBurn, rhoP, P0, kappa, cp, newShape.getFCritical(), shapeAnalyzer.getInitialSBurn());

        std::cout << "Creating SRM sovler...\n";
        kae::GasState<ElemType> initialGasState{ static_cast<ElemType>(0.5), static_cast<ElemType>(0.0), static_cast<ElemType>(0.0), physicalProperties.P0 };
        kae::GasParameters<ElemType> gasParameters{ physicalProperties.kappa, physicalProperties.R };
        SrmSolverType srmSolver{ grid, physicalProperties, levelSetSolver, newShape, initialGasState, gasParameters, static_cast<ElemType>(0.8) };
        std::cout << "Initialization is done!\n";

        std::cout << physicalProperties;
        std::cout << grid;

        std::cout << "******************************************\n";
        std::cout << "Geometry shape. \n";
        std::cout << "Initial burning surface area is: " << shapeAnalyzer.getInitialSBurn() << " m^2\n";
        std::cout << "Critical section is found at x = " << shapeAnalyzer.findCriticalPoint().get<0>() << "\n";
        std::cout << "Critical section surface area is: " << newShape.getFCritical() << " m^2\n";
        std::cout << "Outlet coordinate is found at x = " << newShape.getOutletCoordinate() << "\n";

        const auto burnRate = kae::BurningRate::get(static_cast<ElemType>(1), physicalProperties.nu, physicalProperties.mt, physicalProperties.rhoP);
        const auto dt = grid.hx / 2 / burnRate;
        const auto writeDtUndim = writeDt * physicalProperties.uScale;
        std::cout << "Level set time step integration is: " << dt / physicalProperties.uScale << '\n';
        std::cout << "Write to fodler time is: " << writeDtUndim << '\n';
        std::cout << "Type of integration is: " << integrationType << '\n';

        std::cout << "******************************************\n";
        std::cout << "Start integration\n";
        if (integrationType == "quasi-dynamic")
        {
            srmSolver.quasiStationaryDynamicIntegrate(2000U, dt, writeDtUndim, kae::ETimeDiscretizationOrder::eTwo, callback);
        }
        else
        {
            srmSolver.dynamicIntegrate(2000U, dt, writeDtUndim, kae::ETimeDiscretizationOrder::eTwo, callback);
        }
        std::cout << "Integration finished\n";
        std::cin.get();
    }
    catch (const std::exception& e)
    {
        std::cout << e.what() << '\n';
        std::cin.get();
    }
    catch (...)
    {
        std::cout << "Unknown exception caught\n";
        std::cin.get();
    }
}
