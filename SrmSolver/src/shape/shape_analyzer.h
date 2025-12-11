#pragma once

#pragma warning(push, 0)
#include <boost/geometry.hpp>
#pragma warning(pop)

#include "cuda_includes.h"

#include "gpu_grid.h"
#include "shape/shape.h"

namespace kae {

    namespace bg = boost::geometry;

    template <class ElemT>
    class ShapeAnalyzer {
    public:
        using Point2d = bg::model::d2::point_xy<ElemT>;
        using Box2d = bg::model::box<Point2d>;
        using Polygon2d = bg::model::polygon<Point2d, false>;
        using Linestring2d = bg::model::linestring<Point2d>;

        ShapeAnalyzer(Linestring2d initialSrmShape, Linestring2d propellantShape) : m_initialSrmShape{ std::move(initialSrmShape) }, m_propellantShape{ std::move(propellantShape) }
        {
            bg::correct(m_initialSrmShape);
            bg::correct(m_propellantShape);
        }

        ElemT getInitialSBurn() const
        {
            ElemT sBurn{};

            Polygon2d polygon;
            std::copy(std::begin(m_propellantShape), std::end(m_propellantShape), std::back_inserter(polygon.outer()));
            bg::correct(polygon);

            bg::for_each_segment(m_initialSrmShape, [&](auto segment) {
                const auto p0 = segment.first;
                const auto p1 = segment.second;
                if (bg::covered_by(p0, polygon) && bg::covered_by(p1, polygon))
                {
                    const auto distance = bg::distance(p1, p0);
                    sBurn += static_cast<ElemT>(M_PI) * (bg::get<1>(p1) + bg::get<1>(p0)) * distance;
                }
                });

            return sBurn;
        }

        Shape<ElemT> buildShape(const GpuGridT<ElemT>& grid) const
        {
            auto initialOffsettedSrmShape = getOffsetShape(m_initialSrmShape, grid.hx);
            auto propellantOffsettedShape = getOffsetShape(m_propellantShape, grid.hx);

            auto initialBbox = bg::return_envelope<Box2d>(initialOffsettedSrmShape);
            auto propellantBbox = bg::return_envelope<Box2d>(propellantOffsettedShape);
            const auto criticalPoint = findCriticalPoint();
            const auto offset = getOffset(grid.hx);

            const auto yBottom = bg::get<0, 1>(initialBbox);
            const auto xRight = bg::get<1, 0>(initialBbox);
            const auto rkr = bg::get<1>(criticalPoint);
            const auto xkr = bg::get<0>(criticalPoint) + bg::get<0>(offset);
            const auto propellantMin = CudaFloat2T<ElemT>{ bg::get<0, 0>(propellantBbox), bg::get<0, 1>(propellantBbox) };
            const auto propellantMax = CudaFloat2T<ElemT>{ bg::get<1, 0>(propellantBbox), bg::get<1, 1>(propellantBbox) };

            CudaFloat2T<ElemT>* deviceValues = nullptr;
            cudaMalloc((void**)&deviceValues, sizeof(CudaFloat2T<ElemT>) * propellantOffsettedShape.size());
            cudaMemcpy(deviceValues, propellantOffsettedShape.data(), sizeof(CudaFloat2T<ElemT>) * propellantOffsettedShape.size(), cudaMemcpyHostToDevice);

            return Shape<ElemT>(yBottom, xRight, rkr, xkr, propellantMin, propellantMax, thrust::device_ptr<CudaFloat2T<ElemT>>(deviceValues), static_cast<unsigned>(propellantOffsettedShape.size()));
        }

        Point2d findCriticalPoint() const
        {
            Point2d p;
            for (std::size_t i{ 3ULL }; i + 1 < m_initialSrmShape.size(); ++i)
            {
                const auto prevPoint = m_initialSrmShape.at(i - 1ULL);
                const auto currPoint = m_initialSrmShape.at(i);
                const auto nextPoint = m_initialSrmShape.at(i + 1ULL);
                if (bg::get<1>(currPoint) < bg::get<1>(prevPoint) && bg::get<1>(currPoint) <= bg::get<1>(nextPoint))
                {
                    p = currPoint;
                    break;
                }
            }

            return p;
        }

        thrust::host_vector<ElemT> getSignedDistances(const GpuGridT<ElemT>& grid) const {

            auto initialSrmShape = getOffsetShape(m_initialSrmShape, grid.hx);

            Polygon2d polygon;
            std::copy(std::begin(initialSrmShape), std::end(initialSrmShape), std::back_inserter(polygon.outer()));
            bg::correct(polygon);

            thrust::host_vector<ElemT> distances(grid.nx * grid.ny);
            for (unsigned i = 0U; i < grid.nx; ++i)
            {
                const auto x = i * grid.hx;
                for (unsigned j = 0U; j < grid.ny; ++j)
                {
                    const auto y = j * grid.hy;
                    const Point2d point{ x, y };
                    const auto distance = static_cast<ElemT>(bg::distance(point, initialSrmShape));
                    const auto isInside = bg::covered_by(point, polygon);

                    const auto index = j * grid.nx + i;
                    distances[index] = isInside ? -std::fabs(distance) : std::fabs(distance);
                }
            }

            return distances;
        }

        GpuGridT<ElemT> calculateGrid(ElemT h) const
        {
            const auto bbox = getGeometryBoundingBox(h);

            const auto lx = bg::get<0U>(bbox.max_corner()) - bg::get<0U>(bbox.min_corner());
            const auto ly = bg::get<1U>(bbox.max_corner()) - bg::get<1U>(bbox.min_corner());

            const unsigned nx = static_cast<unsigned>(std::ceil(lx / h));
            const unsigned ny = static_cast<unsigned>(std::ceil(ly / h));

            return GpuGridT<ElemT>(nx + 1, ny + 1, nx * h, ny * h, 3U);
        }

        Box2d getGeometryBoundingBox(ElemT h) const {
            auto initialBbox = bg::return_envelope<Box2d>(m_initialSrmShape);
            bg::expand(initialBbox, bg::return_envelope<Box2d>(m_propellantShape));

            auto startPoint = getOffset(h);
            bg::multiply_value(startPoint, static_cast<ElemT>(2.0));
            bg::add_point(initialBbox.max_corner(), startPoint);

            return initialBbox;
        }

        static Linestring2d getOffsetShape(Linestring2d shape, ElemT h)
        {
            const auto offsetPoint = getOffset(h);
            std::for_each(std::begin(shape), std::end(shape), [=](auto& point)
                {
                    bg::add_point(point, offsetPoint);
                });

            return shape;
        }

        static Point2d getOffset(ElemT h)
        {
            return{ offsetPointsX * h, offsetPointsY * h };
        }

    private:
        static constexpr auto offsetPointsX = static_cast<ElemT>(16.5);
        static constexpr auto offsetPointsY = static_cast<ElemT>(16.5);

    private:
        Linestring2d m_initialSrmShape;
        Linestring2d m_propellantShape;
    };

} // namespace kae
