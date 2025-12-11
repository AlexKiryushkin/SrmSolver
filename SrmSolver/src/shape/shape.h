#pragma once

#include "boundary_condition.h"
#include "cuda_float_types.h"
#include "cuda_includes.h"

namespace kae {

    template <class ElemT>
    class Shape
    {
    public:
        Shape(ElemT yBottom, ElemT xRight, ElemT rkr, ElemT xkr,
            CudaFloat2T<ElemT> propellantMin,
            CudaFloat2T<ElemT> propellantMax, thrust::device_ptr<CudaFloat2T<ElemT>> pPropellantPoints,
            unsigned propellantPointsCount)
            : m_yBottom{ yBottom }, m_xRight{ xRight }, m_rkr{ rkr }, m_xkr{ xkr },
            m_propellantMin{ propellantMin }, m_propellantMax{ propellantMax }, m_pPropellantPoints{ pPropellantPoints }, m_propellantPointsCount{ propellantPointsCount }
        {
        }

        HOST_DEVICE EBoundaryCondition getBoundaryCondition(ElemT x, ElemT y, ElemT h) const
        {
            if (isPointOnGrain(x, y, h))
            {
                return EBoundaryCondition::eMassFlowInlet;
            }

            if (std::fabs(x - m_xRight) < static_cast<ElemT>(0.1) * h)
            {
                return EBoundaryCondition::ePressureOutlet;
            }

            return EBoundaryCondition::eWall;
        }

        HOST_DEVICE bool isChamber(ElemT x, ElemT /*y*/, ElemT /*h*/) const
        {
            if (m_propellantMax.x > m_xkr)
            {
                return true;
            }

            const auto lastXChamber = static_cast<ElemT>(0.5) * (m_propellantMax.x + m_xkr);
            return x < lastXChamber;
        }

        DEVICE bool isPointOnGrain(ElemT x, ElemT y, ElemT h) const
        {
            const auto threshold = static_cast<ElemT>(0.1) * h;

            bool inside = false;
            for (int i = 0; i + 1 < m_propellantPointsCount; ++i)
            {
                unsigned j = i + 1;

                CudaFloat2T<ElemT> pi = m_pPropellantPoints[i];
                CudaFloat2T<ElemT> pj = m_pPropellantPoints[j];
                if (((pi.y > y) != (pj.y > y)) &&
                    (x < (pj.x - pi.x) * (y - pi.y) / ((pj.y - pi.y) == 0 ? 1e-7f : (pj.y - pi.y)) + pi.x))
                {
                    inside = !inside;
                }
            }
            if (inside)
                return true;

            // If not inside, check if within threshold distance to any edge
            float thr2 = threshold * threshold;
            for (int i = 0; i + 1 < m_propellantPointsCount; ++i)
            {
                unsigned j = i + 1;

                CudaFloat2T<ElemT> pi = m_pPropellantPoints[i];
                CudaFloat2T<ElemT> pj = m_pPropellantPoints[j];

                ElemT dx = pj.x - pi.x;
                ElemT dy = pj.y - pi.y;
                ElemT len2 = std::max({ dx * dx + dy * dy, static_cast<ElemT>(1e-7) });
                ElemT t = std::clamp(((x - pi.x) * dx + (y - pi.y) * dy) / len2, static_cast<ElemT>(0), static_cast<ElemT>(1));

                ElemT closestX = pi.x + t * dx;
                ElemT closestY = pi.y + t * dy;
                ElemT dist2 = (x - closestX) * (x - closestX) + (y - closestY) * (y - closestY);
                if (dist2 <= thr2)
                    return true;
            }
            return false;
        }

        HOST_DEVICE bool shouldApplyScheme(unsigned i, unsigned j, ElemT h) const {
            const auto isFarFromPropellant = ((i * h + 15 * h <= m_propellantMin.x) || (i * h - 15 * h >= m_propellantMax.x)) &&
                ((j * h + 15 * h <= m_propellantMin.y) || (j * h - 15 * h >= m_propellantMax.y));
            return !isFarFromPropellant;
        }

        HOST_DEVICE ElemT getRadius(ElemT x, ElemT y) const
        {
            return y - m_yBottom;
        }

        HOST_DEVICE ElemT getFCritical() const
        {
            return static_cast<ElemT>(M_PI) * m_rkr * m_rkr;
        }

        HOST_DEVICE ElemT getOutletCoordinate() const { return m_xRight; }

    private:
        ElemT m_yBottom;
        ElemT m_xRight;
        ElemT m_rkr;
        ElemT m_xkr;
        CudaFloat2T<ElemT> m_propellantMin;
        CudaFloat2T<ElemT> m_propellantMax;
        thrust::device_ptr<CudaFloat2T<ElemT>> m_pPropellantPoints;
        unsigned m_propellantPointsCount;
    };
} // namespace kae
