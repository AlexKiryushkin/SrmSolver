#pragma once

#include "cuda_float_types.h"
#include "math_utilities.h"

namespace kae {

    namespace detail {

        constexpr unsigned derivativesCount = 5U;
        constexpr unsigned fluxesCount = 3U;

        template <class ElemT>
        constexpr ElemT a[fluxesCount] = { 0.1, 0.6, 0.3 };

        template <class ElemT>
        constexpr ElemT LC[fluxesCount][fluxesCount] =
        { 1.0 / 3.0, -7.0 / 6.0, 11.0 / 6.0,
          -1.0 / 6.0,  5.0 / 6.0,  1.0 / 3.0,
           1.0 / 3.0,  5.0 / 6.0, -1.0 / 6.0 };

        template <class ElemT>
        constexpr ElemT WC[2U] = { 13.0 / 12.0, 0.25 };

        template <class ElemT, bool IsPlus>
        struct Derivative
        {
            HOST_DEVICE Derivative(ElemT hxReciprocal, unsigned step) :m_hxReciprocal{ hxReciprocal }, m_step{ step } {}

            HOST_DEVICE ElemT operator()(const ElemT* arr, const unsigned i, const int offset) const
            {
                return (arr[i + (offset + 1) * m_step] - arr[i + offset * m_step]) * m_hxReciprocal;
            }

        private:
            ElemT m_hxReciprocal;
            unsigned m_step;
        };

        template <class ElemT>
        struct Derivative<ElemT, false>
        {
            HOST_DEVICE Derivative(ElemT hxReciprocal, unsigned step) :m_hxReciprocal{ hxReciprocal }, m_step{ step } {}

            HOST_DEVICE ElemT operator()(const ElemT* arr, const unsigned i, const int offset) const
            {
                return (arr[i - offset * m_step] - arr[i - (offset + 1) * m_step]) * m_hxReciprocal;
            }

        private:
            ElemT m_hxReciprocal;
            unsigned m_step;
        };

        template <bool IsPlus, class ElemT>
        HOST_DEVICE ElemT getLevelSetDerivative(const ElemT* arr, const unsigned i, unsigned step, ElemT hx, ElemT hxReciprocal)
        {
            const auto derivativeFunc = Derivative<ElemT, IsPlus>{ hxReciprocal, step };

            const ElemT v[derivativesCount] = { derivativeFunc(arr, i, 2),
                                                derivativeFunc(arr, i, 1),
                                                derivativeFunc(arr, i, 0),
                                                derivativeFunc(arr, i, -1),
                                                derivativeFunc(arr, i, -2) };

            const ElemT flux[fluxesCount] =
            { LC<ElemT>[0][0] * v[0] + LC<ElemT>[0][1] * v[1] + LC<ElemT>[0][2] * v[2],
              LC<ElemT>[1][0] * v[1] + LC<ElemT>[1][1] * v[2] + LC<ElemT>[1][2] * v[3],
              LC<ElemT>[2][0] * v[2] + LC<ElemT>[2][1] * v[3] + LC<ElemT>[2][2] * v[4] };

            const ElemT s[fluxesCount] =
            { WC<ElemT>[0] * sqr(v[0] - 2 * v[1] + v[2]) + WC<ElemT>[1] * sqr(v[0] - 4 * v[1] + 3 * v[2]),
              WC<ElemT>[0] * sqr(v[1] - 2 * v[2] + v[3]) + WC<ElemT>[1] * sqr(v[1] - v[3]),
              WC<ElemT>[0] * sqr(v[2] - 2 * v[3] + v[4]) + WC<ElemT>[1] * sqr(3 * v[2] - 4 * v[3] + v[4]) };

            const ElemT epsilon = sqr(hx);
            const ElemT alpha[fluxesCount] = {
              a<ElemT>[0] / sqr(s[0] + epsilon),
              a<ElemT>[1] / sqr(s[1] + epsilon),
              a<ElemT>[2] / sqr(s[2] + epsilon) };

            return (alpha[0] * flux[0] + alpha[1] * flux[1] + alpha[2] * flux[2]) / (alpha[0] + alpha[1] + alpha[2]);
        }

        template <class ElemT>
        HOST_DEVICE ElemT getLevelSetDerivative(const ElemT* arr, unsigned i, unsigned nx, ElemT hx, ElemT hxReciprocal, bool isPositiveVelocity)
        {
            if (isPositiveVelocity)
            {
                ElemT val1 = thrust::max(getLevelSetDerivative<false>(arr, i, nx, hx, hxReciprocal), static_cast<ElemT>(0.0));
                ElemT val2 = thrust::min(getLevelSetDerivative<true>(arr, i, nx, hx, hxReciprocal), static_cast<ElemT>(0.0));

                return kae::absmax(val1, val2);
            }

            ElemT val1 = thrust::min(getLevelSetDerivative<false>(arr, i, nx, hx, hxReciprocal), static_cast<ElemT>(0.0));
            ElemT val2 = thrust::max(getLevelSetDerivative<true>(arr, i, nx, hx, hxReciprocal), static_cast<ElemT>(0.0));

            return kae::absmax(val1, val2);
        }

        template <class ElemT>
        HOST_DEVICE CudaFloat2T<ElemT> getLevelSetGradient(const ElemT* arr, unsigned i, unsigned nx, ElemT hx, ElemT hxReciprocal, bool isPositiveVelocity)
        {
            ElemT derivativeX = getLevelSetDerivative(arr, i, 1, hx, hxReciprocal, isPositiveVelocity);
            ElemT derivativeY = getLevelSetDerivative(arr, i, nx, hx, hxReciprocal, isPositiveVelocity);

            return { derivativeX, derivativeY };
        }

        template <class ElemT>
        HOST_DEVICE ElemT getLevelSetAbsGradient(const ElemT* arr, unsigned i, unsigned nx, ElemT hx, ElemT hxReciprocal, bool isPositiveVelocity)
        {
            ElemT derivativeX = getLevelSetDerivative(arr, i, 1, hx, hxReciprocal, isPositiveVelocity);
            ElemT derivativeY = getLevelSetDerivative(arr, i, nx, hx, hxReciprocal, isPositiveVelocity);

            return std::hypot(derivativeX, derivativeY);
        }

    } // namespace detail

} // namespace kae
