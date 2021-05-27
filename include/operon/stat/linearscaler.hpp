// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef LINEARSCALER_HPP
#define LINEARSCALER_HPP

#include "core/common.hpp"
#include "stat/meanvariance.hpp"
#include "stat/pearson.hpp"

namespace Operon {
    class LinearScalingCalculator {
        public:
            LinearScalingCalculator() : scale(0), offset(0)
            {
                Reset();
            }

            void Reset()
            {
                calc.Reset();
            }

            template<typename T>
            void Add(T original, T target)
            {
                calc.Add(original, target);

                auto variance = calc.Count() > 1 ? calc.SampleVarianceX() : 0;
                scale = variance < std::numeric_limits<double>::epsilon() ? 1 : (calc.SampleCovariance() / variance);
                offset = calc.MeanY() - scale * calc.MeanX();
            }

            double Scale() const { return scale; }
            double Offset() const { return offset; }

            template <typename InputIt1, typename InputIt2, typename U = typename std::iterator_traits<InputIt1>::value_type>
            static std::pair<double, double> Calculate(InputIt1 xBegin, InputIt1 xEnd, InputIt2 yBegin)
            {
                static_assert(std::is_floating_point_v<U>);
                LinearScalingCalculator calc;
                for (; xBegin != xEnd; ++xBegin, ++yBegin) {
                    calc.Add(*xBegin, *yBegin);
                }
                return { calc.Scale(), calc.Offset() };
            }

            template<typename T>
            static inline std::pair<double, double> Calculate(Operon::Span<T const> lhs, Operon::Span<T const> rhs)
            {
                EXPECT(lhs.size() == rhs.size());
                EXPECT(lhs.size() > 0);
                double s, o;
                Operon::PearsonsRCalculator calc;
                calc.Add(lhs, rhs);
                s = calc.SampleCovariance() / calc.SampleVarianceX();
                o = calc.MeanY() - s * calc.MeanX();
                return { s, o };
            }

        private:
            double scale;
            double offset;

            PearsonsRCalculator calc;
    };


} // namespace

#endif


