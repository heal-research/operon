/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2019 Bogdan Burlacu 
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE. 
 */

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

            template <typename InputIt1, typename InputIt2, typename U = typename InputIt1::value_type>
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
            static inline std::pair<double, double> Calculate(gsl::span<T const> lhs, gsl::span<T const> rhs)
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


