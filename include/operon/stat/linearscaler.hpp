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
            LinearScalingCalculator() { Reset(); }
            void Reset()
            {
                tCalculator.Reset();
                ovCalculator.Reset();
                otCalculator.Reset();

                alpha = 0;
                beta = 0;
            }
            void Add(operon::scalar_t original, operon::scalar_t target)
            {
                tCalculator.Add(target);
                ovCalculator.Add(original);
                otCalculator.Add(original, target);

                //if (ovCalculator.Variance() < eps<operon::scalar_t>) beta = 1;
                auto variance = ovCalculator.Count() > 1 ? ovCalculator.SampleVariance() : 0;
                beta = variance < std::numeric_limits<operon::scalar_t>::epsilon() ? 1 : (otCalculator.SampleCovariance() / variance);
                //else beta = otCalculator.Covariance() / ovCalculator.Variance();
                alpha = tCalculator.Mean() - beta * ovCalculator.Mean();
            }
            operon::scalar_t Beta() const { return beta; }
            operon::scalar_t Alpha() const { return alpha; }

            template <typename InputIt1, typename InputIt2, typename U = typename InputIt1::value_type>
                static std::pair<operon::scalar_t, operon::scalar_t> Calculate(InputIt1 xBegin, InputIt1 xEnd, InputIt2 yBegin)
                {
                    static_assert(std::is_floating_point_v<U>);
                    LinearScalingCalculator calc;
                    for (; xBegin != xEnd; ++xBegin, ++yBegin) {
                        calc.Add(*xBegin, *yBegin);
                    }
                    return { calc.Alpha(), calc.Beta() };
                }

        private:
            operon::scalar_t alpha; // additive constant
            operon::scalar_t beta; // multiplicative factor

            MeanVarianceCalculator tCalculator; // target values
            MeanVarianceCalculator ovCalculator; // original values
            PearsonsRCalculator otCalculator;
    };


} // namespace

#endif


