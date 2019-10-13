/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Copyright (C) 2019 Bogdan Burlacu 
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * SOFTWARE.
 */

#ifndef LINEARSCALER_HPP
#define LINEARSCALER_HPP

#include "core/common.hpp"
#include "core/stat/meanvariance.hpp"
#include "core/stat/pearson.hpp"

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


