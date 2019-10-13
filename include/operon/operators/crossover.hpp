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

#ifndef CROSSOVER_HPP
#define CROSSOVER_HPP

#include <vector>

#include "core/operator.hpp"

namespace Operon {
class SubtreeCrossover : public CrossoverBase {
public:
    SubtreeCrossover(double p, size_t d, size_t l)
        : internalProbability(p)
        , maxDepth(d)
        , maxLength(l)
    {
    }
    auto operator()(operon::rand_t& random, const Tree& lhs, const Tree& rhs) const -> Tree override;

private:
    double internalProbability;
    size_t maxDepth;
    size_t maxLength;
};
}
#endif
