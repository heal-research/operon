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

#ifndef MUTATION_HPP
#define MUTATION_HPP

#include "core/operator.hpp"

namespace Operon {
struct OnePointMutation : public MutatorBase {
    Tree operator()(operon::rand_t& random, Tree tree) const override;
};

struct MultiPointMutation : public MutatorBase {
    Tree operator()(operon::rand_t& random, Tree tree) const override;
};

struct MultiMutation : public MutatorBase {
    Tree operator()(operon::rand_t& random, Tree tree) const override;

    void Add(const MutatorBase& op, double prob)
    {
        operators.push_back(std::ref(op));
        partials.push_back(partials.empty() ? prob : prob + partials.back());
    }

private:
    static constexpr double eps = std::numeric_limits<double>::epsilon();
    std::vector<std::reference_wrapper<const MutatorBase>> operators;
    std::vector<double> partials;
};

struct ChangeVariableMutation : public MutatorBase {
    ChangeVariableMutation(const gsl::span<const Variable> vars)
        : variables(vars)
    {
    }

    Tree operator()(operon::rand_t& random, Tree tree) const override;

private:
    const gsl::span<const Variable> variables;
};
}

#endif
