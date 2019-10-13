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

#include "operators/mutation.hpp"

namespace Operon {
Tree OnePointMutation::operator()(operon::rand_t& random, Tree tree) const
{
    auto& nodes = tree.Nodes();

    auto leafCount = std::count_if(nodes.begin(), nodes.end(), [](const Node& node) { return node.IsLeaf(); });
    std::uniform_int_distribution<gsl::index> uniformInt(1, leafCount);
    auto index = uniformInt(random);

    size_t i = 0;
    for (; i < nodes.size(); ++i) {
        if (nodes[i].IsLeaf() && --index == 0)
            break;
    }

    std::normal_distribution<double> normalReal(0, 1);
    tree[i].Value += normalReal(random);

    return tree;
}

Tree MultiPointMutation::operator()(operon::rand_t& random, Tree tree) const
{
    std::normal_distribution<double> normalReal(0, 1);
    for (auto& node : tree.Nodes()) {
        if (node.IsLeaf()) {
            node.Value += normalReal(random);
        }
    }
    return tree;
}

Tree MultiMutation::operator()(operon::rand_t& random, Tree tree) const
{
    std::uniform_real_distribution<double> uniformReal(0, partials.back() - eps);
    auto r = uniformReal(random);
    auto it = std::find_if(partials.begin(), partials.end(), [=](double p) { return p > r; });
    auto i = std::distance(partials.begin(), it);
    auto op = operators[i];
    return op(random, std::move(tree));
}

Tree ChangeVariableMutation::operator()(operon::rand_t& random, Tree tree) const
{
    auto& nodes = tree.Nodes();

    auto leafCount = std::count_if(nodes.begin(), nodes.end(), [](const Node& node) { return node.IsLeaf(); });
    std::uniform_int_distribution<gsl::index> uniformInt(1, leafCount);
    auto index = uniformInt(random);

    size_t i = 0;
    for (; i < nodes.size(); ++i) {
        if (nodes[i].IsLeaf() && --index == 0)
            break;
    }

    std::uniform_int_distribution<gsl::index> normalInt(0, variables.size() - 1);
    tree[i].HashValue = tree[i].CalculatedHashValue = variables[normalInt(random)].Hash;

    return tree;
}
} // namespace Operon
