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

#include "operators/mutation.hpp"

namespace Operon {
Tree OnePointMutation::operator()(Operon::Random& random, Tree tree) const
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

Tree MultiPointMutation::operator()(Operon::Random& random, Tree tree) const
{
    std::normal_distribution<double> normalReal(0, 1);
    for (auto& node : tree.Nodes()) {
        if (node.IsLeaf()) {
            node.Value += normalReal(random);
        }
    }
    return tree;
}

Tree MultiMutation::operator()(Operon::Random& random, Tree tree) const
{
    auto i = std::discrete_distribution<gsl::index>(probabilities.begin(), probabilities.end())(random);
    auto op = operators[i];
    return op(random, std::move(tree));
}

Tree ChangeVariableMutation::operator()(Operon::Random& random, Tree tree) const
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
