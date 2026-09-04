// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <fmt/format.h>

#include "operon/core/dataset.hpp"
#include "operon/formatter/formatter.hpp"

namespace Operon {

auto DotFormatter::Format(Tree const& tree, Operon::Map<Operon::Hash, std::string> const& variableNames, int decimalPrecision) -> std::string
{
    std::string result;
    result += "strict digraph {\n";
    result += "\trankdir=BT\n";

    auto formatLeaf = [&](auto const& s) -> auto {
        if (s.IsConstant()) {
            auto formatString = fmt::format(fmt::runtime("{{:.{}f}}"), decimalPrecision);
            return fmt::format(fmt::runtime(formatString), s.Value);
        }
        if (s.IsVariable()) {
            auto formatString = fmt::format(fmt::runtime("({{:.{}f}} * {{}})"), decimalPrecision);
            if (auto it = variableNames.find(s.HashValue); it != variableNames.end()) {
                return fmt::format(fmt::runtime(formatString), s.Value, it->second);
            }
            throw std::runtime_error(fmt::format("A key with hash value {} could not be found in the variable map.\n", s.HashValue));
        }
        throw std::runtime_error("node is not a leaf (constant or variable)");
    };

    auto format = [&](auto const& s) -> auto {
        if (s.IsLeaf()) { return formatLeaf(s); }
        if (s.Value != Operon::Scalar{1}) {
            // Function-node weights are real, evaluated state (see
            // Tree::AdjustedLength()/the interpreter's weighted apply) --
            // omitting them here made the label describe a different
            // model than the one actually evaluated.
            auto formatString = fmt::format(fmt::runtime(s.Value < 0 ? "(({{:.{}f}}) * {{}})" : "({{:.{}f}} * {{}})"), decimalPrecision);
            return fmt::format(fmt::runtime(formatString), s.Value, s.Name());
        }
        return s.Name();
    };

    // A Ref is a pointer, not an independent DAG node -- chase it to the
    // node it actually points at so edges connect to the shared node
    // itself rather than to a dangling/duplicated placeholder.
    auto resolve = [&](auto idx) {
        while (tree[idx].IsRef()) { idx = tree[idx].RefTo; }
        return idx;
    };

    for (auto i = 0UL; i < tree.Length(); ++i) {
        if (tree[i].IsRef()) { continue; }

        auto label = format(tree[i]);
        fmt::format_to(std::back_inserter(result), "\t{} [label=\"{}\"]\n", i, label);

        if (tree[i].IsLeaf()) { continue; }

        for (auto j : tree.Indices(i)) {
            fmt::format_to(std::back_inserter(result), "\t{} -> {}\n", resolve(j), i);
        }
    }

    result += "}\n";
    return result;
}

auto DotFormatter::Format(Tree const& tree, Dataset const& dataset, int decimalPrecision) -> std::string
{
    Operon::Map<Operon::Hash, std::string> variableNames;
    for (auto const& var : dataset.GetVariables()) {
        variableNames.insert({ var.Hash, var.Name });
    }
    return DotFormatter::Format(tree, variableNames, decimalPrecision);
}

} // namespace Operon
