// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "detail.hpp"

namespace Operon::Fmt::Detail {

namespace {

void FormatNode(Tree const& tree, NameView const& names, std::size_t i, std::string& current, ValueSpec spec)
{
    auto const& s = tree[i];

    switch (s.Type) {
    case NodeType::Constant: {
        AppendValue(current, s.Value, spec);
        break;
    }
    case NodeType::Variable: {
        current += "(";
        AppendValue(current, s.Value, spec);
        current += " * ";
        AppendVariableName(current, names, s.HashValue);
        current += ")";
        break;
    }
    case NodeType::Ref: {
        // Ref has no tokens of its own -- replay the referenced subtree's
        // token range (it occupies the contiguous flat range
        // [RefTo-Length, RefTo], same as any other subtree) instead of
        // emitting a bare, non-evaluable "ref" token.
        auto const& target = tree[s.RefTo];
        auto const first = static_cast<std::size_t>(s.RefTo - target.Length);
        auto const last = static_cast<std::size_t>(s.RefTo);
        for (auto j = first;; ++j) {
            auto const parentStart = static_cast<std::size_t>(tree[j].Parent - tree[tree[j].Parent].Length);
            if (j == parentStart) {
                current += "(";
            }
            FormatNode(tree, names, j, current, spec);
            if (!tree[j].IsLeaf()) {
                current += ")";
            }
            if (j == last) { break; }
            current += " ";
        }
        break;
    }
    default: {
        // s.Name() is data, appended directly -- never passed through a
        // runtime format string (a registered function's name may legally
        // contain '{'/'}').
        current += s.Name();
        if (s.Value != Operon::Scalar{1}) {
            // Unlike the leaf/variable case above (a single "(w * name)"
            // token, not real RPN), a function node already has its own
            // argument tokens preceding it in the stream, so its weight is
            // represented as genuine trailing RPN: "... name w *" pushes
            // the weight and multiplies it onto the function's result,
            // exactly mirroring what the interpreter evaluates.
            current += " ";
            AppendValue(current, s.Value, spec);
            current += " *";
        }
    }
    }
}

} // namespace

auto FormatPostfix(Tree const& tree, NameView const& names, ValueSpec spec) -> std::string
{
    std::string result;
    for (auto i = std::size_t{0}; i < tree.Length(); ++i) {
        auto const parentStart = static_cast<std::size_t>(tree[i].Parent - tree[tree[i].Parent].Length);
        if (i == parentStart) {
            result += "(";
        }
        FormatNode(tree, names, i, result, spec);
        if (!tree[i].IsLeaf()) {
            result += ")";
        }
        result += " ";
    }
    return result;
}

} // namespace Operon::Fmt::Detail
