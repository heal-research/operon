// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <fmt/format.h>

#include "detail.hpp"

namespace Operon::Fmt::Detail {

namespace {

// Graphviz quoted-string escaping: a variable/registered-function name
// containing a quote, backslash, or line break previously went straight
// into `label="..."` unescaped, producing invalid or silently
// reinterpreted DOT.
void AppendEscapedLabel(std::string& out, std::string_view text)
{
    for (char const c : text) {
        switch (c) {
        case '\\': out += "\\\\"; break;
        case '"':  out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        default:   out += c;
        }
    }
}

auto FormatLabel(Tree const& tree, NameView const& names, std::size_t i, ValueSpec spec) -> std::string
{
    auto const& s = tree[i];
    std::string label;
    if (s.IsConstant()) {
        AppendValue(label, s.Value, spec);
        return label;
    }
    if (s.IsVariable()) {
        label += "(";
        AppendValue(label, s.Value, spec);
        label += " * ";
        AppendVariableName(label, names, s.HashValue);
        label += ")";
        return label;
    }
    // Function node. Weights are real, evaluated state (see
    // Tree::AdjustedLength()/the interpreter's weighted apply) -- omitting
    // them here made the label describe a different model than the one
    // actually evaluated.
    if (s.Value != Operon::Scalar{1}) {
        label += "(";
        AppendValue(label, s.Value, spec);
        label += " * ";
        label += s.Name();
        label += ")";
    } else {
        label = s.Name();
    }
    return label;
}

} // namespace

auto FormatDot(Tree const& tree, NameView const& names, ValueSpec spec) -> std::string
{
    std::string result;
    result += "strict digraph {\n";
    result += "\trankdir=BT\n";

    // A Ref is a pointer, not an independent DAG node -- chase it to the
    // node it actually points at so edges connect to the shared node
    // itself rather than to a dangling/duplicated placeholder.
    auto resolve = [&](std::size_t idx) -> std::size_t {
        while (tree[idx].IsRef()) { idx = tree[idx].RefTo; }
        return idx;
    };

    for (auto i = 0UL; i < tree.Length(); ++i) {
        if (tree[i].IsRef()) { continue; }

        fmt::format_to(std::back_inserter(result), "\t{} [label=\"", i);
        AppendEscapedLabel(result, FormatLabel(tree, names, i, spec));
        result += "\"]\n";

        if (tree[i].IsLeaf()) { continue; }

        for (auto j : tree.Indices(i)) {
            fmt::format_to(std::back_inserter(result), "\t{} -> {}\n", resolve(j), i);
        }
    }

    result += "}\n";
    return result;
}

} // namespace Operon::Fmt::Detail
