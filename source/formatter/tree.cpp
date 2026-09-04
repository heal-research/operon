// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <fmt/format.h>

#include "detail.hpp"

namespace Operon::Fmt::Detail {

namespace {

void FormatNode(Tree const& tree, NameView const& names, std::size_t i, std::string& current,
                 std::string indent, bool isLast, bool initialMarker, ValueSpec spec)
{
    std::string const last{"└── "};
    std::string const notLast{"├── "};

    current += indent;

    if (initialMarker) {
        current += isLast ? last : notLast;
    }

    auto const& s = tree[i];
    if (s.IsConstant()) {
        AppendValue(current, s.Value, spec);
    } else if (s.IsVariable()) {
        AppendValue(current, s.Value, spec);
        current += " * ";
        AppendVariableName(current, names, s.HashValue);
    } else {
        if (s.Value != Operon::Scalar{1}) {
            current += "(";
            AppendValue(current, s.Value, spec);
            current += " * ";
            current += s.Name();
            current += ")";
        } else {
            current += s.Name();
        }
    }
    fmt::format_to(std::back_inserter(current), " D:{} L:{} N:{}\n", s.Depth, s.Level, s.Length + 1);

    if (s.IsLeaf() && !s.IsRef()) {
        return;
    }

    if (i != tree.Length() - 1) {
        indent += isLast ? "    " : "│   ";
    }

    if (s.IsRef()) {
        // Ref has no children of its own -- nest the shared subtree's
        // diagram directly under it so it's visible rather than eliding it.
        FormatNode(tree, names, s.RefTo, current, indent, /*isLast=*/true, /*initialMarker=*/true, spec);
        return;
    }

    std::size_t count = 0;
    for (auto j : tree.Indices(i)) {
        FormatNode(tree, names, j, current, indent, ++count == s.Arity, /*initialMarker=*/true, spec);
    }
}

} // namespace

auto FormatTreeDiagram(Tree const& tree, NameView const& names, ValueSpec spec) -> std::string
{
    std::string result;
    FormatNode(tree, names, tree.Length() - 1, result, "", /*isLast=*/true, /*initialMarker=*/false, spec);
    return result;
}

} // namespace Operon::Fmt::Detail
