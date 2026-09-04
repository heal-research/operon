// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <fmt/format.h>

#include "operon/core/dataset.hpp"
#include "operon/formatter/formatter.hpp"
#include "detail.hpp"

namespace Operon::Fmt {

auto NameView::Resolve(Operon::Hash hash) const -> std::optional<std::string_view>
{
    switch (kind_) {
    case Kind::None:
        return std::nullopt;
    case Kind::Dataset:
        return static_cast<Operon::Dataset const*>(source_)->FindVariableName(hash);
    case Kind::Map: {
        auto const& map = *static_cast<VariableNameMap const*>(source_);
        auto it = map.find(hash);
        return it == map.end() ? std::nullopt : std::optional<std::string_view>{it->second};
    }
    }
    return std::nullopt;
}

namespace Detail {

void AppendValue(std::string& out, Operon::Scalar value, ValueSpec spec)
{
    if (spec.Fixed) {
        if (value < 0) { fmt::format_to(std::back_inserter(out), "({:.{}f})", value, spec.Precision); }
        else            { fmt::format_to(std::back_inserter(out), "{:.{}f}", value, spec.Precision); }
    } else {
        if (value < 0) { fmt::format_to(std::back_inserter(out), "({:.{}g})", value, spec.Precision); }
        else            { fmt::format_to(std::back_inserter(out), "{:.{}g}", value, spec.Precision); }
    }
}

void AppendVariableName(std::string& out, NameView const& names, Operon::Hash hash)
{
    if (auto name = names.Resolve(hash)) {
        out.append(*name);
        return;
    }
    if (names.HasSource()) {
        throw std::runtime_error(fmt::format("Operon tree formatting: no variable name registered for hash {}", hash));
    }
    fmt::format_to(std::back_inserter(out), "X_{:016x}", hash);
}

auto Render(Tree const& tree, Mode mode, NameView const& names, ValueSpec spec) -> std::string
{
    // Dot always produces valid output (a possibly empty digraph) --
    // FormatDot handles the empty-tree case internally. The other three
    // modes are undefined on an empty tree (no root node to start
    // traversal from); their contract is simply "empty tree -> empty
    // string," matching InfixFormatter's pre-existing, already-correct
    // behavior (extended here to Postfix/Tree, which previously lacked
    // this guard -- TreeFormatter in particular underflowed on
    // tree.Length() - 1 and crashed).
    if (mode == Mode::Dot) { return FormatDot(tree, names, spec); }
    if (tree.Empty()) { return {}; }

    switch (mode) {
    case Mode::Infix:   return FormatInfix(tree, names, spec);
    case Mode::Postfix: return FormatPostfix(tree, names, spec);
    case Mode::Tree:     return FormatTreeDiagram(tree, names, spec);
    case Mode::Dot:      break; // unreachable, handled above
    }
    return {};
}

} // namespace Detail
} // namespace Operon::Fmt
