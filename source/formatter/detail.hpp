// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_FORMATTER_DETAIL_HPP
#define OPERON_FORMATTER_DETAIL_HPP

// Private, not-installed header shared by formatter.cpp (the compiled
// fmt::formatter entry point) and the four per-mode backends
// (infix/postfix/tree/dot.cpp). Not part of the public API in
// include/operon/formatter/formatter.hpp -- callers never include this
// directly.

#include <string>

#include "operon/formatter/formatter.hpp"

namespace Operon::Fmt::Detail {

// Compiled per-mode traversal entry points, called only by Render()
// (formatter.cpp) once the caller-facing policy already applies: Render()
// guarantees `tree` is non-empty before calling FormatInfix/FormatPostfix/
// FormatTreeDiagram (their traversal logic assumes at least a root node
// exists); FormatDot handles an empty tree itself (a valid, node-less
// digraph is well-defined, unlike an empty expression/diagram/token
// stream).
auto FormatInfix(Tree const& tree, NameView const& names, ValueSpec spec) -> std::string;
auto FormatPostfix(Tree const& tree, NameView const& names, ValueSpec spec) -> std::string;
auto FormatTreeDiagram(Tree const& tree, NameView const& names, ValueSpec spec) -> std::string;
auto FormatDot(Tree const& tree, NameView const& names, ValueSpec spec) -> std::string;

// Shared leaf-value rendering: a constant, or the numeric weight of a
// weighted variable/function node -- byte-for-byte the same operation in
// every mode (only the surrounding punctuation differs, which stays
// mode-local). Uses fmt's own dynamic-precision replacement field
// ("{:.{}g}"/"{:.{}f}") so the spec is parsed exactly once per call as a
// checked literal, never built as a string and re-parsed via
// fmt::runtime(...) the way the pre-redesign formatters did.
void AppendValue(std::string& out, Operon::Scalar value, ValueSpec spec);

// Resolves a Variable/Ref-target node's display name against `names`,
// appending it to `out`: the resolved name for a Dataset/map source, or a
// deterministic "X_<16 hex digit hash>" placeholder if no source was
// supplied (bare Tree formatting -- a Tree does not itself own variable
// names). Throws std::runtime_error if a source WAS supplied but doesn't
// contain `hash` -- see NameView::HasSource's doc comment in formatter.hpp.
void AppendVariableName(std::string& out, NameView const& names, Operon::Hash hash);

} // namespace Operon::Fmt::Detail

#endif
