// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <fmt/format.h>

#include "operon/core/standard_library.hpp"
#include "detail.hpp"

namespace Operon::Fmt::Detail {

namespace {

void FormatNode(Tree const& tree, NameView const& names, std::size_t i, std::string& current, ValueSpec spec) // NOLINT(readability-function-cognitive-complexity)
{
    auto const& s = tree[i];
    if (s.IsConstant()) {
        AppendValue(current, s.Value, spec);
        return;
    }
    if (s.IsVariable()) {
        current += "(";
        AppendValue(current, s.Value, spec);
        current += " * ";
        AppendVariableName(current, names, s.HashValue);
        current += ")";
        return;
    }
    if (s.IsRef()) {
        // Ref is a leaf (Arity == 0) that points backward to a shared
        // subtree via RefTo, not the physically preceding node -- without
        // this case the generic operator fallback below would recurse into
        // `i-1`, silently formatting the wrong subtree for any tree with
        // structural sharing (e.g. a symbolic-derivative DAG).
        FormatNode(tree, names, s.RefTo, current, spec);
        return;
    }

    // s is a Function node: a built-in math op or a registered
    // user-defined function, distinguished only by HashValue.
    // StandardLibrary::FormattingRule(HashValue) is well-defined for any
    // Hash, built-in or not -- a hash outside the built-in table simply
    // falls through to its GenericCall default, which is exactly the
    // right rendering for a registered function.
    if (s.Value != Operon::Scalar{1}) {
        current += "(";
        AppendValue(current, s.Value, spec);
        current += " * ";
    }

    switch (Operon::StandardLibrary::FormattingRule(static_cast<Operon::BuiltinOp>(s.HashValue))) {
    case Operon::FormatRule::Infix: {
        current += "(";
        if (s.Arity == 1) {
            // Sub/Div's fixed-arity registration (see FormatRule::Infix's
            // declaration) never produces an arity-1 node via ordinary GP
            // construction, but a manually-built or post-simplification
            // tree can: unary Sub is negation (-a), unary Div is
            // inversion (1 / a).
            if (s.IsOp<Operon::BuiltinOp::Sub>()) { current += "-"; }
            else if (s.IsOp<Operon::BuiltinOp::Div>()) { current += "1 / "; }
            FormatNode(tree, names, i - 1, current, spec);
        } else {
            std::size_t count = 0;
            for (auto j : tree.Indices(i)) {
                FormatNode(tree, names, j, current, spec);
                if (++count < s.Arity) {
                    fmt::format_to(std::back_inserter(current), " {} ", s.Name());
                }
            }
        }
        current += ")";
        break;
    }
    case Operon::FormatRule::PowerNotation: {
        current += "(";
        if (s.Arity == 1) { // Square: a ^ 2
            FormatNode(tree, names, i - 1, current, spec);
            current += " ^ 2";
        } else { // Pow: a ^ b
            auto j = i - 1;
            auto k = j - tree[j].Length - 1;
            FormatNode(tree, names, j, current, spec);
            current += " ^ ";
            FormatNode(tree, names, k, current, spec);
        }
        current += ")";
        break;
    }
    case Operon::FormatRule::MinMaxCall: {
        auto j = i - 1;
        auto k = j - tree[j].Length - 1;
        current += s.IsOp<Operon::BuiltinOp::Fmin>() ? "(min(" : "(max(";
        FormatNode(tree, names, j, current, spec);
        current += ", ";
        FormatNode(tree, names, k, current, spec);
        current += "))";
        break;
    }
    case Operon::FormatRule::Composite: {
        if (s.IsOp<Operon::BuiltinOp::Aq>()) {
            // a / (sqrt(1 + b ^ 2))
            auto j = i - 1;
            auto k = j - tree[j].Length - 1;
            current += "(";
            FormatNode(tree, names, j, current, spec);
            current += " / (sqrt(1 + ";
            FormatNode(tree, names, k, current, spec);
            current += " ^ 2)))";
        } else if (s.IsOp<Operon::BuiltinOp::Powabs>()) {
            // abs(a) ^ b
            auto j = i - 1;
            auto k = j - tree[j].Length - 1;
            current += "(abs(";
            FormatNode(tree, names, j, current, spec);
            current += ") ^ ";
            FormatNode(tree, names, k, current, spec);
            current += ")";
        } else if (s.IsOp<Operon::BuiltinOp::Logabs>()) {
            current += "log(abs(";
            FormatNode(tree, names, i - 1, current, spec);
            current += "))";
        } else if (s.IsOp<Operon::BuiltinOp::Log1p>()) {
            current += "log(";
            FormatNode(tree, names, i - 1, current, spec);
            current += "+1)";
        } else { // Sqrtabs
            current += "sqrt(abs(";
            FormatNode(tree, names, i - 1, current, spec);
            current += "))";
        }
        break;
    }
    case Operon::FormatRule::PrefixNegation:
    case Operon::FormatRule::Inversion:
        // Never returned by FormattingRule -- Sub/Div always report
        // FormatRule::Infix (see the enum's own doc comment); the
        // arity-1 special case is handled inline above. Kept as
        // exhaustive switch labels rather than a `default:` so a future
        // FormatRule addition fails to compile here instead of silently
        // falling through to GenericCall.
    case Operon::FormatRule::GenericCall: {
        current += s.Name();
        current += "(";
        if (s.Arity == 1) {
            FormatNode(tree, names, i - 1, current, spec);
        } else {
            // A registered binary/n-ary function (RegisterBinaryFunction/
            // RegisterNaryFunction) reports GenericCall here (its hash is
            // outside the built-in table) -- render every actual child,
            // not just the last one.
            std::size_t count = 0;
            for (auto j : tree.Indices(i)) {
                FormatNode(tree, names, j, current, spec);
                if (++count < s.Arity) { current += ", "; }
            }
        }
        current += ")";
        break;
    }
    }

    if (s.Value != Operon::Scalar{1}) {
        current += ")";
    }
}

} // namespace

auto FormatInfix(Tree const& tree, NameView const& names, ValueSpec spec) -> std::string
{
    std::string result;
    FormatNode(tree, names, tree.Length() - 1, result, spec);
    return result;
}

} // namespace Operon::Fmt::Detail
