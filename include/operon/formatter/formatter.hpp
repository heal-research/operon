// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_FORMAT_HPP
#define OPERON_FORMAT_HPP

#include <algorithm>
#include <concepts>
#include <fmt/base.h>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/operon_export.hpp"

namespace Operon {
class Dataset;
} // namespace Operon

namespace Operon::Fmt {

using VariableNameMap = Operon::Map<Operon::Hash, std::string>;

// Rendering mode selected by the `fmt` format-spec grammar (see
// TreeFormatSpec::parse below): `{}`/`{:infix}`, `{:postfix}`, `{:tree}`,
// `{:dot}`.
enum class Mode : std::uint8_t { Infix, Postfix, Tree, Dot };

// Only Dataset and an explicit Hash->name map are legitimate variable-name
// sources in this codebase; constraining WithNames<Names> to this concept
// turns "wrong type passed" into a clear error at the WithNames call site
// instead of a deep NameView overload-resolution failure.
template <typename T>
concept NameSource = std::same_as<T, Operon::Dataset> || std::same_as<T, VariableNameMap>;

// Type-erased, non-owning view over "resolve a variable Hash to its
// display name," constructed once per fmt::format() call (not once per
// node -- this is what fixes the historical TreeFormatter bug where the
// name map was copied on every recursive call). A default-constructed
// NameView (no Dataset/map supplied) is the bare-Tree fallback: Resolve()
// always returns nullopt, and callers render a deterministic
// X_<hex hash> placeholder instead (see FormatValue's caller in
// source/formatter/detail.hpp) -- diagnostic-only, since a Tree does not
// itself own variable names.
class OPERON_EXPORT NameView {
public:
    NameView() noexcept = default;
    explicit NameView(Operon::Dataset const& dataset) noexcept
        : kind_(Kind::Dataset), source_(&dataset) {}
    explicit NameView(VariableNameMap const& names) noexcept
        : kind_(Kind::Map), source_(&names) {}

    // Resolve() returning nullopt is ambiguous on its own (missing hash vs.
    // no source supplied at all) -- callers needing to throw on a genuinely
    // missing hash from an explicit source, while still falling back to a
    // placeholder when no source was supplied (bare Tree formatting), must
    // check HasSource() first. See FormatVariableName in
    // source/formatter/detail.hpp for the shared implementation of that
    // policy.
    [[nodiscard]] auto HasSource() const noexcept -> bool { return kind_ != Kind::None; }
    [[nodiscard]] auto Resolve(Operon::Hash hash) const -> std::optional<std::string_view>;

private:
    enum class Kind : std::uint8_t { None, Dataset, Map };
    Kind kind_{Kind::None};
    void const* source_{nullptr};
};

// Resolved, per-call (not per-node) value-formatting policy. Infix
// defaults to Fixed=false (significant-digits/general "%g" formatting --
// pass std::numeric_limits<Operon::Scalar>::max_digits10 as Precision, or
// use the ":roundtrip" spec keyword, for output that must round-trip
// exactly back through InfixParser). Tree/Postfix/Dot default to
// Fixed=true (fixed decimal places), matching their long-standing visual
// convention -- see TreeFormatSpec::Resolve.
struct ValueSpec {
    int Precision{2};
    bool Fixed{false};
};

// Wraps a Tree with the variable-name source used to resolve its
// Variable nodes' display names -- what callers actually pass to
// fmt::format/fmt::print: fmt::format("{:infix}", WithNames{tree, dataset}).
// Non-owning; both Subject and Variables must outlive the format call
// (ordinary fmt::format/fmt::print usage, which consumes its arguments
// synchronously, is always safe -- storing a WithNames for later/async
// formatting is not).
// Non-owning formatting arguments: both references are consumed synchronously
// by fmt::format/fmt::print and must therefore remain references.
template <NameSource Names>
struct WithNames {
    Operon::Tree const& Subject; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
    Names const& Variables; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
    int Precision{2};
};

template <NameSource Names>
WithNames(Operon::Tree const&, Names const&) -> WithNames<Names>;
template <NameSource Names>
WithNames(Operon::Tree const&, Names const&, int) -> WithNames<Names>;

// Shared `fmt` format-spec grammar for both fmt::formatter<Operon::Tree>
// and fmt::formatter<Operon::Fmt::WithNames<Names>> (inherited by both, so
// the grammar is parsed in exactly one place):
//
//   tree-spec ::= [mode] [":" precision]
//   mode      ::= "infix" | "postfix" | "tree" | "dot"      (default: infix)
//   precision ::= "roundtrip" | digit+ ["g" | "f"]
//
// Examples: "{}", "{:infix}", "{:tree}", "{:infix:roundtrip}",
// "{:infix:6g}", "{:tree:3f}", "{:dot:6g}". An explicit digit-precision
// override (with or without a trailing presentation letter) always wins
// over WithNames::Precision; "roundtrip" is shorthand for
// max_digits10 significant digits. An unrecognized mode word or trailing
// spec text is a compile error for a literal format string (fmt's
// consteval parse check) or an fmt::format_error at runtime for
// fmt::runtime(...).
struct TreeFormatSpec {
    Mode RenderMode{Mode::Infix};
    std::optional<int> PrecisionOverride;
    std::optional<bool> FixedOverride;

    constexpr auto parse(fmt::format_parse_context& ctx) -> fmt::format_parse_context::iterator; // NOLINT(readability-identifier-naming)

    [[nodiscard]] auto Resolve(int basePrecision) const noexcept -> ValueSpec
    {
        return ValueSpec{
            .Precision = PrecisionOverride.value_or(basePrecision),
            .Fixed = FixedOverride.value_or(RenderMode != Mode::Infix),
        };
    }
};

// Compiled (non-template) rendering entry point. The four traversal
// implementations (infix/postfix/tree-diagram/dot) stay in their own
// source/formatter/{infix,postfix,tree,dot}.cpp files, exactly as
// before -- see source/formatter/detail.hpp for their internal
// declarations. Only this dispatcher needs to be public/header-visible.
namespace Detail {
    OPERON_EXPORT auto Render(Tree const& tree, Mode mode, NameView const& names, ValueSpec spec) -> std::string;
} // namespace Detail

// NOLINTNEXTLINE(readability-function-cognitive-complexity, readability-identifier-naming)
constexpr auto TreeFormatSpec::parse(fmt::format_parse_context& ctx) -> fmt::format_parse_context::iterator
{
    auto it = ctx.begin(); // NOLINT(llvm-qualified-auto, readability-qualified-auto)
    auto const end = ctx.end(); // NOLINT(llvm-qualified-auto, readability-qualified-auto)

    auto consumeWord = [&](std::string_view word) -> bool {
        auto const n = word.size();
        if (std::cmp_less(end - it, n) || !std::equal(word.begin(), word.end(), it)) {
            return false;
        }
        // Reject a mode word that's actually a prefix of some longer,
        // unrecognized token (e.g. "treexyz") rather than silently
        // accepting "tree" out of it.
        if (it + static_cast<std::ptrdiff_t>(n) != end) {
            auto next = *(it + static_cast<std::ptrdiff_t>(n));
            if (next != ':' && next != '}') { return false; }
        }
        it += static_cast<std::ptrdiff_t>(n);
        return true;
    };

    if (consumeWord("infix")) { RenderMode = Mode::Infix; }
    else if (consumeWord("postfix")) { RenderMode = Mode::Postfix; }
    else if (consumeWord("tree")) { RenderMode = Mode::Tree; }
    else if (consumeWord("dot")) { RenderMode = Mode::Dot; }

    if (it != end && *it == ':') {
        ++it;
        if (consumeWord("roundtrip")) {
            PrecisionOverride = std::numeric_limits<Operon::Scalar>::max_digits10;
            FixedOverride = false;
        } else {
            int value = 0;
            bool any = false;
            while (it != end && *it >= '0' && *it <= '9') {
                any = true;
                auto const digit = *it - '0';
                if (value > (std::numeric_limits<int>::max() - digit) / 10) {
                    fmt::report_error("Operon tree format spec: precision too large");
                }
                value = (value * 10) + digit;
                ++it;
            }
            if (!any) {
                fmt::report_error("Operon tree format spec: expected a precision digit after ':'");
            }
            PrecisionOverride = value;
            if (it != end && (*it == 'g' || *it == 'f')) {
                FixedOverride = (*it == 'f');
                ++it;
            }
        }
    }

    if (it != end && *it != '}') {
        fmt::report_error("Operon tree format spec: expected 'infix'/'postfix'/'tree'/'dot', an optional ':' precision, then '}'");
    }

    return it;
}

} // namespace Operon::Fmt

template <>
struct fmt::formatter<Operon::Tree> : Operon::Fmt::TreeFormatSpec {
    template <typename FormatContext>
    auto format(Operon::Tree const& tree, FormatContext& ctx) const -> decltype(ctx.out()) // NOLINT(readability-identifier-naming)
    {
        auto text = Operon::Fmt::Detail::Render(tree, RenderMode, Operon::Fmt::NameView{}, Resolve(2));
        return fmt::format_to(ctx.out(), "{}", text);
    }
};

template <Operon::Fmt::NameSource Names>
struct fmt::formatter<Operon::Fmt::WithNames<Names>> : Operon::Fmt::TreeFormatSpec {
    template <typename FormatContext>
    auto format(Operon::Fmt::WithNames<Names> const& w, FormatContext& ctx) const -> decltype(ctx.out()) // NOLINT(readability-identifier-naming)
    {
        auto text = Operon::Fmt::Detail::Render(w.Subject, RenderMode, Operon::Fmt::NameView{w.Variables}, Resolve(w.Precision));
        return fmt::format_to(ctx.out(), "{}", text);
    }
};

#endif
