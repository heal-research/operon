// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_STANDARD_LIBRARY_HPP
#define OPERON_STANDARD_LIBRARY_HPP

#include <array>
#include <cstdint>
#include <string_view>
#include <utility>

#include "dispatch.hpp"
#include "node.hpp"

// Registers the built-in math ops into a DispatchTable via the same
// RegisterFunction<T> API used for user-defined Dynamic functions (see
// symbol_library.hpp). Each entry's Callable/CallableDiff comes from
// Dispatch::MakeFunctionCall/MakeDiffCall<Type,T,S> - the same
// compile-time-specialized kernels DispatchTable's constructor uses
// directly, so the compiled code for a given (Type,T,S) is unaffected by
// which path registered it.

namespace Operon {

// How a node's infix-notation rendering diverges from plain call syntax
// `name(child, ...)` (GenericCall, the default). Consumed by the formatter;
// stored here so a node's rendering rule travels with its other metadata
// instead of living as a separate `if (s.Type == NodeType::X)` chain.
enum class FormatRule : uint8_t {
    GenericCall,    // name(a) / name(a, b, ...)
    Infix,          // a <op> b <op> c ...
    PrefixNegation, // unary form renders as -a (Sub with arity 1; not
                    // reachable via the registry below since Sub's
                    // MinArity/MaxArity is fixed at 2 there)
    Inversion,      // unary form renders as 1 / a (Div with arity 1; same
                    // caveat as PrefixNegation)
    PowerNotation,  // a ^ b
    MinMaxCall,     // min(a, b) / max(a, b)
    Composite,      // bespoke multi-token expansion, e.g. log(abs(a))
};

struct StandardLibrary {
    static void RegisterNames()
    {
        static auto const registered = [] {
            for (auto const& entry : BuiltinEntries) {
                Node::RegisterName(static_cast<Operon::Hash>(entry.Op), std::string(entry.Name), std::string(entry.Desc));
            }
            for (auto const& entry : TerminalEntries) {
                Node::RegisterName(Node::DefaultHash(entry.Type), std::string(entry.Name), std::string(entry.Desc));
            }
            return true;
        }();
        static_cast<void>(registered);
    }

    // Register all built-in math ops into `dt`, for every scalar type
    // DTable supports. Constant/Variable/Ref/(the generic "dyn" Function
    // fallback) have no numeric Callable/CallableDiff to register - they're
    // terminals, not evaluated via the dispatch table.
    template<typename... Ts>
    static void Register(DispatchTable<Ts...>& dt)
    {
        RegisterNames();
        RegisterAll(dt, std::make_index_sequence<BuiltinEntries.size()>{});
    }

    // Min/max arity for a built-in op, sourced from the registry instead of
    // position-based inference (removed along with NodeType's old
    // math-op enumerators; this walks the same data the registry already
    // carries explicitly, which is what PrimitiveSet's preset configs
    // consume).
    [[nodiscard]] static constexpr auto ArityLimits(BuiltinOp op) -> std::pair<uint16_t, uint16_t>
    {
        for (auto const& entry : BuiltinEntries) {
            if (entry.Op == op) { return { entry.MinArity, entry.MaxArity }; }
        }
        return { 0, 0 };
    }

private:
    struct BuiltinEntry {
        BuiltinOp Op;
        std::string_view Name;
        std::string_view Desc;
        uint16_t MinArity;
        uint16_t MaxArity;
        FormatRule Rule;
    };

    // Ordered to match BuiltinOp's own enumerator order exactly - RegisterAll
    // below relies on index I mapping directly to static_cast<BuiltinOp>(I).
    static constexpr std::array BuiltinEntries {
        BuiltinEntry{ BuiltinOp::Add, "+", "n-ary addition f(a,b,c,...) = a + b + c + ...", 2, 2, FormatRule::Infix },
        BuiltinEntry{ BuiltinOp::Mul, "*", "n-ary multiplication f(a,b,c,...) = a * b * c * ...", 2, 2, FormatRule::Infix },
        BuiltinEntry{ BuiltinOp::Sub, "-", "n-ary subtraction f(a,b,c,...) = a - (b + c + ...)", 2, 2, FormatRule::Infix },
        BuiltinEntry{ BuiltinOp::Div, "/", "n-ary division f(a,b,c,..) = a / (b * c * ...)", 2, 2, FormatRule::Infix },
        BuiltinEntry{ BuiltinOp::Fmin, "fmin", "minimum function f(a,b) = min(a,b)", 2, 2, FormatRule::MinMaxCall },
        BuiltinEntry{ BuiltinOp::Fmax, "fmax", "maximum function f(a,b) = max(a,b)", 2, 2, FormatRule::MinMaxCall },
        BuiltinEntry{ BuiltinOp::Aq, "aq", "analytical quotient f(a,b) = a / sqrt(1 + b^2)", 2, 2, FormatRule::Composite },
        BuiltinEntry{ BuiltinOp::Pow, "pow", "raise to power f(a,b) = a^b", 2, 2, FormatRule::PowerNotation },
        BuiltinEntry{ BuiltinOp::Powabs, "powabs", "raise absolute value to power f(a,b) = |a|^b", 2, 2, FormatRule::Composite },
        BuiltinEntry{ BuiltinOp::Abs, "abs", "absolute value function f(a) = abs(a)", 1, 1, FormatRule::GenericCall },
        BuiltinEntry{ BuiltinOp::Acos, "acos", "inverse cosine function f(a) = acos(a)", 1, 1, FormatRule::GenericCall },
        BuiltinEntry{ BuiltinOp::Asin, "asin", "inverse sine function f(a) = asin(a)", 1, 1, FormatRule::GenericCall },
        BuiltinEntry{ BuiltinOp::Atan, "atan", "inverse tangent function f(a) = atan(a)", 1, 1, FormatRule::GenericCall },
        BuiltinEntry{ BuiltinOp::Cbrt, "cbrt", "cube root function f(a) = cbrt(a)", 1, 1, FormatRule::GenericCall },
        BuiltinEntry{ BuiltinOp::Ceil, "ceil", "ceiling function f(a) = ceil(a)", 1, 1, FormatRule::GenericCall },
        BuiltinEntry{ BuiltinOp::Cos, "cos", "cosine function f(a) = cos(a)", 1, 1, FormatRule::GenericCall },
        BuiltinEntry{ BuiltinOp::Cosh, "cosh", "hyperbolic cosine function f(a) = cosh(a)", 1, 1, FormatRule::GenericCall },
        BuiltinEntry{ BuiltinOp::Exp, "exp", "e raised to the given power f(a) = e^a", 1, 1, FormatRule::GenericCall },
        BuiltinEntry{ BuiltinOp::Floor, "floor", "floor function f(a) = floor(a)", 1, 1, FormatRule::GenericCall },
        BuiltinEntry{ BuiltinOp::Log, "log", "natural (base e) logarithm f(a) = ln(a)", 1, 1, FormatRule::GenericCall },
        BuiltinEntry{ BuiltinOp::Logabs, "logabs", "natural (base e) logarithm of absolute value f(a) = ln(|a|)", 1, 1, FormatRule::Composite },
        BuiltinEntry{ BuiltinOp::Log1p, "log1p", "f(a) = ln(a + 1), accurate even when a is close to zero", 1, 1, FormatRule::Composite },
        BuiltinEntry{ BuiltinOp::Sin, "sin", "sine function f(a) = sin(a)", 1, 1, FormatRule::GenericCall },
        BuiltinEntry{ BuiltinOp::Sinh, "sinh", "hyperbolic sine function f(a) = sinh(a)", 1, 1, FormatRule::GenericCall },
        BuiltinEntry{ BuiltinOp::Sqrt, "sqrt", "square root function f(a) = sqrt(a)", 1, 1, FormatRule::GenericCall },
        BuiltinEntry{ BuiltinOp::Sqrtabs, "sqrtabs", "square root of absolute value function f(a) = sqrt(|a|)", 1, 1, FormatRule::Composite },
        BuiltinEntry{ BuiltinOp::Tan, "tan", "tangent function f(a) = tan(a)", 1, 1, FormatRule::GenericCall },
        BuiltinEntry{ BuiltinOp::Tanh, "tanh", "hyperbolic tangent function f(a) = tanh(a)", 1, 1, FormatRule::GenericCall },
        BuiltinEntry{ BuiltinOp::Square, "square", "square function f(a) = a^2", 1, 1, FormatRule::PowerNotation },
    };

    struct TerminalEntry {
        NodeType Type;
        std::string_view Name;
        std::string_view Desc;
    };

    // Function's entry is the generic fallback name/desc for an
    // unregistered user function (mirrors the old Dynamic entry) - a
    // registered user function's own name/desc (via Node::RegisterName)
    // takes precedence, this is only reached when that lookup misses.
    static constexpr std::array TerminalEntries {
        TerminalEntry{ NodeType::Function, "dyn", "user-defined function" },
        TerminalEntry{ NodeType::Constant, "constant", "a constant value" },
        TerminalEntry{ NodeType::Variable, "variable", "a dataset input with an associated weight" },
        TerminalEntry{ NodeType::Ref, "ref", "structural reference to another node (enables DAG sharing)" },
    };

    template<typename... Ts, std::size_t... I>
    static void RegisterAll(DispatchTable<Ts...>& dt, std::index_sequence<I...> /*unused*/)
    {
        (RegisterOne<static_cast<BuiltinOp>(I)>(dt), ...);
    }

    template<BuiltinOp Op, typename... Ts>
    static void RegisterOne(DispatchTable<Ts...>& dt)
    {
        auto const hash = static_cast<Operon::Hash>(Op);
        (RegisterForType<Op, Ts>(dt, hash), ...);
    }

    template<BuiltinOp Op, typename T, typename... Ts>
    requires Operon::Concepts::Arithmetic<T>
    static void RegisterForType(DispatchTable<Ts...>& dt, Operon::Hash hash)
    {
        constexpr auto S = DispatchTable<Ts...>::template BatchSize<T>;
        dt.template RegisterFunction<T>(
            hash,
            Dispatch::MakeFunctionCall<Op, T, S>(),
            Dispatch::MakeDiffCall<Op, T, S>());
    }

    template<BuiltinOp Op, typename T, typename... Ts>
    requires (!Operon::Concepts::Arithmetic<T>)
    static void RegisterForType(DispatchTable<Ts...>& /*dt*/, Operon::Hash /*hash*/)
    {
    }
};

} // namespace Operon

template<typename... Ts>
Operon::DispatchTable<Ts...>::DispatchTable()
{
    Operon::StandardLibrary::Register(*this);
}

#endif
