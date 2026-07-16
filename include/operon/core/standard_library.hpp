// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_STANDARD_LIBRARY_HPP
#define OPERON_STANDARD_LIBRARY_HPP

#include <array>
#include <string_view>

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

struct StandardLibrary {
    static void RegisterNames()
    {
        static auto const registered = [] {
            for (auto const& entry : Entries) {
                Node::RegisterName(Node(entry.Type).HashValue, std::string(entry.Name), std::string(entry.Desc));
            }
            return true;
        }();
        static_cast<void>(registered);
    }

    // Register all built-in math ops (everything except Dynamic, Constant,
    // Variable, Ref) into `dt`, for every scalar type DTable supports.
    template<typename... Ts>
    static void Register(DispatchTable<Ts...>& dt)
    {
        RegisterNames();
        RegisterAll(dt, std::make_index_sequence<NodeTypes::Count - 4>{});
    }

private:
    struct Entry {
        NodeType Type;
        std::string_view Name;
        std::string_view Desc;
    };

    static constexpr std::array Entries {
        Entry{ NodeType::Add, "+", "n-ary addition f(a,b,c,...) = a + b + c + ..." },
        Entry{ NodeType::Mul, "*", "n-ary multiplication f(a,b,c,...) = a * b * c * ..." },
        Entry{ NodeType::Sub, "-", "n-ary subtraction f(a,b,c,...) = a - (b + c + ...)" },
        Entry{ NodeType::Div, "/", "n-ary division f(a,b,c,..) = a / (b * c * ...)" },
        Entry{ NodeType::Fmin, "fmin", "minimum function f(a,b) = min(a,b)" },
        Entry{ NodeType::Fmax, "fmax", "maximum function f(a,b) = max(a,b)" },
        Entry{ NodeType::Aq, "aq", "analytical quotient f(a,b) = a / sqrt(1 + b^2)" },
        Entry{ NodeType::Pow, "pow", "raise to power f(a,b) = a^b" },
        Entry{ NodeType::Powabs, "powabs", "raise absolute value to power f(a,b) = |a|^b" },
        Entry{ NodeType::Abs, "abs", "absolute value function f(a) = abs(a)" },
        Entry{ NodeType::Acos, "acos", "inverse cosine function f(a) = acos(a)" },
        Entry{ NodeType::Asin, "asin", "inverse sine function f(a) = asin(a)" },
        Entry{ NodeType::Atan, "atan", "inverse tangent function f(a) = atan(a)" },
        Entry{ NodeType::Cbrt, "cbrt", "cube root function f(a) = cbrt(a)" },
        Entry{ NodeType::Ceil, "ceil", "ceiling function f(a) = ceil(a)" },
        Entry{ NodeType::Cos, "cos", "cosine function f(a) = cos(a)" },
        Entry{ NodeType::Cosh, "cosh", "hyperbolic cosine function f(a) = cosh(a)" },
        Entry{ NodeType::Exp, "exp", "e raised to the given power f(a) = e^a" },
        Entry{ NodeType::Floor, "floor", "floor function f(a) = floor(a)" },
        Entry{ NodeType::Log, "log", "natural (base e) logarithm f(a) = ln(a)" },
        Entry{ NodeType::Logabs, "logabs", "natural (base e) logarithm of absolute value f(a) = ln(|a|)" },
        Entry{ NodeType::Log1p, "log1p", "f(a) = ln(a + 1), accurate even when a is close to zero" },
        Entry{ NodeType::Sin, "sin", "sine function f(a) = sin(a)" },
        Entry{ NodeType::Sinh, "sinh", "hyperbolic sine function f(a) = sinh(a)" },
        Entry{ NodeType::Sqrt, "sqrt", "square root function f(a) = sqrt(a)" },
        Entry{ NodeType::Sqrtabs, "sqrtabs", "square root of absolute value function f(a) = sqrt(|a|)" },
        Entry{ NodeType::Tan, "tan", "tangent function f(a) = tan(a)" },
        Entry{ NodeType::Tanh, "tanh", "hyperbolic tangent function f(a) = tanh(a)" },
        Entry{ NodeType::Square, "square", "square function f(a) = a^2" },
        Entry{ NodeType::Dynamic, "dyn", "user-defined function" },
        Entry{ NodeType::Constant, "constant", "a constant value" },
        Entry{ NodeType::Variable, "variable", "a dataset input with an associated weight" },
        Entry{ NodeType::Ref, "ref", "structural reference to another node (enables DAG sharing)" },
    };

    template<typename... Ts, std::size_t... I>
    static void RegisterAll(DispatchTable<Ts...>& dt, std::index_sequence<I...> /*unused*/)
    {
        (RegisterOne<static_cast<NodeType>(I)>(dt), ...);
    }

    template<NodeType Type, typename... Ts>
    static void RegisterOne(DispatchTable<Ts...>& dt)
    {
        auto const hash = Node(Type).HashValue;
        (RegisterForType<Type, Ts>(dt, hash), ...);
    }

    template<NodeType Type, typename T, typename... Ts>
    requires Operon::Concepts::Arithmetic<T>
    static void RegisterForType(DispatchTable<Ts...>& dt, Operon::Hash hash)
    {
        constexpr auto S = DispatchTable<Ts...>::template BatchSize<T>;
        dt.template RegisterFunction<T>(
            hash,
            Dispatch::MakeFunctionCall<Type, T, S>(),
            Dispatch::MakeDiffCall<Type, T, S>());
    }

    template<NodeType Type, typename T, typename... Ts>
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
