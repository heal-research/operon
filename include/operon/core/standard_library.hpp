// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_STANDARD_LIBRARY_HPP
#define OPERON_STANDARD_LIBRARY_HPP

#include "dispatch.hpp"
#include "node.hpp"

// standard_library.hpp: registers the built-in math ops into a DispatchTable
// via the same public RegisterFunction<T> API used for user-defined Dynamic
// functions (see symbol_library.hpp), instead of the compile-time
// index_sequence loop DispatchTable's default constructor uses.
//
// This is Phase 1 of the generic-node-registry migration
// (docs/research plan: NodeType -> {Constant,Variable,Ref,Unary,Binary,Nary}).
// It is purely additive: it does not change DispatchTable's default
// construction path, Node::Type values, or any existing behavior. Its only
// purpose is to prove that built-in ops can be populated through the exact
// same runtime registration mechanism as custom functions, with identical
// underlying callables (see test/source/implementation/standard_library.cpp
// for the equivalence proof), before anything downstream is migrated to
// depend on it.
//
// Each built-in's Callable/CallableDiff is built via Dispatch::MakeFunctionCall
// / MakeDiffCall<Type,T,S> - the exact same compile-time-specialized kernels
// (NaryOp/BinaryOp/UnaryOp wrapping Func<T,Type,C,S>/Diff<T,Type,S>) that
// DispatchTable's constructor already uses. Only the population code path
// differs; the compiled kernel for a given (Type,T,S) is identical either way.

namespace Operon {

struct StandardLibrary {
    // Register all built-in math ops (everything except Dynamic, Constant,
    // Variable, Ref) into `dt`, for every scalar type DTable supports.
    template<typename... Ts>
    static void Register(DispatchTable<Ts...>& dt)
    {
        RegisterAll(dt, std::make_index_sequence<NodeTypes::Count - 4>{});
    }

private:
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
    static void RegisterForType(DispatchTable<Ts...>& dt, Operon::Hash hash)
    {
        constexpr auto S = DispatchTable<Ts...>::template BatchSize<T>;
        dt.template RegisterFunction<T>(
            hash,
            Dispatch::MakeFunctionCall<Type, T, S>(),
            Dispatch::MakeDiffCall<Type, T, S>());
    }
};

} // namespace Operon

#endif
