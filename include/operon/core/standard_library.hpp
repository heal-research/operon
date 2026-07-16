// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_STANDARD_LIBRARY_HPP
#define OPERON_STANDARD_LIBRARY_HPP

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
