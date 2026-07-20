// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_HASH_REGISTRY_HPP
#define OPERON_HASH_REGISTRY_HPP

#include <stdexcept>
#include <utility>

#include "types.hpp"

// A small, generic hash-keyed registry shared by the symbolic-differentiation
// (tree_diff.cpp), interval/affine (interval_evaluator.hpp/affine_evaluator.hpp),
// and JIT-codegen (jit_compiler.cpp) registries. Each of those instantiates
// HashRegistry<Fn> for its own distinct Fn type, kept as separate C++ types
// rather than one shared per-hash struct: they have different compile-time
// availability (JIT is conditional on HAVE_ASMJIT; the others are always
// present) and different consumers, so bundling them would leak
// backend-specific types (e.g. asmjit) onto callers that don't care.
//
// This header has no dependency on any of its consumers' Fn types.

namespace Operon {

template<typename Fn>
class HashRegistry {
public:
    // Write-once: throws on a hash that's already registered. Deliberately
    // stricter than DispatchTable::RegisterFunction (which overwrites), so a
    // repeat registration is a caught mistake rather than a silent, later
    // divergence between what's registered and what's cached/assumed
    // elsewhere under the same hash.
    void Register(Operon::Hash hash, Fn fn)
    {
        auto [it, inserted] = map_.try_emplace(hash, std::move(fn));
        if (!inserted) {
            throw std::invalid_argument("HashRegistry: hash is already registered");
        }
    }

    [[nodiscard]] auto TryGet(Operon::Hash hash) const -> Fn const*
    {
        auto it = map_.find(hash);
        return it != map_.end() ? &it->second : nullptr;
    }

    [[nodiscard]] auto Contains(Operon::Hash hash) const -> bool { return map_.contains(hash); }

private:
    Operon::Map<Operon::Hash, Fn> map_;
};

} // namespace Operon

#endif
