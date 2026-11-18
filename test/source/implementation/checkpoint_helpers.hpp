// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_TEST_CHECKPOINT_HELPERS_HPP
#define OPERON_TEST_CHECKPOINT_HELPERS_HPP

#include <algorithm>
#include <catch2/catch_message.hpp>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "operon/algorithms/ga_base.hpp"
#include "operon/core/serialization.hpp"
#include "operon/hash/zobrist.hpp"

// What a command-line checkpoint stores and what --resume restores, step by step as cli/source/util.cpp does
// (MaybeSaveCheckpoint, ResumeFromCheckpoint), without the option parsing. A resumed run built with RestoreCheckpoint
// starts from nothing but the serialized bytes, the way a new process does.
namespace Operon::Test {

// Equality of two long sequences (signatures, serialized state) for use inside CHECK. On a mismatch Catch would print
// both sequences, which can be megabytes; this reports only the sizes and the first differing index.
template <typename Sequence> auto Same(Sequence const& a, Sequence const& b) -> bool
{
    if (a == b) {
        return true;
    }
    auto const [x, y] = std::ranges::mismatch(a, b);
    UNSCOPED_INFO("sequences differ: sizes " << a.size() << " and " << b.size() << ", first difference at index "
                                             << (x - a.begin()));
    return false;
}

inline auto CaptureCheckpoint(GeneticAlgorithmBase const& algo, RandomGenerator const& rng) -> std::string
{
    Serialization::Checkpoint cp;
    cp.RngState = rng.state();
    cp.Generation = algo.Generation();
    cp.Population.assign(algo.Parents().begin(), algo.Parents().end());
    for (auto const& worker : algo.WorkerRngs()) {
        cp.WorkerRngStates.push_back(worker.state());
    }
    if (auto const* cache = algo.GetGenerator()->Cache()) {
        cp.Cache = Serialization::ToCacheCheckpoint(cache->Export());
    }
    return Serialization::ToBeve(cp);
}

inline void RestoreCheckpoint(GeneticAlgorithmBase& algo, RandomGenerator& rng, std::string_view bytes)
{
    auto cp = Serialization::CheckpointFromBeve(bytes);
    if (!cp) {
        throw std::runtime_error { "checkpoint could not be decoded" };
    }
    rng.set_state(cp->RngState);
    algo.Generation() = cp->Generation;
    if (auto* cache = algo.GetGenerator()->Cache()) {
        if (cp->Cache) {
            cache->Import(
                Serialization::ToZobristState(*cp->Cache), algo.GetGenerator()->Evaluator()->ObjectiveCount());
        } else {
            cache->SetGeneration(cp->Generation);
        }
    }
    auto parents = algo.Parents();
    for (std::size_t i = 0; i < cp->Population.size(); ++i) {
        parents[i] = std::move(cp->Population[i]);
    }
    auto& workers = algo.WorkerRngs();
    workers.clear();
    for (auto const& state : cp->WorkerRngStates) {
        RandomGenerator worker(0ULL);
        worker.set_state(state);
        workers.push_back(std::move(worker));
    }
    algo.IsFitted() = true;
}

} // namespace Operon::Test

#endif
