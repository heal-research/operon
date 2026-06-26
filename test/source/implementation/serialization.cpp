// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "operon/core/individual.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/serialization.hpp"
#include "operon/core/types.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/initializer.hpp"

namespace Operon::Test {
namespace {

// Minimal variable hash used across all tree-building helpers.
static constexpr Operon::Hash VarHash = 1234567890ULL;

auto MakeTree(Operon::RandomGenerator& rng) -> Operon::Tree
{
    Operon::PrimitiveSet pset;
    pset.SetConfig(PrimitiveSet::Arithmetic);
    std::vector<Operon::Hash> vars{ VarHash };
    Operon::BalancedTreeCreator creator{ &pset, vars, /*bias=*/0.0, /*maxLength=*/20 };
    Operon::UniformTreeInitializer treeInit(&creator);
    treeInit.ParameterizeDistribution(5UL, 20UL);
    treeInit.SetMaxDepth(6);
    Operon::UniformCoefficientInitializer coeffInit;
    coeffInit.ParameterizeDistribution(Operon::Scalar{-5}, Operon::Scalar{+5});
    auto tree = treeInit(rng);
    coeffInit(rng, tree);
    return tree;
}

auto MakeIndividual(Operon::RandomGenerator& rng, std::size_t nObjectives = 2) -> Operon::Individual
{
    Operon::Individual ind;
    ind.Genotype = MakeTree(rng);
    ind.Fitness.resize(nObjectives);
    std::uniform_real_distribution<Operon::Scalar> dist{-1, 1};
    for (auto& f : ind.Fitness) { f = dist(rng); }
    ind.Rank     = 0;
    ind.Distance = Operon::Scalar{1};
    return ind;
}

auto TreesEqual(Operon::Tree const& a, Operon::Tree const& b) -> bool
{
    if (a.Length() != b.Length()) { return false; }
    for (std::size_t i = 0; i < a.Length(); ++i) {
        auto const& na = a[i];
        auto const& nb = b[i];
        if (na.Type      != nb.Type)      { return false; }
        if (na.HashValue != nb.HashValue) { return false; }
        if (na.IsEnabled != nb.IsEnabled) { return false; }
        if (na.Optimize  != nb.Optimize)  { return false; }
        if (na.RefTo     != nb.RefTo)     { return false; }
        if (std::abs(na.Value - nb.Value) > Operon::Scalar{1e-6}) { return false; }
    }
    return true;
}

auto IndividualsEqual(Operon::Individual const& a, Operon::Individual const& b) -> bool
{
    if (!TreesEqual(a.Genotype, b.Genotype))       { return false; }
    if (a.Fitness.size() != b.Fitness.size())      { return false; }
    for (std::size_t i = 0; i < a.Fitness.size(); ++i) {
        if (std::abs(a.Fitness[i] - b.Fitness[i]) > Operon::Scalar{1e-6}) { return false; }
    }
    return a.Rank == b.Rank && std::abs(a.Distance - b.Distance) < Operon::Scalar{1e-6};
}

} // namespace

TEST_CASE("Tree JSON round-trip", "[serialization]")
{
    Operon::RandomGenerator rng(42);
    for (auto i = 0; i < 20; ++i) {
        auto original = MakeTree(rng);
        auto json     = Operon::Serialization::ToJson(original);
        auto restored = Operon::Serialization::TreeFromJson(json);
        CHECK(TreesEqual(original, restored));
    }
}

TEST_CASE("Individual JSON round-trip", "[serialization]")
{
    Operon::RandomGenerator rng(42);
    for (auto i = 0; i < 20; ++i) {
        auto original = MakeIndividual(rng);
        auto json     = Operon::Serialization::ToJson(original);
        auto restored = Operon::Serialization::IndividualFromJson(json);
        CHECK(IndividualsEqual(original, restored));
    }
}

TEST_CASE("Pareto front JSON round-trip", "[serialization]")
{
    Operon::RandomGenerator rng(42);
    Operon::Vector<Operon::Individual> front(10);
    for (auto& ind : front) { ind = MakeIndividual(rng); }

    auto json = Operon::Serialization::ToJson(std::span<Operon::Individual const>(front));
    CHECK(!json.empty());
    CHECK(json.front() == '[');

    for (auto const& original : front) {
        auto elemJson = Operon::Serialization::ToJson(original);
        auto restored = Operon::Serialization::IndividualFromJson(elemJson);
        CHECK(IndividualsEqual(original, restored));
    }
}

TEST_CASE("Tree BEVE round-trip", "[serialization]")
{
    Operon::RandomGenerator rng(42);
    for (auto i = 0; i < 20; ++i) {
        auto original = MakeTree(rng);
        auto beve     = Operon::Serialization::ToBeve(original);
        auto restored = Operon::Serialization::TreeFromBeve(beve);
        CHECK(TreesEqual(original, restored));
    }
}

TEST_CASE("Individual BEVE round-trip", "[serialization]")
{
    Operon::RandomGenerator rng(42);
    for (auto i = 0; i < 20; ++i) {
        auto original = MakeIndividual(rng);
        auto beve     = Operon::Serialization::ToBeve(original);
        auto restored = Operon::Serialization::IndividualFromBeve(beve);
        CHECK(IndividualsEqual(original, restored));
    }
}

TEST_CASE("Checkpoint BEVE round-trip", "[serialization]")
{
    Operon::RandomGenerator rng(99);
    for (auto i = 0; i < 1000; ++i) { (void)rng(); }

    Operon::Serialization::Checkpoint original;
    original.RngState   = rng.state();
    original.Generation = 42;
    original.Population.resize(50);
    for (auto& ind : original.Population) { ind = MakeIndividual(rng); }
    original.WorkerRngStates.resize(8);
    for (auto& s : original.WorkerRngStates) {
        for (auto i = 0; i < 100; ++i) { (void)rng(); }
        s = rng.state();
    }

    auto beve     = Operon::Serialization::ToBeve(original);
    auto restored = Operon::Serialization::CheckpointFromBeve(beve);

    CHECK(restored.RngState   == original.RngState);
    CHECK(restored.Generation == original.Generation);
    REQUIRE(restored.Population.size() == original.Population.size());
    for (std::size_t i = 0; i < original.Population.size(); ++i) {
        CHECK(IndividualsEqual(original.Population[i], restored.Population[i]));
    }
    REQUIRE(restored.WorkerRngStates.size() == original.WorkerRngStates.size());
    for (std::size_t i = 0; i < original.WorkerRngStates.size(); ++i) {
        CHECK(restored.WorkerRngStates[i] == original.WorkerRngStates[i]);
    }
}

TEST_CASE("Checkpoint file save/load round-trip", "[serialization]")
{
    Operon::RandomGenerator rng(7);
    for (auto i = 0; i < 500; ++i) { (void)rng(); }

    Operon::Serialization::Checkpoint original;
    original.RngState   = rng.state();
    original.Generation = 10;
    original.Population.resize(20);
    for (auto& ind : original.Population) { ind = MakeIndividual(rng); }
    original.WorkerRngStates.resize(4);
    for (auto& s : original.WorkerRngStates) {
        for (auto i = 0; i < 100; ++i) { (void)rng(); }
        s = rng.state();
    }

    auto const path = (std::filesystem::temp_directory_path() / "operon_test_checkpoint.beve").string();
    Operon::Serialization::SaveCheckpoint(original, path);
    auto restored = Operon::Serialization::LoadCheckpoint(path);

    CHECK(restored.RngState   == original.RngState);
    CHECK(restored.Generation == original.Generation);
    REQUIRE(restored.Population.size() == original.Population.size());
    for (std::size_t i = 0; i < original.Population.size(); ++i) {
        CHECK(IndividualsEqual(original.Population[i], restored.Population[i]));
    }
    REQUIRE(restored.WorkerRngStates.size() == original.WorkerRngStates.size());
    for (std::size_t i = 0; i < original.WorkerRngStates.size(); ++i) {
        CHECK(restored.WorkerRngStates[i] == original.WorkerRngStates[i]);
    }
}

} // namespace Operon::Test
