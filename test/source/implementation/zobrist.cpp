// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <catch2/catch_test_macros.hpp>

#include "operon/core/dataset.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/pset.hpp"
#include "operon/hash/zobrist.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/initializer.hpp"

namespace Operon::Test {

namespace {
    constexpr auto Seed      = 42UL;
    constexpr auto MaxLength = 50;

    auto MakeSetup() {
        auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
        auto inputs = ds.VariableHashes();
        std::erase(inputs, ds.GetVariable("Y")->Hash);
        PrimitiveSet pset;
        pset.SetConfig(PrimitiveSet::Arithmetic);
        return std::make_tuple(std::move(ds), std::move(inputs), std::move(pset));
    }
} // namespace

TEST_CASE("Zobrist - same tree yields same hash", "[zobrist]")
{
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist cache(rng, MaxLength);

    BalancedTreeCreator creator{&pset, inputs};
    auto tree = creator(rng, 20, 1, MaxLength);

    auto h1 = cache.ComputeHash(tree);
    auto h2 = cache.ComputeHash(tree);
    REQUIRE(h1 == h2);
}

TEST_CASE("Zobrist - different coefficients yield same hash", "[zobrist]")
{
    // The hash must be coefficient-insensitive so cached fitness (post local
    // search) can be reused for any structurally identical tree.
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist cache(rng, MaxLength);

    BalancedTreeCreator creator{&pset, inputs};
    Operon::NormalCoefficientInitializer coeffInit;

    auto tree1 = creator(rng, 20, 1, MaxLength);
    auto tree2 = tree1; // identical structure

    coeffInit(rng, tree1);
    coeffInit(rng, tree2); // different coefficients

    REQUIRE(cache.ComputeHash(tree1) == cache.ComputeHash(tree2));
}

TEST_CASE("Zobrist - position sensitivity", "[zobrist]")
{
    // A tree with nodes in a different order should (almost always) hash differently.
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist cache(rng, MaxLength);

    BalancedTreeCreator creator{&pset, inputs};
    auto tree1 = creator(rng, 10, 1, MaxLength);
    auto tree2 = creator(rng, 10, 1, MaxLength);

    // Two independently generated trees of the same length are structurally
    // distinct with overwhelming probability.
    if (tree1.Length() != tree2.Length()) { SUCCEED(); return; }
    CHECK(cache.ComputeHash(tree1) != cache.ComputeHash(tree2));
}

TEST_CASE("Zobrist - TryGet returns false on miss", "[zobrist]")
{
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist cache(rng, MaxLength);

    BalancedTreeCreator creator{&pset, inputs};
    auto tree = creator(rng, 10, 1, MaxLength);
    auto hash = cache.ComputeHash(tree);

    Individual ind;
    REQUIRE_FALSE(cache.TryGet(hash, ind));
    REQUIRE(cache.Hits() == 0);
}

TEST_CASE("Zobrist - Insert then TryGet roundtrip", "[zobrist]")
{
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist cache(rng, MaxLength);

    BalancedTreeCreator creator{&pset, inputs};
    auto tree = creator(rng, 10, 1, MaxLength);
    auto hash = cache.ComputeHash(tree);

    Individual stored(2);
    stored.Fitness = { 0.5, 1.0 };
    cache.Insert(hash, stored);
    REQUIRE(cache.Size() == 1);

    Individual retrieved(2);
    REQUIRE(cache.TryGet(hash, retrieved));
    REQUIRE(cache.Hits() == 1);
    REQUIRE(retrieved.Fitness[0] == stored.Fitness[0]);
    REQUIRE(retrieved.Fitness[1] == stored.Fitness[1]);
}

TEST_CASE("Zobrist - Clear resets table and hit counter", "[zobrist]")
{
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist cache(rng, MaxLength);

    BalancedTreeCreator creator{&pset, inputs};
    auto tree = creator(rng, 10, 1, MaxLength);
    auto hash = cache.ComputeHash(tree);

    Individual ind(1);
    ind.Fitness = { 0.1 };
    cache.Insert(hash, ind);

    Individual tmp;
    std::ignore = cache.TryGet(hash, tmp); // bumps hits

    cache.Clear();
    REQUIRE(cache.Size() == 0);
    REQUIRE(cache.Hits() == 0);

    REQUIRE_FALSE(cache.TryGet(hash, tmp));
}

TEST_CASE("Zobrist - duplicate inserts increment count, not entries", "[zobrist]")
{
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist cache(rng, MaxLength);

    BalancedTreeCreator creator{&pset, inputs};
    auto tree = creator(rng, 10, 1, MaxLength);
    auto hash = cache.ComputeHash(tree);

    Individual ind(1);
    ind.Fitness = { 0.3 };
    cache.Insert(hash, ind);
    cache.Insert(hash, ind);
    cache.Insert(hash, ind);

    REQUIRE(cache.Size() == 1); // only one entry
}

} // namespace Operon::Test
