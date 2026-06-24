// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <catch2/catch_test_macros.hpp>

#include "operon/core/dataset.hpp"
#include "operon/core/node.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/tree.hpp"
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
        std::erase(inputs, ds.GetVariable("Y").value().Hash);
        PrimitiveSet pset;
        pset.SetConfig(PrimitiveSet::Arithmetic);
        return std::make_tuple(std::move(ds), std::move(inputs), std::move(pset));
    }
} // namespace

TEST_CASE("Zobrist - same tree yields same hash", "[zobrist]")
{
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist const cache(rng, MaxLength, inputs);

    BalancedTreeCreator const creator{&pset, inputs, /* bias= */ 0.0, MaxLength};
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
    Zobrist const cache(rng, MaxLength, inputs);

    BalancedTreeCreator const creator{&pset, inputs, /* bias= */ 0.0, MaxLength};
    Operon::NormalCoefficientInitializer const coeffInit;

    auto tree1 = creator(rng, 20, 1, MaxLength);
    auto tree2 = tree1; // identical structure

    coeffInit(rng, tree1);
    coeffInit(rng, tree2); // different coefficients

    REQUIRE(cache.ComputeHash(tree1) == cache.ComputeHash(tree2));
}

TEST_CASE("Zobrist - position sensitivity (deterministic)", "[zobrist]")
{
    // sin(cos(c)) and cos(sin(c)) have the same node types but at different
    // positions — the position-aware hash must distinguish them.
    Operon::RandomGenerator rng(Seed);
    Zobrist const cache(rng, MaxLength, {});  // no variables needed: trees use only constants

    // postfix: [Constant, Cos, Sin]  =>  sin(cos(c))
    Tree const tree1 = Tree({ Node(NodeType::Constant), Node(NodeType::Cos), Node(NodeType::Sin) }).UpdateNodes();
    // postfix: [Constant, Sin, Cos]  =>  cos(sin(c))
    Tree const tree2 = Tree({ Node(NodeType::Constant), Node(NodeType::Sin), Node(NodeType::Cos) }).UpdateNodes();

    REQUIRE(cache.ComputeHash(tree1) != cache.ComputeHash(tree2));
}

TEST_CASE("Zobrist - commuted variables yield different hashes", "[zobrist]")
{
    // Add(X, Y) and Add(Y, X) must hash differently so the JIT cache never
    // applies a compiled function to a tree with swapped variable columns.
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist const cache(rng, MaxLength, inputs);

    auto const varX = ds.GetVariable("X1").value();
    auto const varY = ds.GetVariable("X2").value();

    Node nX(NodeType::Variable); nX.HashValue = varX.Hash; nX.IsEnabled = true;
    Node nY(NodeType::Variable); nY.HashValue = varY.Hash; nY.IsEnabled = true;

    // postfix: [X, Y, Add] => Add(X, Y)
    Tree const treeXY = Tree({ nX, nY, Node(NodeType::Add) }).UpdateNodes();
    // postfix: [Y, X, Add] => Add(Y, X)
    Tree const treeYX = Tree({ nY, nX, Node(NodeType::Add) }).UpdateNodes();

    REQUIRE(cache.ComputeHash(treeXY) != cache.ComputeHash(treeYX));
}

TEST_CASE("Zobrist - TryGet returns false on miss", "[zobrist]")
{
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist const cache(rng, MaxLength, inputs);

    BalancedTreeCreator const creator{&pset, inputs, /* bias= */ 0.0, MaxLength};
    auto tree = creator(rng, 10, 1, MaxLength);
    auto hash = cache.ComputeHash(tree);

    Operon::Vector<Operon::Scalar> val;
    REQUIRE_FALSE(cache.TryGet(hash, val));
    REQUIRE(cache.Hits() == 0);
}

TEST_CASE("Zobrist - Insert then TryGet roundtrip", "[zobrist]")
{
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist cache(rng, MaxLength, inputs);

    BalancedTreeCreator const creator{&pset, inputs, /* bias= */ 0.0, MaxLength};
    auto tree = creator(rng, 10, 1, MaxLength);
    auto hash = cache.ComputeHash(tree);

    Operon::Vector<Operon::Scalar> const stored = { Operon::Scalar{0.5}, Operon::Scalar{1.0} };
    cache.Insert(hash, stored);
    REQUIRE(cache.Size() == 1);

    Operon::Vector<Operon::Scalar> retrieved(2);
    REQUIRE(cache.TryGet(hash, retrieved));
    REQUIRE(cache.Hits() == 1);
    REQUIRE(retrieved[0] == stored[0]);
    REQUIRE(retrieved[1] == stored[1]);
}

TEST_CASE("Zobrist - Clear resets table and hit counter", "[zobrist]")
{
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist cache(rng, MaxLength, inputs);

    BalancedTreeCreator const creator{&pset, inputs, /* bias= */ 0.0, MaxLength};
    auto tree = creator(rng, 10, 1, MaxLength);
    auto hash = cache.ComputeHash(tree);

    Operon::Vector<Operon::Scalar> const val = { Operon::Scalar{0.1} };
    cache.Insert(hash, val);

    Operon::Vector<Operon::Scalar> tmp;
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
    Zobrist cache(rng, MaxLength, inputs);

    BalancedTreeCreator const creator{&pset, inputs, /* bias= */ 0.0, MaxLength};
    auto tree = creator(rng, 10, 1, MaxLength);
    auto hash = cache.ComputeHash(tree);

    Operon::Vector<Operon::Scalar> const val = { Operon::Scalar{0.3} };
    cache.Insert(hash, val);
    cache.Insert(hash, val);
    cache.Insert(hash, val);

    REQUIRE(cache.Size() == 1); // only one entry
}

TEST_CASE("Zobrist - different Optimize flags yield different hashes", "[zobrist]")
{
    Operon::RandomGenerator rng(Seed);
    Zobrist const cache(rng, MaxLength, {});

    // Two identical trees: sin(constant)
    // One has the constant optimizable, the other does not.
    Node c1(NodeType::Constant); c1.Value = 1.0f; c1.Optimize = true;
    Node c2(NodeType::Constant); c2.Value = 1.0f; c2.Optimize = false;

    Tree const tree1 = Tree({ c1, Node(NodeType::Sin) }).UpdateNodes();
    Tree const tree2 = Tree({ c2, Node(NodeType::Sin) }).UpdateNodes();

    REQUIRE(cache.ComputeHash(tree1) != cache.ComputeHash(tree2));
}

} // namespace Operon::Test
