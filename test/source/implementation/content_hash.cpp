// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include <unordered_set>

#include "operon/core/dataset.hpp"
#include "operon/core/node.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/tree.hpp"
#include "operon/hash/content_hash.hpp"
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

TEST_CASE("ContentHash - determinism", "[content_hash]")
{
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist const zobrist(rng, MaxLength, inputs);

    BalancedTreeCreator const creator{&pset, inputs, /* bias= */ 0.0, MaxLength};
    Operon::CoefficientInitializer<std::uniform_real_distribution<Operon::Scalar>> const initializer;

    auto tree = creator(rng, 20, 1, MaxLength);
    initializer(rng, tree);

    auto h1 = ComputeContentHash(tree, zobrist);
    auto h2 = ComputeContentHash(tree, zobrist);
    REQUIRE(h1 == h2);
}

TEST_CASE("ContentHash - commutative invariance", "[content_hash]")
{
    // Add(X, Y) and Add(Y, X) must hash identically - the whole point of this
    // hash is to collapse commutative reorderings for dedup purposes.
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist const zobrist(rng, MaxLength, inputs);

    auto const varX = ds.GetVariable("X1").value();
    auto const varY = ds.GetVariable("X2").value();

    Node nX(NodeType::Variable); nX.HashValue = varX.Hash;
    Node nY(NodeType::Variable); nY.HashValue = varY.Hash;

    Tree const treeXY = Tree({ nX, nY, Node(NodeType::Add) }).UpdateNodes();
    Tree const treeYX = Tree({ nY, nX, Node(NodeType::Add) }).UpdateNodes();

    REQUIRE(ComputeContentHash(treeXY, zobrist) == ComputeContentHash(treeYX, zobrist));
}

TEST_CASE("ContentHash - non-commutative sensitivity", "[content_hash]")
{
    // Sub(X, Y) and Sub(Y, X) are different expressions and must hash
    // differently - only commutative operators get their children reordered.
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist const zobrist(rng, MaxLength, inputs);

    auto const varX = ds.GetVariable("X1").value();
    auto const varY = ds.GetVariable("X2").value();

    Node nX(NodeType::Variable); nX.HashValue = varX.Hash;
    Node nY(NodeType::Variable); nY.HashValue = varY.Hash;

    Tree const treeXY = Tree({ nX, nY, Node(NodeType::Sub) }).UpdateNodes();
    Tree const treeYX = Tree({ nY, nX, Node(NodeType::Sub) }).UpdateNodes();

    REQUIRE(ComputeContentHash(treeXY, zobrist) != ComputeContentHash(treeYX, zobrist));
}

TEST_CASE("ContentHash - position independence", "[content_hash]")
{
    // The same subexpression (x+y) embedded at different absolute array
    // positions in two different larger trees must hash identically - this is
    // the key property distinguishing this hash from Zobrist::ComputeHash
    // (which is deliberately position-*dependent* for its transposition-cache
    // role). Build (x+y)*z [subtree root at index 2] and z*(x+y) [subtree root
    // at index 3] and compare the scratch value at each subtree's own root.
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist const zobrist(rng, MaxLength, inputs);

    auto const varX = ds.GetVariable("X1").value();
    auto const varY = ds.GetVariable("X2").value();
    auto const varZ = ds.GetVariable("X3").value();

    Node nX(NodeType::Variable); nX.HashValue = varX.Hash;
    Node nY(NodeType::Variable); nY.HashValue = varY.Hash;
    Node nZ(NodeType::Variable); nZ.HashValue = varZ.Hash;

    // (x+y)*z : postfix [x, y, Add, z, Mul] - subtree root (Add) at index 2
    Tree const treeA = Tree({ nX, nY, Node(NodeType::Add), nZ, Node(NodeType::Mul) }).UpdateNodes();
    // z*(x+y) : postfix [z, x, y, Add, Mul] - subtree root (Add) at index 3
    Tree const treeB = Tree({ nZ, nX, nY, Node(NodeType::Add), Node(NodeType::Mul) }).UpdateNodes();

    std::vector<Operon::Hash> scratchA(treeA.Nodes().size());
    std::vector<Operon::Hash> scratchB(treeB.Nodes().size());
    std::ignore = ComputeContentHash(treeA, zobrist, scratchA);
    std::ignore = ComputeContentHash(treeB, zobrist, scratchB);

    REQUIRE(scratchA[2] == scratchB[3]);
}

TEST_CASE("ContentHash - scratch overload matches allocating overload", "[content_hash]")
{
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist const zobrist(rng, MaxLength, inputs);

    BalancedTreeCreator const creator{&pset, inputs, /* bias= */ 0.0, MaxLength};
    auto tree = creator(rng, 20, 1, MaxLength);

    std::vector<Operon::Hash> scratch(tree.Nodes().size());
    auto viaScratch = ComputeContentHash(tree, zobrist, scratch);
    auto viaAlloc = ComputeContentHash(tree, zobrist);

    REQUIRE(viaScratch == viaAlloc);
}

TEST_CASE("ContentHash - coefficient insensitivity", "[content_hash]")
{
    // Mirrors Zobrist's own "different coefficients yield same hash" test:
    // this hash never looks at Node::Value, only type/variable-identity/
    // commutative ordering, so structurally identical trees with different
    // constant values must hash identically (constants are optimizable
    // weights to be re-fit, not distinct symbols for dedup purposes).
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist const zobrist(rng, MaxLength, inputs);

    BalancedTreeCreator const creator{&pset, inputs, /* bias= */ 0.0, MaxLength};
    Operon::NormalCoefficientInitializer const coeffInit;

    auto tree1 = creator(rng, 20, 1, MaxLength);
    auto tree2 = tree1; // identical structure

    coeffInit(rng, tree1);
    coeffInit(rng, tree2); // different coefficients

    REQUIRE(ComputeContentHash(tree1, zobrist) == ComputeContentHash(tree2, zobrist));
}

TEST_CASE("ContentHash - Optimize-flag insensitivity", "[content_hash]")
{
    // Unlike Zobrist::ComputeHash (whole-tree transposition cache, where
    // Optimize legitimately participates), this hash must ignore which
    // leaves are marked optimizable: two structurally-identical subtrees
    // differing only in Optimize flags are the same dedup candidate, since
    // weights are re-fit rather than treated as distinct symbols.
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist const zobrist(rng, MaxLength, inputs);

    BalancedTreeCreator const creator{&pset, inputs, /* bias= */ 0.0, MaxLength};

    auto tree1 = creator(rng, 20, 1, MaxLength);
    auto tree2 = tree1; // identical structure and coefficients

    for (auto& n : tree2.Nodes()) {
        if (n.IsLeaf()) { n.Optimize = !n.Optimize; }
    }

    REQUIRE(ComputeContentHash(tree1, zobrist) == ComputeContentHash(tree2, zobrist));
}

TEST_CASE("ContentHash - Ref-aware (structural sharing doesn't change the hash)", "[content_hash]")
{
    // Mirrors Tree::Hash()'s own Ref-normalization (tree.cpp): x*x built with
    // an explicit Ref to the first x (structural sharing) must hash the same
    // as x*x built with two independent Variable nodes (no sharing) - both
    // are the same expression, and content-hash dedup must not distinguish
    // them just because one representation happens to use Ref.
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist const zobrist(rng, MaxLength, inputs);

    auto const varX = ds.GetVariable("X1").value();
    Node nX(NodeType::Variable); nX.HashValue = varX.Hash;

    // x * x, no sharing: two independent Variable nodes.
    Tree const noRef = Tree({ nX, nX, Node(NodeType::Mul) }).UpdateNodes();

    // x * Ref(x), sharing the first x via a backward Ref.
    Tree const withRef = Tree({ nX, Node::Ref(0), Node(NodeType::Mul) }).UpdateNodes();

    REQUIRE(ComputeContentHash(noRef, zobrist) == ComputeContentHash(withRef, zobrist));
}

TEST_CASE("ContentHash - collision rate sanity check", "[content_hash]")
{
    // Coefficient values don't affect this hash (see "coefficient insensitivity"
    // above), so vary only tree *shape* here - applying random coefficients
    // would make genuinely-distinct-shape trees collide by design, which isn't
    // a hash-quality signal, just the intended coefficient-insensitivity.
    // Size range starts well above 1 so that duplicate *shapes* arising purely
    // by chance (e.g. two draws both producing a single-variable tree) are
    // implausible - otherwise an observed "collision" could just mean two
    // draws produced the same small tree, not that the hash actually collided.
    size_t const n = 5000;
    size_t const minLength = 20;
    size_t const maxLength = 100;
    size_t const minDepth = 1;
    size_t const maxDepth = 1000;

    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rd(1234);
    Zobrist const zobrist(rd, static_cast<int>(maxLength), inputs);

    std::uniform_int_distribution<size_t> sizeDistribution(minLength, maxLength);
    auto const btc = BalancedTreeCreator{&pset, inputs, /* bias= */ 0.0, maxLength};

    std::unordered_set<Operon::Hash> seen;
    for (size_t i = 0; i < n; ++i) {
        auto tree = btc(rd, sizeDistribution(rd), minDepth, maxDepth);
        seen.insert(ComputeContentHash(tree, zobrist));
    }

    auto uniqueRatio = static_cast<double>(seen.size()) / static_cast<double>(n);
    CHECK(uniqueRatio > 0.98);
}

TEST_CASE("ContentHash - empty tree returns 0 rather than underflowing", "[content_hash]")
{
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist const zobrist(rng, MaxLength, inputs);

    Tree const empty;
    CHECK(empty.Nodes().empty());
    CHECK(ComputeContentHash(empty, zobrist) == 0);
}

} // namespace Operon::Test
