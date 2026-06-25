// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>
#include <fmt/core.h>

#include "operon/core/dataset.hpp"
#include "operon/core/node.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/variable.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/operators/mutation.hpp"

namespace Operon::Test {

// Helper: build a tree from a node vector and update metadata.
auto MakeTree(std::initializer_list<Node> nodes) -> Tree
{
    return Tree(nodes).UpdateNodes();
}

TEST_CASE("HashCons deduplicates identical subtrees", "[dag]")
{
    // Build: add(mul(x, y), mul(x, y))
    // Postfix: x y mul x y mul add
    auto x1 = Node(NodeType::Variable);
    x1.HashValue = x1.CalculatedHashValue = 1001;
    x1.Value = 1.0;
    auto y1 = Node(NodeType::Variable);
    y1.HashValue = y1.CalculatedHashValue = 1002;
    y1.Value = 1.0;
    auto mul1 = Node(NodeType::Mul);
    mul1.Arity = 2;
    mul1.Value = 1.0;

    auto x2 = x1; // same variable
    auto y2 = y1;
    auto mul2 = Node(NodeType::Mul);
    mul2.Arity = 2;
    mul2.Value = 1.0;

    auto add = Node(NodeType::Add);
    add.Arity = 2;
    add.Value = 1.0;

    auto tree = MakeTree({x1, y1, mul1, x2, y2, mul2, add});
    REQUIRE(tree.Length() == 7);

    tree.HashCons();

    // After hash-consing: x y mul ref(->mul) add = 5 nodes
    CHECK(tree.Length() == 5);

    auto const& nodes = tree.Nodes();
    // The 4th node (index 3) should be a Ref pointing to mul at index 2
    CHECK(nodes[3].IsRef());
    CHECK(nodes[3].RefTo == 2);

    // Root is still add
    CHECK(nodes[4].Is<NodeType::Add>());
}

TEST_CASE("HashCons is idempotent", "[dag]")
{
    auto x1 = Node(NodeType::Variable);
    x1.HashValue = x1.CalculatedHashValue = 1001;
    x1.Value = 1.0;
    auto y1 = Node(NodeType::Variable);
    y1.HashValue = y1.CalculatedHashValue = 1002;
    y1.Value = 1.0;
    auto mul1 = Node(NodeType::Mul);
    mul1.Arity = 2;
    mul1.Value = 1.0;

    auto x2 = x1;
    auto y2 = y1;
    auto mul2 = Node(NodeType::Mul);
    mul2.Arity = 2;
    mul2.Value = 1.0;

    auto add = Node(NodeType::Add);
    add.Arity = 2;
    add.Value = 1.0;

    auto tree = MakeTree({x1, y1, mul1, x2, y2, mul2, add});
    tree.HashCons();
    auto len1 = tree.Length();

    tree.HashCons();
    auto len2 = tree.Length();

    CHECK(len1 == len2);
}

TEST_CASE("HashCons does not deduplicate subtrees with different coefficients", "[dag]")
{
    auto x1 = Node(NodeType::Variable);
    x1.HashValue = x1.CalculatedHashValue = 1001;
    x1.Value = 1.0;
    auto c1 = Node(NodeType::Constant);
    c1.Value = 2.0;

    auto mul1 = Node(NodeType::Mul);
    mul1.Arity = 2;
    mul1.Value = 1.0;

    auto x2 = x1;
    auto c2 = Node(NodeType::Constant);
    c2.Value = 3.0; // different coefficient

    auto mul2 = Node(NodeType::Mul);
    mul2.Arity = 2;
    mul2.Value = 1.0;

    auto add = Node(NodeType::Add);
    add.Arity = 2;
    add.Value = 1.0;

    auto tree = MakeTree({x1, c1, mul1, x2, c2, mul2, add});
    REQUIRE(tree.Length() == 7);

    tree.HashCons();

    // Different coefficients → strict hash differs → no deduplication
    CHECK(tree.Length() == 7);
}

TEST_CASE("HashCons leaves single-node trees unchanged", "[dag]")
{
    auto x = Node(NodeType::Variable);
    x.HashValue = x.CalculatedHashValue = 1001;
    x.Value = 1.0;
    auto tree = MakeTree({x});
    tree.HashCons();
    CHECK(tree.Length() == 1);
}

TEST_CASE("RepairRefs fixes forward references", "[dag]")
{
    Operon::RandomGenerator rng(42);

    // Build a small tree with a broken Ref
    auto x = Node(NodeType::Variable);
    x.HashValue = x.CalculatedHashValue = 1001;
    x.Value = 1.0;

    auto bad = Node::Ref(5); // forward reference — invalid

    auto add = Node(NodeType::Add);
    add.Arity = 2;
    add.Value = 1.0;

    auto tree = MakeTree({x, bad, add});
    REQUIRE(tree.Length() == 3);

    tree.RepairRefs(rng);

    auto const& nodes = tree.Nodes();
    // The Ref at index 1 should now point to index 0 (the only valid target)
    CHECK(nodes[1].IsRef());
    CHECK(nodes[1].RefTo == 0);
}

TEST_CASE("RepairRefs demotes Ref when no valid target exists", "[dag]")
{
    Operon::RandomGenerator rng(42);

    // Tree with only Ref nodes before the broken one
    auto ref1 = Node::Ref(0);
    auto ref2 = Node::Ref(0);

    auto add = Node(NodeType::Add);
    add.Arity = 2;
    add.Value = 1.0;

    auto tree = MakeTree({ref1, ref2, add});
    tree.RepairRefs(rng);

    auto const& nodes = tree.Nodes();
    // At index 0: Ref with no backward targets → demoted to constant
    CHECK(nodes[0].Is<NodeType::Constant>());
    // At index 1: now has a valid target (index 0 is no longer a Ref)
    CHECK(nodes[1].IsRef());
    CHECK(nodes[1].RefTo == 0);
}

TEST_CASE("RepairRefs leaves valid Refs untouched", "[dag]")
{
    Operon::RandomGenerator rng(42);

    auto x = Node(NodeType::Variable);
    x.HashValue = x.CalculatedHashValue = 1001;
    x.Value = 1.0;
    auto ref = Node::Ref(0); // valid backward reference to x

    auto add = Node(NodeType::Add);
    add.Arity = 2;
    add.Value = 1.0;

    auto tree = MakeTree({x, ref, add});
    tree.RepairRefs(rng);

    auto const& nodes = tree.Nodes();
    CHECK(nodes[1].IsRef());
    CHECK(nodes[1].RefTo == 0);
}

TEST_CASE("RefTargetMutation retargets a Ref node", "[dag]")
{
    Operon::RandomGenerator rng(42);

    auto x = Node(NodeType::Variable);
    x.HashValue = x.CalculatedHashValue = 1001;
    x.Value = 1.0;
    auto y = Node(NodeType::Variable);
    y.HashValue = y.CalculatedHashValue = 1002;
    y.Value = 1.0;
    auto ref = Node::Ref(0); // points to x

    auto add = Node(NodeType::Add);
    add.Arity = 2;
    add.Value = 1.0;

    // tree: x y ref(->x) add — an add of y and ref(x)
    // wait, add has arity 2, children are y(1) and ref(2)... postfix order
    // Actually: x at 0, y at 1, ref at 2, add at 3
    // add's children: ref(2) and y(1)... no. In postfix, add at 3 with arity 2.
    // Children are at index 2 (direct child) and index 1.
    // So tree is: add(y, ref(x))
    auto tree = MakeTree({x, y, ref, add});
    REQUIRE(tree.Length() == 4);

    RefTargetMutation mut;
    bool changed = false;
    for (int i = 0; i < 100; ++i) {
        auto child = mut(rng, tree);
        auto const& nodes = child.Nodes();
        if (nodes[2].IsRef() && nodes[2].RefTo == 1) {
            changed = true;
            break;
        }
    }
    CHECK(changed);
}

TEST_CASE("RefTargetMutation is no-op on trees without Refs", "[dag]")
{
    Operon::RandomGenerator rng(42);

    auto x = Node(NodeType::Variable);
    x.HashValue = x.CalculatedHashValue = 1001;
    x.Value = 1.0;
    auto y = Node(NodeType::Variable);
    y.HashValue = y.CalculatedHashValue = 1002;
    y.Value = 1.0;
    auto add = Node(NodeType::Add);
    add.Arity = 2;
    add.Value = 1.0;

    auto tree = MakeTree({x, y, add});
    RefTargetMutation mut;
    auto child = mut(rng, tree);
    CHECK(child.Length() == tree.Length());
}

TEST_CASE("Crossover + RepairRefs produces valid DAG offspring", "[dag]")
{
    auto ds = Dataset("./data/Poly-10.csv", true);
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable("Y").value().Hash);

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic);

    BalancedTreeCreator btc{&grammar, inputs, 0.0, 50};
    UniformCoefficientInitializer cfi;

    Operon::RandomGenerator rng(1234);
    SubtreeCrossover cx{1.0, 50, 1000};

    for (int trial = 0; trial < 100; ++trial) {
        auto len = std::uniform_int_distribution<size_t>(5, 30)(rng);
        auto t1 = btc(rng, len, 1, 1000);
        cfi(rng, t1);
        t1.HashCons();

        auto t2 = btc(rng, len, 1, 1000);
        cfi(rng, t2);
        t2.HashCons();

        auto child = cx(rng, t1, t2);
        child.RepairRefs(rng);

        // validate all Refs
        auto const& nodes = child.Nodes();
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (!nodes[i].IsRef()) { continue; }
            CHECK(static_cast<size_t>(nodes[i].RefTo) < i);
            CHECK(!nodes[nodes[i].RefTo].IsRef());
        }
    }
}

TEST_CASE("HashCons handles nested duplicates", "[dag]")
{
    // add(add(x, y), add(x, y)) — the inner add(x,y) is duplicated
    auto x1 = Node(NodeType::Variable);
    x1.HashValue = x1.CalculatedHashValue = 1001;
    x1.Value = 1.0;
    auto y1 = Node(NodeType::Variable);
    y1.HashValue = y1.CalculatedHashValue = 1002;
    y1.Value = 1.0;
    auto add1 = Node(NodeType::Add);
    add1.Arity = 2;
    add1.Value = 1.0;

    auto x2 = x1;
    auto y2 = y1;
    auto add2 = Node(NodeType::Add);
    add2.Arity = 2;
    add2.Value = 1.0;

    auto root = Node(NodeType::Add);
    root.Arity = 2;
    root.Value = 1.0;

    auto tree = MakeTree({x1, y1, add1, x2, y2, add2, root});
    REQUIRE(tree.Length() == 7);

    tree.HashCons();

    // inner add(x,y) at [0,1,2] stays; [3,4,5] replaced by ref→2
    CHECK(tree.Length() == 5);

    auto const& nodes = tree.Nodes();
    CHECK(nodes[3].IsRef());
    CHECK(nodes[3].RefTo == 2);
    CHECK(nodes[4].Is<NodeType::Add>());
}

TEST_CASE("Creator produces Ref nodes when enabled", "[dag]")
{
    auto ds = Dataset("./data/Poly-10.csv", true);
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable("Y").value().Hash);

    PrimitiveSet pset;
    pset.SetConfig(PrimitiveSet::Arithmetic | NodeType::Sin | NodeType::Cos
                 | NodeType::Exp | NodeType::Log | NodeType::Sqrt
                 | NodeType::Tanh | NodeType::Ref);

    BalancedTreeCreator btc{&pset, inputs, 0.0, 50};
    UniformCoefficientInitializer cfi;

    Operon::RandomGenerator rng(42);

    int totalTrees = 1000;
    int treesWithRefs = 0;
    int totalRefs = 0;
    int totalNodes = 0;

    for (int i = 0; i < totalTrees; ++i) {
        auto len = std::uniform_int_distribution<size_t>(5, 50)(rng);
        auto tree = btc(rng, len, 1, 1000);
        cfi(rng, tree);

        int refs = 0;
        for (auto const& n : tree.Nodes()) {
            if (n.IsRef()) { ++refs; }
        }
        totalNodes += static_cast<int>(tree.Length());
        totalRefs += refs;
        if (refs > 0) { ++treesWithRefs; }
    }

    fmt::print("Trees with refs: {}/{} ({:.1f}%)\n", treesWithRefs, totalTrees,
               100.0 * treesWithRefs / totalTrees);
    fmt::print("Total refs: {}/{} nodes ({:.1f}%)\n", totalRefs, totalNodes,
               100.0 * totalRefs / totalNodes);
    fmt::print("Avg refs per tree (when present): {:.1f}\n",
               treesWithRefs > 0 ? static_cast<double>(totalRefs) / treesWithRefs : 0.0);

    CHECK(treesWithRefs > 0);
}

} // namespace Operon::Test
