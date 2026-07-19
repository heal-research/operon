// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <random>

#include "../operon_test.hpp"

#include "operon/core/dataset.hpp"
#include "operon/core/pset.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/initializer.hpp"

namespace Operon::Test {
namespace {

auto GenerateTrees(Operon::RandomGenerator& random, Operon::CreatorBase& creator, std::vector<size_t> lengths, size_t maxDepth) -> std::vector<Tree>
{
    std::vector<Tree> trees;
    trees.reserve(lengths.size());
    UniformTreeInitializer treeInit(&creator);
    treeInit.ParameterizeDistribution(1UL, 100UL);
    treeInit.SetMaxDepth(maxDepth);
    UniformCoefficientInitializer coeffInit;
    coeffInit.ParameterizeDistribution(Operon::Scalar{-1}, Operon::Scalar{+1});

    std::transform(lengths.begin(), lengths.end(), std::back_inserter(trees), [&](size_t /*not used*/) -> Operon::Tree {
        auto tree = treeInit(random);
        coeffInit(random, tree);
        return tree;
    });

    return trees;
}
} // namespace

TEST_CASE("Grammar sampling", "[operators]")
{
    PrimitiveSet grammar;
    grammar.SetConfig(~PrimitiveSetConfig{});
    Operon::RandomGenerator rd(1234);

    // Bucket by primitive hash rather than NodeType: post-collapse, every
    // built-in math op shares NodeType::Function, so a single "Function"
    // bucket would conflate all 29 ops. Enumerate the actual registered
    // primitives instead - every BuiltinOp plus the 3 non-Function
    // terminals (Constant/Variable/Ref; Function itself has no single real
    // hash and is never registered by SetConfig, see pset.cpp).
    std::vector<Operon::Hash> hashes;
    for (size_t i = 0; i < BuiltinOpCount; ++i) {
        hashes.push_back(static_cast<Operon::Hash>(static_cast<BuiltinOp>(i)));
    }
    for (auto type : { NodeType::Constant, NodeType::Variable, NodeType::Ref }) {
        hashes.push_back(Node(type).HashValue);
    }

    Operon::Map<Operon::Hash, double> observed;
    for (auto h : hashes) { observed[h] = 0; }
    size_t const r = grammar.EnabledPrimitives().size() + 1;

    const size_t nTrials = 1'000'000;
    for (auto i = 0U; i < nTrials; ++i) {
        auto node = grammar.SampleRandomSymbol(rd, 0, 2);
        ++observed[node.HashValue];
    }
    for (auto& [h, v] : observed) { v /= static_cast<double>(nTrials); }

    Operon::Map<Operon::Hash, double> actual;
    for (auto h : hashes) { actual[h] = static_cast<double>(grammar.Frequency(h)); }
    auto freqSum = 0.0;
    for (auto const& [h, v] : actual) { freqSum += v; }
    for (auto& [h, v] : actual) { v /= freqSum; }

    auto chi = 0.0;
    for (auto h : hashes) {
        if (!grammar.IsEnabled(h)) {
            continue;
        }
        auto x = observed[h];
        auto y = actual[h];
        chi += (x - y) * (x - y) / y;
    }
    chi *= nTrials;

    auto criticalValue = static_cast<double>(r) + (2 * std::sqrt(r));
    REQUIRE(chi <= criticalValue);
}

TEST_CASE("GROW creator", "[operators]") // NOLINT(readability-function-cognitive-complexity)
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable("Y").value().Hash);
    size_t const maxDepth = 10;
    size_t const maxLength = 100;
    size_t const n = 1000;

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic | BuiltinOp::Log | BuiltinOp::Exp);
    grammar.SetMaximumArity(Util::MakeOp<BuiltinOp::Add>(), 2);
    grammar.SetMaximumArity(Util::MakeOp<BuiltinOp::Mul>(), 2);
    grammar.SetMaximumArity(Util::MakeOp<BuiltinOp::Sub>(), 2);
    grammar.SetMaximumArity(Util::MakeOp<BuiltinOp::Div>(), 2);

    GrowTreeCreator gtc{&grammar, inputs, maxLength};
    Operon::RandomGenerator random(1234);
    auto sizeDistribution = std::uniform_int_distribution<size_t>(1, maxLength);

    SECTION("Trees are within size bounds") {
        std::vector<size_t> lengths(n);
        std::generate(lengths.begin(), lengths.end(), [&]() -> size_t { return sizeDistribution(random); });
        auto trees = GenerateTrees(random, gtc, lengths, maxDepth);
        for (auto const& tree : trees) {
            CHECK(tree.Length() > 0);
            CHECK(tree.Length() <= maxLength + 10); // allow some slack for tree construction
        }
    }

    SECTION("Only enabled primitives appear") {
        auto tree = gtc(random, 20, 1, maxDepth);
        for (auto const& node : tree.Nodes()) {
            if (!node.IsLeaf()) {
                CHECK(grammar.IsEnabled(node.HashValue));
            }
        }
    }
}

TEST_CASE("BTC creator", "[operators]")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable("Y").value().Hash);
    size_t const maxDepth = 1000;
    size_t const maxLength = 100;
    size_t const n = 1000;

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic | BuiltinOp::Log | BuiltinOp::Exp);
    grammar.SetMaximumArity(Util::MakeOp<BuiltinOp::Add>(), 2);
    grammar.SetMaximumArity(Util::MakeOp<BuiltinOp::Mul>(), 2);
    grammar.SetMaximumArity(Util::MakeOp<BuiltinOp::Sub>(), 2);
    grammar.SetMaximumArity(Util::MakeOp<BuiltinOp::Div>(), 2);

    BalancedTreeCreator btc{&grammar, inputs, /* bias= */ 0.0, maxLength};
    Operon::RandomGenerator random(1234);
    auto sizeDistribution = std::uniform_int_distribution<size_t>(1, maxLength);

    SECTION("Trees are within size bounds") {
        std::vector<size_t> lengths(n);
        std::generate(lengths.begin(), lengths.end(), [&]() -> size_t { return sizeDistribution(random); });
        auto trees = GenerateTrees(random, btc, lengths, maxDepth);
        for (auto const& tree : trees) {
            CHECK(tree.Length() > 0);
        }
    }

    SECTION("Coefficients are initialized") {
        auto tree = btc(random, 20, 1, maxDepth);
        UniformCoefficientInitializer const coeffInit;
        coeffInit.ParameterizeDistribution(Operon::Scalar{-1}, Operon::Scalar{+1});
        coeffInit(random, tree);

        bool hasCoeff = false;
        for (auto const& node : tree.Nodes()) {
            if (node.IsLeaf() && node.IsConstant()) {
                hasCoeff = true;
            }
        }
        CHECK(hasCoeff);
    }
}

TEST_CASE("PTC2 creator", "[operators]")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable("Y").value().Hash);
    size_t const maxDepth = 1000;
    size_t const maxLength = 100;
    size_t const n = 1000;

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic | BuiltinOp::Log | BuiltinOp::Exp);

    ProbabilisticTreeCreator ptc{&grammar, inputs, /* bias= */ 0.0, maxLength};
    Operon::RandomGenerator random(1234);
    auto sizeDistribution = std::uniform_int_distribution<size_t>(1, maxLength);

    SECTION("Trees are within size bounds") {
        std::vector<size_t> lengths(n);
        std::generate(lengths.begin(), lengths.end(), [&]() -> size_t { return sizeDistribution(random); });
        auto trees = GenerateTrees(random, ptc, lengths, maxDepth);
        for (auto const& tree : trees) {
            CHECK(tree.Length() > 0);
        }
    }
}

TEST_CASE("AchievableLength snap-down table", "[operators]") // NOLINT(readability-function-cognitive-complexity)
{
    // A thin subclass that exposes the protected AchievableLength method.
    struct TestCreator final : public CreatorBase {
        TestCreator(PrimitiveSet const* pset, size_t maxLen)
            : CreatorBase(pset, {}, maxLen) {}
        auto operator()(RandomGenerator& /*rng*/, size_t /*targetLen*/, size_t /*minDepth*/, size_t /*maxDepth*/) const -> Tree override {
            return Tree({ Node(NodeType::Constant) }).UpdateNodes();
        }
        [[nodiscard]] auto SnapDown(size_t n) const -> size_t { return AchievableLength(n); }
    };

    SECTION("Edge cases") {
        PrimitiveSet pset;
        pset.SetConfig(PrimitiveSet::Arithmetic);
        TestCreator const tc(&pset, 20);
        CHECK(tc.SnapDown(0) == 1);
        CHECK(tc.SnapDown(1) == 1);
    }

    SECTION("Binary-only pset: achievable lengths are 1, 3, 5, 7, ...") {
        // Only arity-2 functions: n > 1 is achievable iff (n-1) is a multiple of 2,
        // i.e. n is odd. So snap_[i] carries the last odd value seen.
        PrimitiveSet pset;
        pset.SetConfig(BuiltinOp::Add | NodeType::Variable);
        pset.SetMinMaxArity(Util::MakeOp<BuiltinOp::Add>(), 2, 2);
        TestCreator const tc(&pset, 20);

        CHECK(tc.SnapDown(1) == 1);
        CHECK(tc.SnapDown(2) == 1); // 2 not achievable → snap down to 1
        CHECK(tc.SnapDown(3) == 3);
        CHECK(tc.SnapDown(4) == 3); // 4 not achievable → snap down to 3
        CHECK(tc.SnapDown(5) == 5);
        CHECK(tc.SnapDown(6) == 5);
        CHECK(tc.SnapDown(7) == 7);
    }

    SECTION("Mixed pset (arities 1 and 2): every length is achievable") {
        // Arity-1 means we can always add exactly 1 node, so all lengths >= 1 are reachable.
        PrimitiveSet pset;
        pset.SetConfig(BuiltinOp::Sin | BuiltinOp::Add | NodeType::Variable);
        pset.SetMinMaxArity(Util::MakeOp<BuiltinOp::Sin>(), 1, 1);
        pset.SetMinMaxArity(Util::MakeOp<BuiltinOp::Add>(), 2, 2);
        TestCreator const tc(&pset, 15);

        for (size_t n = 1; n <= 15; ++n) {
            CHECK(tc.SnapDown(n) == n);
        }
    }

    SECTION("Ternary-only pset: achievable lengths are 1, 4, 7, 10, ...") {
        // Only arity-3 functions: n > 1 is achievable iff (n-1) is a multiple of 3.
        // Achievable: 1, 4, 7, 10, 13, ...
        PrimitiveSet pset;
        pset.SetConfig(BuiltinOp::Add | NodeType::Variable);
        pset.SetMinMaxArity(Util::MakeOp<BuiltinOp::Add>(), 3, 3);
        TestCreator const tc(&pset, 20);

        CHECK(tc.SnapDown(1) == 1);
        CHECK(tc.SnapDown(2) == 1);
        CHECK(tc.SnapDown(3) == 1);
        CHECK(tc.SnapDown(4) == 4);
        CHECK(tc.SnapDown(5) == 4);
        CHECK(tc.SnapDown(6) == 4);
        CHECK(tc.SnapDown(7) == 7);
        CHECK(tc.SnapDown(8) == 7);
        CHECK(tc.SnapDown(9) == 7);
        CHECK(tc.SnapDown(10) == 10);
    }
}

TEST_CASE("Creator length contract with unachievable targets", "[operators]") // NOLINT(readability-function-cognitive-complexity)
{
    // Binary-only pset: achievable lengths are 1, 3, 5, 7, ...
    // Requesting any even target must snap down — the returned tree must be strictly
    // within the requested bound.
    PrimitiveSet pset;
    pset.SetConfig(BuiltinOp::Add | NodeType::Variable);
    pset.SetMinMaxArity(Util::MakeOp<BuiltinOp::Add>(), 2, 2);

    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable("Y").value().Hash);
    Operon::RandomGenerator rng(42);
    constexpr size_t maxDepth = 1000;
    constexpr size_t maxLength = 20;

    SECTION("BTC never exceeds requested length") {
        BalancedTreeCreator const btc(&pset, inputs, /* bias= */ 0.0, maxLength);
        for (size_t target = 1; target <= maxLength; ++target) {
            auto tree = btc(rng, target, 1, maxDepth);
            CHECK(tree.Length() > 0);
            CHECK(tree.Length() <= target);
        }
    }

    SECTION("PTC2 never exceeds requested length") {
        ProbabilisticTreeCreator const ptc(&pset, inputs, /* bias= */ 0.0, maxLength);
        for (size_t target = 1; target <= maxLength; ++target) {
            auto tree = ptc(rng, target, 1, maxDepth);
            CHECK(tree.Length() > 0);
            CHECK(tree.Length() <= target);
        }
    }
}

} // namespace Operon::Test
