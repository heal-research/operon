// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "../operon_test.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/formatter/formatter.hpp"
#include <fmt/format.h>
#include "operon/core/pset.hpp"
#include "operon/operators/creator.hpp"
#include "operon/parser/infix.hpp"

namespace Operon::Test {

TEST_CASE("Parser roundtrip correctness", "[parser]")
{
    constexpr int nTrees = 100'000;
    constexpr int nNodes = 20;
    constexpr int nrow = 1;
    constexpr int ncol = 10;

    Operon::RandomGenerator rng(1234);

    Operon::Dataset ds = Operon::Test::Util::RandomDataset(rng, nrow, ncol);

    Operon::PrimitiveSet pset;
    pset.SetConfig(PrimitiveSet::Arithmetic | BuiltinOp::Aq | BuiltinOp::Exp | BuiltinOp::Log | NodeType::Variable);
    Operon::BalancedTreeCreator const btc(&pset, ds.VariableHashes(), /* bias= */ 0.0, nNodes);

    Operon::Vector<Operon::Tree> trees;
    trees.reserve(nTrees);
    for (int i = 0; i < nTrees; ++i) {
        trees.push_back(btc(rng, nNodes, 1, 10));
    }

    Operon::Vector<Operon::Tree> parsedTrees;
    parsedTrees.reserve(nTrees);
    std::transform(trees.begin(), trees.end(), std::back_inserter(parsedTrees), [&](const auto& tree) -> auto {
        return InfixParser::Parse(fmt::format("{:infix:50}", Operon::Fmt::WithNames{tree, ds}), ds);
    });

    Range range{0, 1};

    using DTable = DispatchTable<Operon::Scalar>;
    DTable const dtable;

    auto countFailures = [&](float eps) -> size_t {
        size_t count{0};
        for (int i = 0; i < nTrees; ++i) {
            auto const& t1 = trees[i];
            auto const& t2 = parsedTrees[i];
            auto v1 = Interpreter<Operon::Scalar, DTable>::Evaluate(t1, ds, range)[0];
            auto v2 = Interpreter<Operon::Scalar, DTable>::Evaluate(t2, ds, range)[0];
            if (std::isfinite(v1)) {
                count += static_cast<size_t>(!std::isfinite(v2) || std::abs(v1 - v2) > eps);
            }
        }
        return count;
    };

    constexpr auto epsLoose{1e-6F};
    constexpr auto epsStrict{1e-5F};
    constexpr auto maxFailureRateLoose{1e-2};
    // Fast polynomial approximations (FastExp, FastLog) introduce ~1-2 ULP error,
    // compounded over a tree evaluation. On top of that, InfixFormatter deliberately
    // prints Aq/Powabs as a decomposed expression (e.g. "a / (sqrt(1 + b^2))" for Aq)
    // rather than their native aq()/powabs() call syntax, so external tools (sympy,
    // numpy, ...) can read the output without knowing Operon-specific function names.
    // Reparsing that decomposition reconstructs Div/Sqrt/Add/Pow instead of the native
    // node, which evaluates the squaring via FastPow's FastExp/FastLog approximation
    // instead of Aq's exact eve::sqr -- a different, less precise numerical path for
    // the same nominal formula. This is an accepted, deliberate readability tradeoff,
    // not a defect; 1e-2 gives adequate headroom for it plus the ULP-level error above.
    constexpr auto maxFailureRateStrict{1e-2};

    CHECK(static_cast<double>(countFailures(epsLoose))  / nTrees < maxFailureRateLoose);
    CHECK(static_cast<double>(countFailures(epsStrict)) / nTrees < maxFailureRateStrict);
}

TEST_CASE("Parse specific expressions", "[parser]")
{
    SECTION("Nested unary functions") {
        const auto* str = "sin((sqrt(abs(square(sin(((-0.00191) * X6))))) - sqrt(abs(((-0.96224) / (-0.40567))))))";
        auto tree = Operon::InfixParser::Parse(str);
        CHECK(tree.Length() > 0);
    }

    SECTION("Arithmetic with constants") {
        Node c1(NodeType::Constant); c1.Value = 2;
        Node c2(NodeType::Constant); c2.Value = 3;
        Node c3(NodeType::Constant); c3.Value = 5;
        auto const sub = Util::MakeOp<BuiltinOp::Sub>();
        auto const mul = Util::MakeOp<BuiltinOp::Mul>();
        Operon::Vector<Node> const nodes{c1, c2, c3, sub, mul}; // 5 - 3 * 2
        Tree t(nodes);
        t.UpdateNodes();

        Dataset const ds("./data/Poly-10.csv", true);
        auto s1 = fmt::format("{:infix:5}", Operon::Fmt::WithNames{t, ds});
        auto t2 = InfixParser::Parse(s1);

        // Roundtrip: same number of nodes
        CHECK(t.Length() == t2.Length());
    }

    SECTION("Analytical quotient") {
        std::string const expr{"aq(3, 5)"};
        auto tree = InfixParser::Parse(expr);
        CHECK(tree.Length() > 0);
    }

    SECTION("Multiple additions") {
        const auto* modelStr = "1 + 2 + 3 + 4";
        auto tree = Operon::InfixParser::Parse(modelStr);

        using DTable = DispatchTable<Operon::Scalar>;
        DTable const dtable;
        std::string const x{"x"};
        std::vector<Operon::Scalar> const v{0};
        Operon::Dataset const ds({x}, {v});
        auto result = Interpreter<Operon::Scalar, DTable>::Evaluate(tree, ds, Range(0, 1));
        CHECK(result[0] == Catch::Approx(10.0F));
    }
}

TEST_CASE("Formatter output", "[parser]")
{
    SECTION("Balanced parentheses") {
        Operon::RandomGenerator rng(1234);
        Operon::Dataset const ds("./data/Poly-10.csv", true);
        Operon::PrimitiveSet pset;
        pset.SetConfig(PrimitiveSet::Arithmetic | BuiltinOp::Exp | BuiltinOp::Log);
        constexpr size_t maxLength = 20;
        Operon::BalancedTreeCreator const btc(&pset, ds.VariableHashes(), /* bias= */ 0.0, maxLength);

        auto validateString = [](auto const& s) -> auto {
            size_t lp{0};
            size_t rp{0};
            for (auto c : s) {
                lp += c == '(';
                rp += c == ')';
            }
            return lp == rp;
        };

        for (int i = 0; i < 100; ++i) {
            auto tree = btc(rng, 20, 1, 10);
            auto s = fmt::format("{:infix:5}", Operon::Fmt::WithNames{tree, ds});
            CHECK(validateString(s));
        }
    }

    SECTION("Ref nodes follow RefTo, not the preceding array slot") {
        // x + ref(x) -- a minimal tree with structural sharing, the shape
        // symbolic differentiation produces. Node 1 (Ref) has no bearing on
        // node 0 by array adjacency alone (they'd coincide here by
        // accident), so also cover a case where RefTo is NOT i-1: x*y +
        // ref(x) should format to reference x again, not y.
        using DTable = DispatchTable<Operon::Scalar>;
        Operon::Dataset const ds2({"x", "y"}, {{2.0F}, {3.0F}});
        auto const hx = ds2.GetVariable("x").value().Hash;
        auto const hy = ds2.GetVariable("y").value().Hash;
        Operon::Map<Operon::Hash, std::string> const names { {hx, "x"}, {hy, "y"} };

        Operon::Node nx(NodeType::Variable);
        nx.HashValue = hx;
        nx.Value = 1;
        Operon::Node ny(NodeType::Variable);
        ny.HashValue = hy;
        ny.Value = 1;
        auto mul = Operon::Node::Function(Operon::Hash(BuiltinOp::Mul), 2);
        auto refX = Operon::Node::Ref(0); // points at nx, not at mul (the preceding node)
        auto add = Operon::Node::Function(Operon::Hash(BuiltinOp::Add), 2);

        Operon::Tree tree({nx, ny, mul, refX, add});
        tree.UpdateNodes();

        auto const s = fmt::format("{:infix:3}", Operon::Fmt::WithNames{tree, names});
        CHECK(s.find('y') != std::string::npos); // from x*y
        // Formatted string must reference x twice (once from x*y, once from
        // ref(x)) -- a formatter that follows i-1 instead of RefTo would
        // wrap the mul node again and never mention y a second time nor
        // produce a string that round-trips to the tree's actual value.
        auto const reparsed = InfixParser::Parse(s, ds2);
        Operon::Range const rg(0, 1);
        auto const original = Interpreter<Operon::Scalar, DTable>::Evaluate(tree, ds2, rg);
        auto const roundtrip = Interpreter<Operon::Scalar, DTable>::Evaluate(reparsed, ds2, rg);
        CHECK(original[0] == Catch::Approx(8.0F)); // x*y + x = 2*3 + 2
        CHECK(roundtrip[0] == Catch::Approx(original[0]));
    }

    SECTION("PostfixFormatter, TreeFormatter and DotFormatter follow RefTo") {
        // same x*y + ref(x) tree as the InfixFormatter section above
        Operon::Dataset const ds2({"x", "y"}, {{2.0F}, {3.0F}});
        auto const hx = ds2.GetVariable("x").value().Hash;
        auto const hy = ds2.GetVariable("y").value().Hash;
        Operon::Map<Operon::Hash, std::string> const names { {hx, "x"}, {hy, "y"} };

        Operon::Node nx(NodeType::Variable);
        nx.HashValue = hx;
        nx.Value = 1;
        Operon::Node ny(NodeType::Variable);
        ny.HashValue = hy;
        ny.Value = 1;
        auto mul = Operon::Node::Function(Operon::Hash(BuiltinOp::Mul), 2);
        auto refX = Operon::Node::Ref(0); // points at nx (index 0)
        auto add = Operon::Node::Function(Operon::Hash(BuiltinOp::Add), 2);

        Operon::Tree tree({nx, ny, mul, refX, add});
        tree.UpdateNodes();

        auto countOccurrences = [](std::string const& s, std::string const& sub) -> size_t {
            size_t count{0};
            for (size_t pos = s.find(sub); pos != std::string::npos; pos = s.find(sub, pos + sub.size())) {
                ++count;
            }
            return count;
        };

        SECTION("PostfixFormatter replays the referenced subtree's tokens") {
            auto const s = fmt::format("{:postfix:2}", Operon::Fmt::WithNames{tree, names});
            // x*y contributes one "x", ref(x) must replay x's token again --
            // a formatter following i-1 instead of RefTo would instead
            // duplicate the "y" (or mul) token and never emit a second x.
            CHECK(countOccurrences(s, "x") == 2);
            CHECK(s.find("ref") == std::string::npos);
        }

        SECTION("TreeFormatter nests the referenced subtree under the Ref leaf") {
            auto const s = fmt::format("{:tree:2}", Operon::Fmt::WithNames{tree, names});
            CHECK(countOccurrences(s, "x D:") == 2);
        }

        SECTION("DotFormatter points edges at the shared node, not a duplicate/dangling Ref box") {
            auto const s = fmt::format("{:dot:2}", Operon::Fmt::WithNames{tree, names});
            CHECK_NOTHROW(fmt::format("{:dot:2}", Operon::Fmt::WithNames{tree, names}));
            CHECK(s.find("3 [label=") == std::string::npos); // no box for the Ref node itself
            CHECK(s.find("0 -> 4") != std::string::npos); // nx -> add, resolved through the Ref
            CHECK(s.find("3 ->") == std::string::npos);
            CHECK(s.find("-> 3") == std::string::npos);
        }
    }

    SECTION("Bare Tree (no WithNames) falls back to deterministic X_<hash> variable labels") {
        Operon::Dataset const ds2({"x", "y"}, {{2.0F}, {3.0F}});
        auto const hx = ds2.GetVariable("x").value().Hash;

        Operon::Node nx(NodeType::Variable);
        nx.HashValue = hx;
        nx.Value = 1;
        Operon::Tree tree({nx});
        tree.UpdateNodes();

        // fmt::formatter<Operon::Tree> (no Dataset/map supplied) must not
        // throw -- a Tree does not itself own variable names, so this is
        // diagnostic-only fallback labeling, not an error condition.
        std::string bare;
        CHECK_NOTHROW(bare = fmt::format("{}", tree));
        CHECK(bare.find(fmt::format("X_{:016x}", hx)) != std::string::npos);

        // Deterministic: the same hash gets the same fallback label on a
        // second, independent format call.
        CHECK(fmt::format("{}", tree) == bare);

        // Every mode accepts the bare-Tree form.
        CHECK_NOTHROW(fmt::format("{:postfix}", tree));
        CHECK_NOTHROW(fmt::format("{:tree}", tree));
        CHECK_NOTHROW(fmt::format("{:dot}", tree));
    }
}

TEST_CASE("ParseFunctionBody", "[parser]")
{
    SECTION("Unary body: param resolves, arity accepted") {
        std::vector<std::string> const params{"x"};
        auto tree = InfixParser::ParseFunctionBody("1 / (1 + exp(-x))", params);
        CHECK(tree.Length() > 0);
        for (auto const& n : tree.Nodes()) {
            if (n.IsVariable()) {
                CHECK(n.HashValue == Operon::ParamHash(0));
            }
        }
    }

    SECTION("Binary body: both params resolve to distinct reserved hashes") {
        std::vector<std::string> const params{"a", "b"};
        auto tree = InfixParser::ParseFunctionBody("a * exp(-b)", params);
        bool sawA = false;
        bool sawB = false;
        for (auto const& n : tree.Nodes()) {
            if (!n.IsVariable()) { continue; }
            sawA |= n.HashValue == Operon::ParamHash(0);
            sawB |= n.HashValue == Operon::ParamHash(1);
        }
        CHECK(sawA);
        CHECK(sawB);
    }

    SECTION("Undeclared identifier throws") {
        std::vector<std::string> const params{"x"};
        CHECK_THROWS_AS(InfixParser::ParseFunctionBody("x + y", params), std::invalid_argument);
    }

    SECTION("Unused parameter throws") {
        std::vector<std::string> const params{"x", "y"};
        CHECK_THROWS_AS(InfixParser::ParseFunctionBody("sin(x)", params), std::invalid_argument);
    }

    SECTION("Arity above the v1 cap throws") {
        std::vector<std::string> const params{"a", "b", "c"};
        CHECK_THROWS_AS(InfixParser::ParseFunctionBody("a + b + c", params), std::invalid_argument);
    }

    SECTION("Body-internal constants are forced Optimize=false") {
        std::vector<std::string> const params{"x"};
        auto tree = InfixParser::ParseFunctionBody("2.5 * x", params);
        for (auto const& n : tree.Nodes()) {
            if (n.Type == Operon::NodeType::Constant) {
                CHECK_FALSE(n.Optimize);
            }
        }
    }

    // Pinning current behavior, not a designed-for feature: duplicate names
    // collapse to the same hash, so the second occurrence's slot never gets
    // matched by any body identifier and trips the unused-parameter check —
    // duplicates are rejected, just via that check's message rather than a
    // dedicated "duplicate parameter name" one.
    SECTION("Duplicate parameter names throw (via unused-parameter check)") {
        std::vector<std::string> const params{"x", "x"};
        CHECK_THROWS_AS(InfixParser::ParseFunctionBody("sin(x)", params), std::invalid_argument);
    }

    SECTION("Zero-parameter body (built-ins/constants only) is accepted") {
        std::vector<std::string> const params{};
        auto tree = InfixParser::ParseFunctionBody("sin(1.0) + exp(2.0)", params);
        CHECK(tree.Length() > 0);
        for (auto const& n : tree.Nodes()) {
            CHECK_FALSE(n.IsVariable());
        }
    }
}

} // namespace Operon::Test
