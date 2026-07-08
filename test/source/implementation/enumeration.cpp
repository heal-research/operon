// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include "operon/algorithms/enumeration.hpp"
#include "operon/core/grammar.hpp"
#include "operon/core/pset.hpp"

namespace Operon::Test {

TEST_CASE("Complexity - counts all non-Constant nodes", "[enumeration]")
{
    // (x + y) * 2 : postfix [x, y, Add, Constant(2), Mul] - complexity should
    // count Variable x, Variable y, Add, Mul (4), excluding the Constant.
    Node nx(NodeType::Variable); nx.HashValue = 1;
    Node ny(NodeType::Variable); ny.HashValue = 2;
    Tree const tree = Tree({ nx, ny, Node(NodeType::Add), Node::Constant(2.0), Node(NodeType::Mul) }).UpdateNodes();
    CHECK(Complexity(tree) == 4);
}

TEST_CASE("Complexity - single variable has complexity 1", "[enumeration]")
{
    Node nx(NodeType::Variable); nx.HashValue = 1;
    Tree const tree = Tree({ nx }).UpdateNodes();
    CHECK(Complexity(tree) == 1);
}

TEST_CASE("Complexity - single constant has complexity 0", "[enumeration]")
{
    Tree const tree = Tree({ Node::Constant(3.0) }).UpdateNodes();
    CHECK(Complexity(tree) == 0);
}

TEST_CASE("EnumerationEngine - seeds RecurringFactor/SimpleTerm/Term at budget 1", "[enumeration]")
{
    std::vector<Operon::Hash> const vars{ 10, 20, 30 };
    Grammar grammar(PrimitiveSet::Arithmetic, vars);
    Operon::RandomGenerator rng(42);
    EnumerationEngine engine(grammar, /*maxComplexity=*/5, rng);

    engine.Build();

    CHECK(engine.Bucket(GrammarSymbol::RecurringFactor, 1).size() == vars.size());
    CHECK(engine.Bucket(GrammarSymbol::SimpleTerm, 1).size() == vars.size());
    // Term[1] = RecurringFactor[1] via coercion - same count, no new nodes.
    CHECK(engine.Bucket(GrammarSymbol::Term, 1).size() == vars.size());
    for (auto const& t : engine.Bucket(GrammarSymbol::Term, 1)) {
        CHECK(Complexity(t) == 1);
    }
}

TEST_CASE("EnumerationEngine - Term products dedup commutative reorderings", "[enumeration]")
{
    // With 2 variables x, y: a 2-factor product [x, y, Mul] has complexity 3
    // (Variable + Variable + Mul all count) - Term[3] should contain exactly
    // the 3 distinct products {x*x, x*y, y*y}; x*y reached via both (x,y) and
    // (y,x) operand orderings must collapse to one entry, not two.
    std::vector<Operon::Hash> const vars{ 10, 20 };
    Grammar grammar(PrimitiveSet::Arithmetic, vars);
    Operon::RandomGenerator rng(42);
    EnumerationEngine engine(grammar, /*maxComplexity=*/5, rng);

    engine.Build();

    auto term3 = engine.Bucket(GrammarSymbol::Term, 3);
    CHECK(term3.size() == 3);
    for (auto const& t : term3) {
        CHECK(Complexity(t) == 3);
    }
}

TEST_CASE("EnumerationEngine - Expression's cheapest shape has complexity 3", "[enumeration]")
{
    // const*x + const, for each variable - Expression[1] and Expression[2]
    // must stay empty (MinComplexity(Expression) == 3).
    std::vector<Operon::Hash> const vars{ 10, 20 };
    Grammar grammar(PrimitiveSet::Arithmetic, vars);
    Operon::RandomGenerator rng(42);
    EnumerationEngine engine(grammar, /*maxComplexity=*/5, rng);

    engine.Build();

    CHECK(engine.Bucket(GrammarSymbol::Expression, 1).empty());
    CHECK(engine.Bucket(GrammarSymbol::Expression, 2).empty());
    CHECK(engine.Bucket(GrammarSymbol::Expression, 3).size() == vars.size());
    for (auto const& t : engine.Bucket(GrammarSymbol::Expression, 3)) {
        CHECK(Complexity(t) == 3);
    }
}

TEST_CASE("EnumerationEngine - MaxComplexity bound is respected", "[enumeration]")
{
    std::vector<Operon::Hash> const vars{ 10, 20, 30 };
    Grammar grammar(PrimitiveSet::Full, vars);
    Operon::RandomGenerator rng(7);
    constexpr std::size_t maxComplexity = 6;
    EnumerationEngine engine(grammar, maxComplexity, rng);

    engine.Build();

    for (auto nt : { GrammarSymbol::Expression, GrammarSymbol::Term, GrammarSymbol::RecurringFactor,
                     GrammarSymbol::SimpleExpr, GrammarSymbol::SimpleTerm }) {
        for (auto const& t : engine.Bucket(nt, maxComplexity)) {
            CHECK(Complexity(t) <= maxComplexity);
        }
    }
}

TEST_CASE("EnumerationEngine - unary wraps populate RecurringFactor beyond budget 1", "[enumeration]")
{
    // Log(const*x + const) has complexity 1(Log) + 3(SimpleExpr) = 4.
    std::vector<Operon::Hash> const vars{ 10 };
    Grammar grammar(PrimitiveSet::TypeCoherent, vars);
    Operon::RandomGenerator rng(42);
    EnumerationEngine engine(grammar, /*maxComplexity=*/6, rng);

    engine.Build();

    CHECK_FALSE(engine.Bucket(GrammarSymbol::RecurringFactor, 4).empty());
    for (auto const& t : engine.Bucket(GrammarSymbol::RecurringFactor, 4)) {
        CHECK(Complexity(t) == 4);
    }
}

} // namespace Operon::Test
