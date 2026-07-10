// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include "operon/algorithms/enumeration.hpp"
#include "operon/core/grammar.hpp"
#include "operon/core/pset.hpp"

namespace Operon::Test {

namespace {
    // C(n+k-1, k) ("n multichoose k"), via the standard running-product form,
    // which stays exact at every step since each prefix is itself a binomial
    // coefficient. Used by the completeness tests below to check every
    // Term/SimpleTerm bucket against its closed-form multiset count, rather
    // than spot-checking a handful of small buckets.
    auto Multichoose(std::size_t n, std::size_t k) -> std::size_t
    {
        std::size_t const top = n + k - 1;
        std::size_t result = 1;
        for (std::size_t i = 0; i < k; ++i) { result = result * (top - i) / (i + 1); }
        return result;
    }
} // namespace

TEST_CASE("Complexity - counts all non-Constant nodes", "[enumeration]")
{
    // (x + y) * 2 : postfix [x, y, Add, Constant(2), Mul] - complexity should
    // count Variable x, Variable y, Add, Mul (4), excluding the Constant.
    Node nx(NodeType::Variable); nx.HashValue = 1;
    Node ny(NodeType::Variable); ny.HashValue = 2;
    Tree const tree = Tree({ nx, ny, Node(NodeType::Add), Node::Constant(2.0), Node(NodeType::Mul) }).UpdateNodes();
    CHECK(SymbolicComplexity(tree) == 4);
}

TEST_CASE("Complexity - single variable has complexity 1", "[enumeration]")
{
    Node nx(NodeType::Variable); nx.HashValue = 1;
    Tree const tree = Tree({ nx }).UpdateNodes();
    CHECK(SymbolicComplexity(tree) == 1);
}

TEST_CASE("Complexity - single constant has complexity 0", "[enumeration]")
{
    Tree const tree = Tree({ Node::Constant(3.0) }).UpdateNodes();
    CHECK(SymbolicComplexity(tree) == 0);
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
        CHECK(SymbolicComplexity(t) == 1);
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
        CHECK(SymbolicComplexity(t) == 3);
    }
}

TEST_CASE("EnumerationEngine - SimpleTerm bucket sizes match the multiset-count closed form", "[enumeration]")
{
    // SimpleTerm's only production is a commutative self-combine (SimpleTerm *
    // SimpleTerm, flattened by Tree::Reduce() into one flat n-ary Mul), so the
    // distinct SimpleTerm trees built from k variable factors are exactly the
    // multisets of size k drawn from the n variables - a completeness
    // invariant with a known closed form, C(n+k-1, k) ("n multichoose k").
    // A handful of spot-checked bucket sizes (as in the test above) wouldn't
    // catch dedup or budget-accounting bugs that only manifest at larger k;
    // checking every bucket against the closed form does.
    //
    // Budget-to-k mapping: budget 1 is the bare terminal (k=1, no Mul node);
    // budget 2 is unreachable (the cheapest 2-factor product costs 3: two
    // Variable leaves + one flattened Mul); budget b>=3 is k=b-1 factors
    // (k Variable leaves + one flattened Mul node).
    std::vector<Operon::Hash> const vars{ 10, 20, 30 };
    std::size_t const n = vars.size();
    constexpr std::size_t maxComplexity = 8;
    Grammar grammar(PrimitiveSet::Arithmetic, vars);
    Operon::RandomGenerator rng(42);
    EnumerationEngine engine(grammar, maxComplexity, rng);

    engine.Build();

    CHECK(engine.Bucket(GrammarSymbol::SimpleTerm, 1).size() == Multichoose(n, 1));
    CHECK(engine.Bucket(GrammarSymbol::SimpleTerm, 2).empty());
    for (std::size_t budget = 3; budget <= maxComplexity; ++budget) {
        auto const k = budget - 1;
        CHECK(engine.Bucket(GrammarSymbol::SimpleTerm, budget).size() == Multichoose(n, k));
    }
}

TEST_CASE("EnumerationEngine - SimpleTerm completeness holds at a larger budget/variable count", "[enumeration]")
{
    // Same closed-form check as above, but with more variables and a higher
    // ceiling - guards against the budget-accounting overshoot (see
    // WorkingBudgetMargin in enumeration.cpp) resurfacing or compounding at
    // larger targets, which the smaller case above wouldn't necessarily catch.
    std::vector<Operon::Hash> const vars{ 10, 20, 30, 40 };
    std::size_t const n = vars.size();
    constexpr std::size_t maxComplexity = 12;
    Grammar grammar(PrimitiveSet::Arithmetic, vars);
    Operon::RandomGenerator rng(7);
    EnumerationEngine engine(grammar, maxComplexity, rng);

    engine.Build();

    CHECK(engine.Bucket(GrammarSymbol::SimpleTerm, 1).size() == Multichoose(n, 1));
    CHECK(engine.Bucket(GrammarSymbol::SimpleTerm, 2).empty());
    for (std::size_t budget = 3; budget <= maxComplexity; ++budget) {
        auto const k = budget - 1;
        CHECK(engine.Bucket(GrammarSymbol::SimpleTerm, budget).size() == Multichoose(n, k));
    }
}

TEST_CASE("EnumerationEngine - Term bucket sizes match the multiset-count closed form", "[enumeration]")
{
    // Term's shape mirrors SimpleTerm's under PrimitiveSet::Arithmetic (no
    // unary functions enabled, so RecurringFactor's only production is the
    // bare-Variable coercion from SimpleTerm's sibling seeding - see
    // Grammar::AllowsVariable) - the same closed form and budget-to-k mapping
    // applies, and exercises the same self-combine budget-accounting path
    // through Term's own coercion layer.
    std::vector<Operon::Hash> const vars{ 10, 20, 30 };
    std::size_t const n = vars.size();
    constexpr std::size_t maxComplexity = 8;
    Grammar grammar(PrimitiveSet::Arithmetic, vars);
    Operon::RandomGenerator rng(42);
    EnumerationEngine engine(grammar, maxComplexity, rng);

    engine.Build();

    CHECK(engine.Bucket(GrammarSymbol::Term, 1).size() == Multichoose(n, 1));
    CHECK(engine.Bucket(GrammarSymbol::Term, 2).empty());
    for (std::size_t budget = 3; budget <= maxComplexity; ++budget) {
        auto const k = budget - 1;
        CHECK(engine.Bucket(GrammarSymbol::Term, budget).size() == Multichoose(n, k));
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
        CHECK(SymbolicComplexity(t) == 3);
    }
}

TEST_CASE("EnumerationEngine - Expression's recursive term-accumulation isn't silently truncated at the ceiling", "[enumeration]")
{
    // Targets the same budget-accounting overshoot as the Term/SimpleTerm
    // completeness tests above, but for Expression's Add(Term, Expression)
    // recursion: Expression is always Add-rooted, so combining a new
    // weighted Term with an already-built Expression merges into ONE flat
    // Add node rather than adding a distinct one - the same overshoot shape
    // as Term/SimpleTerm's Mul self-combine (see WorkingBudgetMargin in
    // enumeration.cpp). Expression's full combinatorial closed form is
    // harder to derive by hand than Term/SimpleTerm's clean multiset count
    // (it depends on how Simplify() handles repeated-identical terms), so
    // rather than asserting an exact count, this checks the specific
    // failure mode found: that 2-term expressions aren't silently absent at
    // the budget where they first become reachable, alongside the always-
    // present (no recursion needed) 1-term baseline.
    std::vector<Operon::Hash> const vars{ 10, 20 };
    Grammar grammar(PrimitiveSet::Arithmetic, vars);
    Operon::RandomGenerator rng(42);
    constexpr std::size_t maxComplexity = 5;
    EnumerationEngine engine(grammar, maxComplexity, rng);

    engine.Build();

    // 1-term expressions (Add(weight*t, bias) for each t in Term[3]) alone
    // account for exactly term3.size() entries at complexity 5.
    auto term3 = engine.Bucket(GrammarSymbol::Term, 3);
    auto expr5 = engine.Bucket(GrammarSymbol::Expression, maxComplexity);
    CHECK(expr5.size() > term3.size());
    for (auto const& t : expr5) {
        CHECK(SymbolicComplexity(t) == maxComplexity);
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
            CHECK(SymbolicComplexity(t) <= maxComplexity);
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
        CHECK(SymbolicComplexity(t) == 4);
    }
}

} // namespace Operon::Test
