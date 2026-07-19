// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <limits>

#include "operon/core/grammar.hpp"
#include "operon/core/pset.hpp"

namespace Operon::Test {

namespace {
    auto HasUnaryProduction(std::span<Production const> ps, BuiltinOp op) -> bool {
        return std::ranges::any_of(ps, [&](auto const& p) { return p.Op == op; });
    }
} // namespace

TEST_CASE("Grammar - Arithmetic config has no unary RecurringFactor productions", "[grammar]")
{
    Grammar grammar(PrimitiveSet::Arithmetic, { 1, 2, 3 });
    auto ps = grammar.Productions(GrammarSymbol::RecurringFactor);
    CHECK(ps.empty());
}

TEST_CASE("Grammar - TypeCoherent config enables Log/Exp/Sin but not Sqrt/Cbrt", "[grammar]")
{
    Grammar grammar(PrimitiveSet::TypeCoherent, { 1, 2, 3 });
    auto ps = grammar.Productions(GrammarSymbol::RecurringFactor);

    CHECK(HasUnaryProduction(ps, BuiltinOp::Log));
    CHECK(HasUnaryProduction(ps, BuiltinOp::Exp));
    CHECK(HasUnaryProduction(ps, BuiltinOp::Sin));
    CHECK_FALSE(HasUnaryProduction(ps, BuiltinOp::Sqrt));
    CHECK_FALSE(HasUnaryProduction(ps, BuiltinOp::Cbrt));

    // every RecurringFactor production wraps a SimpleExpr operand
    for (auto const& p : ps) {
        REQUIRE(p.Operands.size() == 1);
        CHECK(p.Operands.front() == GrammarSymbol::SimpleExpr);
    }
}

TEST_CASE("Grammar - Full config enables Sqrt/Cbrt too", "[grammar]")
{
    Grammar grammar(PrimitiveSet::Full, { 1, 2, 3 });
    auto ps = grammar.Productions(GrammarSymbol::RecurringFactor);

    CHECK(HasUnaryProduction(ps, BuiltinOp::Log));
    CHECK(HasUnaryProduction(ps, BuiltinOp::Exp));
    CHECK(HasUnaryProduction(ps, BuiltinOp::Sin));
    CHECK(HasUnaryProduction(ps, BuiltinOp::Sqrt));
    CHECK(HasUnaryProduction(ps, BuiltinOp::Cbrt));
}

TEST_CASE("Grammar - Configure is independent of Reconfigure order", "[grammar]")
{
    Grammar grammar;
    grammar.Configure(PrimitiveSet::Full);
    grammar.SetVariables({ 1, 2, 3 });
    auto ps = grammar.Productions(GrammarSymbol::RecurringFactor);
    CHECK(HasUnaryProduction(ps, BuiltinOp::Sqrt));
    CHECK(grammar.VariableHashes().size() == 3);
}

TEST_CASE("Grammar - VariableHashes matches what was set", "[grammar]")
{
    std::vector<Operon::Hash> const vars{ 10, 20, 30, 40 };
    Grammar grammar(PrimitiveSet::Arithmetic, vars);
    auto got = grammar.VariableHashes();
    REQUIRE(got.size() == vars.size());
    CHECK(std::equal(got.begin(), got.end(), vars.begin()));
}

TEST_CASE("Grammar - AllowsVariable is true only for RecurringFactor/SimpleTerm", "[grammar]")
{
    Grammar grammar(PrimitiveSet::Full, { 1 });
    CHECK(grammar.AllowsVariable(GrammarSymbol::RecurringFactor));
    CHECK(grammar.AllowsVariable(GrammarSymbol::SimpleTerm));
    CHECK_FALSE(grammar.AllowsVariable(GrammarSymbol::Term));
    CHECK_FALSE(grammar.AllowsVariable(GrammarSymbol::Expression));
    CHECK_FALSE(grammar.AllowsVariable(GrammarSymbol::SimpleExpr));
}

TEST_CASE("Grammar - MinComplexity with no variables is unreachable everywhere", "[grammar]")
{
    Grammar grammar(PrimitiveSet::Full, {});
    constexpr auto Unreachable = std::numeric_limits<size_t>::max();
    CHECK(grammar.MinComplexity(GrammarSymbol::RecurringFactor) == Unreachable);
    CHECK(grammar.MinComplexity(GrammarSymbol::Term) == Unreachable);
    CHECK(grammar.MinComplexity(GrammarSymbol::Expression) == Unreachable);
}

TEST_CASE("Grammar - default constructor leaves MinComplexity unreachable everywhere, not zero", "[grammar]")
{
    Grammar const grammar; // must behave like Grammar(PrimitiveSetConfig{}, {}), not leave minComplexity_ zero-initialized
    constexpr auto Unreachable = std::numeric_limits<size_t>::max();
    CHECK(grammar.MinComplexity(GrammarSymbol::RecurringFactor) == Unreachable);
    CHECK(grammar.MinComplexity(GrammarSymbol::Term) == Unreachable);
    CHECK(grammar.MinComplexity(GrammarSymbol::SimpleTerm) == Unreachable);
    CHECK(grammar.MinComplexity(GrammarSymbol::Expression) == Unreachable);
    CHECK(grammar.MinComplexity(GrammarSymbol::SimpleExpr) == Unreachable);
}

TEST_CASE("Grammar - MinComplexity fixed point with variables present", "[grammar]")
{
    // A bare variable (RecurringFactor/SimpleTerm's terminal case) has
    // complexity 1; Term/SimpleTerm's Mul-self-combine can never beat their
    // own coercion/terminal base case, so they stay at 1 too. Expression's
    // and SimpleExpr's cheapest shape is "const*x + const" - a Variable, a
    // Mul (from the implicit weight), and an Add (from the trailing bias) -
    // complexity 3 (the two Constant leaves don't count).
    Grammar grammar(PrimitiveSet::Full, { 1, 2, 3 });
    CHECK(grammar.MinComplexity(GrammarSymbol::RecurringFactor) == 1);
    CHECK(grammar.MinComplexity(GrammarSymbol::Term) == 1);
    CHECK(grammar.MinComplexity(GrammarSymbol::SimpleTerm) == 1);
    CHECK(grammar.MinComplexity(GrammarSymbol::Expression) == 3);
    CHECK(grammar.MinComplexity(GrammarSymbol::SimpleExpr) == 3);
}

TEST_CASE("Grammar - Term and SimpleTerm production shapes", "[grammar]")
{
    Grammar grammar(PrimitiveSet::Arithmetic, { 1 });

    auto term = grammar.Productions(GrammarSymbol::Term);
    REQUIRE(term.size() == 2);
    CHECK(term[0].IsCoercion());
    CHECK(term[0].Operands == std::vector{ GrammarSymbol::RecurringFactor });
    CHECK(term[1].Op == BuiltinOp::Mul);
    CHECK(term[1].Operands == std::vector{ GrammarSymbol::Term, GrammarSymbol::Term });

    auto simpleTerm = grammar.Productions(GrammarSymbol::SimpleTerm);
    REQUIRE(simpleTerm.size() == 1);
    CHECK(simpleTerm[0].Op == BuiltinOp::Mul);
    CHECK(simpleTerm[0].Operands == std::vector{ GrammarSymbol::SimpleTerm, GrammarSymbol::SimpleTerm });
}

TEST_CASE("Grammar - Expression and SimpleExpr production shapes", "[grammar]")
{
    Grammar grammar(PrimitiveSet::Arithmetic, { 1 });

    auto expr = grammar.Productions(GrammarSymbol::Expression);
    REQUIRE(expr.size() == 2);
    CHECK(expr[0].Op == BuiltinOp::Add);
    CHECK(expr[0].WeightFirstOperand);
    CHECK(expr[0].TrailingConstant);
    CHECK(expr[0].Operands == std::vector{ GrammarSymbol::Term });
    CHECK(expr[1].Op == BuiltinOp::Add);
    CHECK(expr[1].WeightFirstOperand);
    CHECK_FALSE(expr[1].TrailingConstant);
    CHECK(expr[1].Operands == std::vector{ GrammarSymbol::Term, GrammarSymbol::Expression });

    auto simpleExpr = grammar.Productions(GrammarSymbol::SimpleExpr);
    REQUIRE(simpleExpr.size() == 2);
    CHECK(simpleExpr[0].Operands == std::vector{ GrammarSymbol::SimpleTerm });
    CHECK(simpleExpr[1].Operands == std::vector{ GrammarSymbol::SimpleTerm, GrammarSymbol::SimpleExpr });
}

} // namespace Operon::Test
