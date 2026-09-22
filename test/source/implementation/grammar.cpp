// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <span>

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
    CHECK(term[0].Operands == std::vector<ProductionOperand>{ GrammarSymbol::RecurringFactor });
    CHECK(term[1].Op == BuiltinOp::Mul);
    CHECK(term[1].Operands == std::vector<ProductionOperand>{ GrammarSymbol::Term, GrammarSymbol::Term });

    auto simpleTerm = grammar.Productions(GrammarSymbol::SimpleTerm);
    REQUIRE(simpleTerm.size() == 1);
    CHECK(simpleTerm[0].Op == BuiltinOp::Mul);
    CHECK(simpleTerm[0].Operands == std::vector<ProductionOperand>{ GrammarSymbol::SimpleTerm, GrammarSymbol::SimpleTerm });
}

TEST_CASE("Grammar - Expression and SimpleExpr production shapes", "[grammar]")
{
    Grammar grammar(PrimitiveSet::Arithmetic, { 1 });

    auto expr = grammar.Productions(GrammarSymbol::Expression);
    REQUIRE(expr.size() == 2);
    CHECK(expr[0].Op == BuiltinOp::Add);
    CHECK(expr[0].WeightFirstOperand);
    CHECK(expr[0].TrailingConstant);
    CHECK(expr[0].Operands == std::vector<ProductionOperand>{ GrammarSymbol::Term });
    CHECK(expr[1].Op == BuiltinOp::Add);
    CHECK(expr[1].WeightFirstOperand);
    CHECK_FALSE(expr[1].TrailingConstant);
    CHECK(expr[1].Operands == std::vector<ProductionOperand>{ GrammarSymbol::Term, GrammarSymbol::Expression });

    auto simpleExpr = grammar.Productions(GrammarSymbol::SimpleExpr);
    REQUIRE(simpleExpr.size() == 2);
    CHECK(simpleExpr[0].Operands == std::vector<ProductionOperand>{ GrammarSymbol::SimpleTerm });
    CHECK(simpleExpr[1].Operands == std::vector<ProductionOperand>{ GrammarSymbol::SimpleTerm, GrammarSymbol::SimpleExpr });
}

TEST_CASE("EnumerationFunctionName/ParseEnumerationFunction round-trip every value", "[grammar]")
{
    for (std::size_t i = 0; i < EnumerationFunctions::Count; ++i) {
        auto const fn = static_cast<EnumerationFunction>(i);
        auto const name = EnumerationFunctionName(fn);
        auto const parsed = ParseEnumerationFunction(name);
        REQUIRE(parsed.has_value());
        CHECK(*parsed == fn);
    }
    CHECK_FALSE(ParseEnumerationFunction("not_a_real_symbol").has_value());
}

TEST_CASE("ParseEnumerationPreset accepts exactly the six ESR names, rejects custom", "[grammar]")
{
    CHECK(ParseEnumerationPreset("keep_duplicates") == EnumerationPreset::KeepDuplicates);
    CHECK(ParseEnumerationPreset("core_maths") == EnumerationPreset::CoreMaths);
    CHECK(ParseEnumerationPreset("ext_maths") == EnumerationPreset::ExtMaths);
    CHECK(ParseEnumerationPreset("osc_maths") == EnumerationPreset::OscMaths);
    CHECK(ParseEnumerationPreset("base10_maths") == EnumerationPreset::Base10Maths);
    CHECK(ParseEnumerationPreset("base_e_maths") == EnumerationPreset::BaseEMaths);
    CHECK_FALSE(ParseEnumerationPreset("custom").has_value());
    CHECK_FALSE(ParseEnumerationPreset("bogus").has_value());
}

namespace {
    // Every named preset always enables exactly these five binary ops, in addition to its own unary set.
    constexpr std::array AlwaysOnBinary {
        EnumerationFunction::Add, EnumerationFunction::Sub, EnumerationFunction::Mul,
        EnumerationFunction::Div, EnumerationFunction::Pow,
    };

    auto CheckPresetFunctions(EnumerationPreset preset, std::span<EnumerationFunction const> unary) -> void {
        auto const functions = PresetFunctions(preset);
        for (auto fn : AlwaysOnBinary) { CHECK(functions.Test(EnumerationFunctions::GetIndex(fn))); }
        for (auto fn : unary) { CHECK(functions.Test(EnumerationFunctions::GetIndex(fn))); }
        // Every other EnumerationFunction (not in AlwaysOnBinary or `unary`) must stay disabled - a
        // preset must emit *exactly* its declared roots, not a superset.
        for (std::size_t i = 0; i < EnumerationFunctions::Count; ++i) {
            auto const fn = static_cast<EnumerationFunction>(i);
            bool const expected = std::ranges::contains(AlwaysOnBinary, fn) || std::ranges::contains(unary, fn);
            CHECK(functions.Test(i) == expected);
        }
    }
} // namespace

TEST_CASE("PresetFunctions - each named preset emits exactly its declared unary set plus the five binary ops", "[grammar]")
{
    using EF = EnumerationFunction;
    CheckPresetFunctions(EnumerationPreset::KeepDuplicates, std::array{ EF::Square, EF::Exp, EF::Inv, EF::SqrtAbs, EF::LogAbs });
    CheckPresetFunctions(EnumerationPreset::CoreMaths, std::array{ EF::Inv });
    CheckPresetFunctions(EnumerationPreset::ExtMaths, std::array{ EF::Inv, EF::SqrtAbs, EF::Square, EF::Exp });
    CheckPresetFunctions(EnumerationPreset::OscMaths, std::array{ EF::Inv, EF::Sin });
    CheckPresetFunctions(EnumerationPreset::Base10Maths, std::array{ EF::TenExp, EF::Inv, EF::Log10Abs });
    CheckPresetFunctions(EnumerationPreset::BaseEMaths, std::array{ EF::Inv, EF::Exp, EF::LogAbs });
}

TEST_CASE("Grammar::Configure(EnumerationFunctionSet) wires every enabled recipe onto RecurringFactor", "[grammar]")
{
    // base10_maths: tenexp (Pow with a fixed base operand), inv (unary Div), log10_abs (Logabs with
    // ResultScale), plus the five always-on binary ops (Add/Sub/Mul/Div/Pow, each Operands.size()==2,
    // both nonterminal).
    Grammar grammar;
    grammar.SetVariables({ 1, 2 });
    grammar.Configure(PresetFunctions(EnumerationPreset::Base10Maths));
    auto ps = grammar.Productions(GrammarSymbol::RecurringFactor);
    REQUIRE(ps.size() == 8); // tenexp, inv, log10_abs, add, sub, mul, div, pow

    auto findOp = [&](BuiltinOp op) -> Production const* {
        for (auto const& p : ps) { if (p.Op == op && p.Operands.size() == 1) { return &p; } }
        return nullptr;
    };

    // inv: Op=Div, single nonterminal operand, no fixed operand, no ResultScale.
    auto const* inv = findOp(BuiltinOp::Div);
    REQUIRE(inv != nullptr);
    CHECK_FALSE(inv->Operands.front().IsFixed());
    CHECK(inv->ResultScale == Operon::Scalar{1});

    // log10_abs: Op=Logabs, single nonterminal operand, ResultScale == 1/ln(10).
    auto const* log10Abs = findOp(BuiltinOp::Logabs);
    REQUIRE(log10Abs != nullptr);
    CHECK_FALSE(log10Abs->Operands.front().IsFixed());
    CHECK(log10Abs->ResultScale != Operon::Scalar{1});

    // tenexp: Op=Pow, two operands - first fixed (base 10), second nonterminal (exponent). Distinct
    // from the ordinary binary Pow(SimpleExpr,SimpleExpr) production (also Op=Pow, Operands.size()==2)
    // by having a fixed operand.
    Production const* tenexp = nullptr;
    for (auto const& p : ps) {
        if (p.Op == BuiltinOp::Pow && p.Operands.size() == 2 && (p.Operands[0].IsFixed() || p.Operands[1].IsFixed())) { tenexp = &p; break; }
    }
    REQUIRE(tenexp != nullptr);
    CHECK(tenexp->Operands[0].IsFixed());
    CHECK(tenexp->Operands[0].FixedValue == Operon::Scalar{10});
    CHECK_FALSE(tenexp->Operands[1].IsFixed());

    // The five binary ops: both nonterminal operands, over SimpleExpr.
    for (auto op : { BuiltinOp::Add, BuiltinOp::Sub, BuiltinOp::Mul, BuiltinOp::Div, BuiltinOp::Pow }) {
        Production const* binary = nullptr;
        for (auto const& p : ps) {
            if (p.Op == op && p.Operands.size() == 2 && !p.Operands[0].IsFixed() && !p.Operands[1].IsFixed()) { binary = &p; break; }
        }
        REQUIRE(binary != nullptr);
        CHECK(binary->Operands[0] == ProductionOperand{ GrammarSymbol::SimpleExpr });
        CHECK(binary->Operands[1] == ProductionOperand{ GrammarSymbol::SimpleExpr });
    }
    // Add/Mul are commutative, Sub/Div/Pow are not - checked only on the pure two-nonterminal-operand
    // productions (Commutative is meaningless for tenexp's mixed fixed/nonterminal Pow, which never
    // reaches ProcessNonterminal's two-nonterminal-operand branch that reads it).
    auto isPureBinary = [](Production const& p) { return p.Operands.size() == 2 && !p.Operands[0].IsFixed() && !p.Operands[1].IsFixed(); };
    for (auto op : { BuiltinOp::Add, BuiltinOp::Mul }) {
        for (auto const& p : ps) { if (p.Op == op && isPureBinary(p)) { CHECK(p.Commutative); } }
    }
    for (auto op : { BuiltinOp::Sub, BuiltinOp::Div, BuiltinOp::Pow }) {
        for (auto const& p : ps) { if (p.Op == op && isPureBinary(p)) { CHECK_FALSE(p.Commutative); } }
    }
}

TEST_CASE("UnderlyingPrimitives maps virtual recipes onto their reused BuiltinOp", "[grammar]")
{
    auto cubeConfig = UnderlyingPrimitives(ToFunctionSet(EnumerationFunction::Cube));
    CHECK(cubeConfig.Test(static_cast<std::size_t>(BuiltinOp::Pow)));

    auto tenExpConfig = UnderlyingPrimitives(ToFunctionSet(EnumerationFunction::TenExp));
    CHECK(tenExpConfig.Test(static_cast<std::size_t>(BuiltinOp::Pow)));

    auto invConfig = UnderlyingPrimitives(ToFunctionSet(EnumerationFunction::Inv));
    CHECK(invConfig.Test(static_cast<std::size_t>(BuiltinOp::Div)));

    auto log10AbsConfig = UnderlyingPrimitives(ToFunctionSet(EnumerationFunction::Log10Abs));
    CHECK(log10AbsConfig.Test(static_cast<std::size_t>(BuiltinOp::Logabs)));
}

} // namespace Operon::Test
