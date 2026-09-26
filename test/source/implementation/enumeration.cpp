// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include "../operon_test.hpp"

#include <cmath>
#include <set>
#include <string>

#include "operon/algorithms/enumeration.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/grammar.hpp"
#include "operon/algorithms/domain_pruning.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/pset.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/optimizer/optimizer.hpp"
#include "operon/core/serialization.hpp"

namespace Operon::Test {

namespace {
    // C(n+k-1, k) ("n multichoose k"), via the standard running-product form,
    // which stays exact at every step since each prefix is itself a binomial
    // coefficient. Used by the completeness tests below to check every
    // Term/SimpleTerm bucket against its closed-form multiset count, rather
    // than spot-checking a handful of small buckets. Not overflow-guarded -
    // fine at these tests' scale (largest call here is well under 1000), but
    // don't reuse this helper for large n/k without adding a check.
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
    Tree const tree = Tree({ nx, ny, Util::MakeOp<BuiltinOp::Add>(), Node::Constant(2.0), Util::MakeOp<BuiltinOp::Mul>() }).UpdateNodes();
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

TEST_CASE("EnumerationEngine - Aq production is gated by PrimitiveSetConfig", "[enumeration]")
{
    std::vector<Operon::Hash> const vars{ 10, 20 };
    Operon::RandomGenerator rng(42);
    constexpr std::size_t maxComplexity = 12;

    // TypeCoherent doesn't include Aq - no RecurringFactor bucket should ever contain an Aq-rooted tree.
    {
        Grammar grammar(PrimitiveSet::TypeCoherent, vars);
        EnumerationEngine engine(grammar, maxComplexity, rng);
        engine.Build();
        for (std::size_t b = 1; b <= maxComplexity; ++b) {
            for (auto const& t : engine.Bucket(GrammarSymbol::RecurringFactor, b)) {
                CHECK_FALSE(t.Nodes().back().IsAq());
            }
        }
    }

    // Full includes Aq - at least one Aq-rooted candidate should be reachable within this budget.
    {
        Grammar grammar(PrimitiveSet::Full, vars);
        EnumerationEngine engine(grammar, maxComplexity, rng);
        engine.Build();
        bool foundAq = false;
        for (std::size_t b = 1; b <= maxComplexity && !foundAq; ++b) {
            for (auto const& t : engine.Bucket(GrammarSymbol::RecurringFactor, b)) {
                if (t.Nodes().back().IsAq()) { foundAq = true; break; }
            }
        }
        CHECK(foundAq);
    }
}

TEST_CASE("EnumerationEngine - Aq's non-commutative self-combine isn't half-dropped by the symmetric skip", "[enumeration]")
{
    // Aq (Operands = {SimpleExpr, SimpleExpr}) is a same-symbol production like Term/SimpleTerm's Mul self-combine,
    // but unlike Mul, aq(a,b) != aq(b,a) - Production::Commutative=false on this production is what keeps
    // ProcessNonterminal's b0 > b1 redundant-work skip (correct only for a genuinely commutative same-symbol
    // combine) from applying here. Regression coverage: find two distinct nonempty SimpleExpr budgets and check
    // both operand-budget orderings actually land in the RecurringFactor bucket - a reintroduced blanket
    // op0==op1 skip would silently keep only the b0<=b1 direction.
    std::vector<Operon::Hash> const vars{ 10, 20 };
    constexpr std::size_t maxComplexity = 14;
    Grammar grammar(PrimitiveSet::Full, vars);
    Operon::RandomGenerator rng(42);
    EnumerationEngine engine(grammar, maxComplexity, rng);

    engine.Build();

    std::size_t b1 = 0;
    std::size_t b2 = 0;
    for (std::size_t b = 3; b <= maxComplexity && b2 == 0; ++b) {
        if (engine.Bucket(GrammarSymbol::SimpleExpr, b).empty()) { continue; }
        if (b1 == 0) { b1 = b; } else { b2 = b; }
    }
    REQUIRE(b1 != 0);
    REQUIRE(b2 != 0);

    auto const target = 1 + b1 + b2;
    REQUIRE(target <= maxComplexity);

    auto const size1 = engine.Bucket(GrammarSymbol::SimpleExpr, b1).size();
    auto const size2 = engine.Bucket(GrammarSymbol::SimpleExpr, b2).size();

    std::size_t aqCount = 0;
    for (auto const& t : engine.Bucket(GrammarSymbol::RecurringFactor, target)) {
        if (t.Nodes().back().IsAq()) { ++aqCount; }
    }
    // Both the (b1,b2) and (b2,b1) operand-budget splits land at this same target budget (1 + b1 + b2 either way):
    // size1*size2 candidates from each direction, and aq(a,b) != aq(b,a) so none of them collide/dedup against
    // each other.
    CHECK(aqCount == 2 * size1 * size2);
}

namespace {
    // Problem is non-movable, so this configures one in place rather than
    // returning it - callers construct `Operon::Problem problem(&ds);` and
    // pass it here by reference.
    void ConfigureProblem(Operon::Dataset& ds, Operon::Problem& problem) {
        auto inputs = ds.VariableHashes();
        std::erase(inputs, ds.GetVariable("Y").value().Hash);
        problem.SetInputs(inputs);
        problem.SetTarget("Y");
        problem.SetTrainingRange({ 0, 50 }); // small subset - this is a wiring smoke test, not a fit-quality test (see Phase 5)
        problem.SetTestRange({ 0, 50 });
    }
} // namespace

TEST_CASE("GrammarEnumerationAlgorithm - Run fits coefficients and tracks best trees", "[enumeration]")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    Operon::Problem problem(&ds);
    ConfigureProblem(ds, problem);

    using DTable = DispatchTable<Operon::Scalar>;
    DTable dtable;
    LBFGSOptimizer<DTable, GaussianLoss<Operon::Scalar>> optimizer{ &dtable, &problem };
    Operon::Evaluator<DTable> evaluator{ &problem, &dtable, Operon::R2{} };

    Grammar grammar(PrimitiveSet::Arithmetic, problem.GetInputs());
    EnumerationConfig config;
    config.MaxComplexity = 4;
    config.TopK = 3;
    config.Ranking = EnumerationRanking::Objective;
    config.EvaluationBufferSize = problem.TrainingRange().Size();

    Operon::RandomGenerator engineRng(42);
    GrammarEnumerationAlgorithm algo(config, grammar, &optimizer, MakeObjectiveScorer(&evaluator), engineRng);

    Operon::RandomGenerator fitRng(42);
    algo.Run(fitRng);

    auto best = algo.BestTrees();
    REQUIRE_FALSE(best.empty());
    CHECK(best.size() <= config.TopK);
    for (auto const& r : best) {
        CHECK(std::isfinite(r.Score));
        CHECK(SymbolicComplexity(r.Tree) <= config.MaxComplexity);
    }
    for (std::size_t i = 1; i < best.size(); ++i) {
        CHECK(best[i - 1].Score <= best[i].Score); // ascending by Score (lower = better)
    }
}

TEST_CASE("GrammarEnumerationAlgorithm - TopK == 0 keeps nothing rather than crashing", "[enumeration]")
{
    // Regression test: ConsiderBest's capacity check (best_.size() >=
    // config_.TopK) is trivially true when TopK == 0 (0 >= 0), and used to
    // fall through to best_.back() on an empty vector - a segfault.
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    Operon::Problem problem(&ds);
    ConfigureProblem(ds, problem);

    using DTable = DispatchTable<Operon::Scalar>;
    DTable dtable;
    LBFGSOptimizer<DTable, GaussianLoss<Operon::Scalar>> optimizer{ &dtable, &problem };
    Operon::Evaluator<DTable> evaluator{ &problem, &dtable, Operon::R2{} };

    Grammar grammar(PrimitiveSet::Arithmetic, problem.GetInputs());
    EnumerationConfig config;
    config.MaxComplexity = 4;
    config.TopK = 0;
    config.EvaluationBufferSize = problem.TrainingRange().Size();

    Operon::RandomGenerator engineRng(42);
    GrammarEnumerationAlgorithm algo(config, grammar, &optimizer, MakeObjectiveScorer(&evaluator), engineRng);

    Operon::RandomGenerator fitRng(42);
    algo.Run(fitRng);

    CHECK(algo.BestTrees().empty());
}

TEST_CASE("GrammarEnumerationAlgorithm - RequestStop halts Run early", "[enumeration]")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    Operon::Problem problem(&ds);
    ConfigureProblem(ds, problem);

    using DTable = DispatchTable<Operon::Scalar>;
    DTable dtable;
    LBFGSOptimizer<DTable, GaussianLoss<Operon::Scalar>> optimizer{ &dtable, &problem };
    Operon::Evaluator<DTable> evaluator{ &problem, &dtable, Operon::R2{} };

    Grammar grammar(PrimitiveSet::Arithmetic, problem.GetInputs());
    EnumerationConfig config;
    config.MaxComplexity = 20; // deliberately large, so an early stop is meaningfully "early"
    config.TopK = 3;
    config.EvaluationBufferSize = problem.TrainingRange().Size();

    Operon::RandomGenerator engineRng(42);
    GrammarEnumerationAlgorithm algo(config, grammar, &optimizer, MakeObjectiveScorer(&evaluator), engineRng);

    Operon::RandomGenerator fitRng(42);
    int reportCalls = 0;
    algo.Run(fitRng, [&]() -> bool {
        ++reportCalls;
        return reportCalls >= 2; // stop after the 2nd budget-level report
    });

    CHECK(algo.StopRequested());
    CHECK(reportCalls == 2);
}

TEST_CASE("GrammarEnumerationAlgorithm - recovers a small ground-truth expression", "[enumeration]")
{
    // y = 2*x0 + 3*x1 - 1 : a two-term weighted sum, exercising Expression's
    // {Term, Expression} continuation production (not just its single-term
    // base case) - needs complexity 6 (fixedCost 2 for the outer weight+Add,
    // +1 for x0, +3 for the inner Expression's own minimal "const*x1+const"
    // shape), the smallest MaxComplexity that can reach a two-term sum.
    constexpr auto Nrow = 50;
    Operon::RandomGenerator dataRng(1234);
    std::vector<Operon::Scalar> x0(Nrow);
    std::vector<Operon::Scalar> x1(Nrow);
    std::vector<Operon::Scalar> y(Nrow);
    for (auto i = 0; i < Nrow; ++i) {
        x0[i] = Operon::Random::Uniform(dataRng, -1.0F, +1.0F);
        x1[i] = Operon::Random::Uniform(dataRng, -1.0F, +1.0F);
        y[i] = 2.0F * x0[i] + 3.0F * x1[i] - 1.0F;
    }
    std::vector<std::vector<Operon::Scalar>> cols{ x0, x1, y };
    Operon::Dataset ds(cols);

    Operon::Problem problem(&ds);
    std::vector<Operon::Hash> const inputs{ ds.GetVariable("X1").value().Hash, ds.GetVariable("X2").value().Hash };
    problem.SetInputs(inputs);
    problem.SetTarget("X3");
    problem.SetTrainingRange({ 0, Nrow });
    problem.SetTestRange({ 0, Nrow });

    using DTable = DispatchTable<Operon::Scalar>;
    DTable dtable;
    LBFGSOptimizer<DTable, GaussianLoss<Operon::Scalar>> optimizer{ &dtable, &problem };
    Operon::Evaluator<DTable> evaluator{ &problem, &dtable, Operon::R2{} };

    Grammar grammar(PrimitiveSet::Arithmetic, problem.GetInputs());
    EnumerationConfig config;
    config.MaxComplexity = 6;
    config.TopK = 5;
    config.EvaluationBufferSize = problem.TrainingRange().Size();

    Operon::RandomGenerator engineRng(42);
    GrammarEnumerationAlgorithm algo(config, grammar, &optimizer, MakeObjectiveScorer(&evaluator), engineRng);
    Operon::RandomGenerator fitRng(42);
    algo.Run(fitRng);

    auto best = algo.BestTrees();
    REQUIRE_FALSE(best.empty());
    // R2's Evaluator convention is -R2Score (lower = better, matching every
    // other Operon ErrorMetric) - a near-perfect fit approaches -1, not 0.
    CHECK(best.front().Score < -0.99); // near-perfect fit for an exactly-representable linear ground truth
}

TEST_CASE("GrammarEnumerationAlgorithm - threaded runs are reproducible", "[enumeration][determinism]")
{
    constexpr auto rows = 50;
    Operon::RandomGenerator dataRng(1234);
    std::vector<Operon::Scalar> x1(rows);
    std::vector<Operon::Scalar> x2(rows);
    std::vector<Operon::Scalar> y(rows);
    for (auto i = 0; i < rows; ++i) {
        x1[i] = Operon::Random::Uniform(dataRng, -1.0F, +1.0F);
        x2[i] = Operon::Random::Uniform(dataRng, -1.0F, +1.0F);
        y[i] = 2.0F * x1[i] + 3.0F * x2[i] - 1.0F;
    }
    std::vector<std::vector<Operon::Scalar>> cols { x1, x2, y };
    Operon::Dataset ds(cols);
    Operon::Problem problem(&ds);
    std::vector<Operon::Hash> const inputs { ds.GetVariable("X1").value().Hash, ds.GetVariable("X2").value().Hash };
    problem.SetInputs(inputs);
    problem.SetTarget("X3");
    problem.SetTrainingRange({ 0, rows });
    problem.SetTestRange({ 0, rows });

    using DTable = DispatchTable<Operon::Scalar>;
    DTable dtable;
    LBFGSOptimizer<DTable, GaussianLoss<Operon::Scalar>> optimizer{ &dtable, &problem };
    Operon::Evaluator<DTable> evaluator{ &problem, &dtable, Operon::R2{} };
    EnumerationConfig config { .MaxComplexity = 6, .TopK = 5, .Ranking = EnumerationRanking::Objective,
                               .EvaluationBufferSize = problem.TrainingRange().Size() };
    auto run = [&]() {
        Operon::RandomGenerator engineRng(42);
        GrammarEnumerationAlgorithm algo(config, Grammar(PrimitiveSet::Arithmetic, inputs),
            &optimizer, MakeObjectiveScorer(&evaluator), engineRng);
        Operon::RandomGenerator fitRng(42);
        algo.Run(fitRng, {}, /*threads=*/2);
        return std::vector<EnumerationResult>(algo.BestTrees().begin(), algo.BestTrees().end());
    };

    auto const first = run();
    auto const second = run();
    REQUIRE(first.size() == second.size());
    for (std::size_t i = 0; i < first.size(); ++i) {
        CHECK(first[i].Score == second[i].Score);
        CHECK(first[i].CanonicalKey == second[i].CanonicalKey);
        CHECK(Serialization::ToBeve(first[i].Tree) == Serialization::ToBeve(second[i].Tree));
    }
}

TEST_CASE("GrammarEnumerationAlgorithm - report can stop fitting batches", "[enumeration]")
{
    constexpr auto rows = 50;
    Operon::RandomGenerator dataRng(1234);
    std::vector<Operon::Scalar> x1(rows);
    std::vector<Operon::Scalar> x2(rows);
    std::vector<Operon::Scalar> y(rows);
    for (auto i = 0; i < rows; ++i) {
        x1[i] = Operon::Random::Uniform(dataRng, -1.0F, +1.0F);
        x2[i] = Operon::Random::Uniform(dataRng, -1.0F, +1.0F);
        y[i] = 2.0F * x1[i] + 3.0F * x2[i] - 1.0F;
    }
    std::vector<std::vector<Operon::Scalar>> cols { x1, x2, y };
    Operon::Dataset ds(cols);
    Operon::Problem problem(&ds);
    std::vector<Operon::Hash> const inputs { ds.GetVariable("X1").value().Hash, ds.GetVariable("X2").value().Hash };
    problem.SetInputs(inputs);
    problem.SetTarget("X3");
    problem.SetTrainingRange({ 0, rows });
    problem.SetTestRange({ 0, rows });

    using DTable = DispatchTable<Operon::Scalar>;
    DTable dtable;
    LBFGSOptimizer<DTable, GaussianLoss<Operon::Scalar>> optimizer{ &dtable, &problem };
    Operon::Evaluator<DTable> evaluator{ &problem, &dtable, Operon::R2{} };
    EnumerationConfig config { .MaxComplexity = 6, .TopK = 100, .Ranking = EnumerationRanking::Objective,
                               .EvaluationBufferSize = problem.TrainingRange().Size() };
    Operon::RandomGenerator engineRng(42);
    GrammarEnumerationAlgorithm algo(config, Grammar(PrimitiveSet::Arithmetic, inputs),
        &optimizer, MakeObjectiveScorer(&evaluator), engineRng);
    Operon::RandomGenerator fitRng(42);
    int calls = 0;
    algo.Run(fitRng, [&] { return ++calls >= 11; }, /*threads=*/2);

    CHECK(algo.StopRequested());
    CHECK(calls == 11);
    CHECK_FALSE(algo.BestTrees().empty());
}
TEST_CASE("CanonicalizeEnumerationTree - commutative reordering shares a canonical key", "[enumeration]")
{
    Node nx(NodeType::Variable); nx.HashValue = 1;
    Node ny(NodeType::Variable); ny.HashValue = 2;
    Tree const xy = Tree({ nx, ny, Util::MakeOp<BuiltinOp::Add>() }).UpdateNodes();
    Tree const yx = Tree({ ny, nx, Util::MakeOp<BuiltinOp::Add>() }).UpdateNodes();
    CHECK(CanonicalizeEnumerationTree(xy).Key == CanonicalizeEnumerationTree(yx).Key);

    Tree const xyMul = Tree({ nx, ny, Util::MakeOp<BuiltinOp::Mul>() }).UpdateNodes();
    Tree const yxMul = Tree({ ny, nx, Util::MakeOp<BuiltinOp::Mul>() }).UpdateNodes();
    CHECK(CanonicalizeEnumerationTree(xyMul).Key == CanonicalizeEnumerationTree(yxMul).Key);
}

TEST_CASE("CanonicalizeEnumerationTree - Sub/Div are not commutative", "[enumeration]")
{
    Node nx(NodeType::Variable); nx.HashValue = 1;
    Node ny(NodeType::Variable); ny.HashValue = 2;
    Tree const xy = Tree({ nx, ny, Util::MakeOp<BuiltinOp::Sub>() }).UpdateNodes();
    Tree const yx = Tree({ ny, nx, Util::MakeOp<BuiltinOp::Sub>() }).UpdateNodes();
    CHECK(CanonicalizeEnumerationTree(xy).Key != CanonicalizeEnumerationTree(yx).Key);
}

TEST_CASE("CanonicalizeEnumerationTree - Square(x) and x*x share a canonical key", "[enumeration]")
{
    Node nx(NodeType::Variable); nx.HashValue = 1;
    Tree const squareForm = Tree({ nx, Util::MakeOp<BuiltinOp::Square>() }).UpdateNodes();
    Tree const mulForm = Tree({ nx, nx, Util::MakeOp<BuiltinOp::Mul>() }).UpdateNodes();
    CHECK(CanonicalizeEnumerationTree(squareForm).Key == CanonicalizeEnumerationTree(mulForm).Key);
}

TEST_CASE("CanonicalizeEnumerationTree - optimizable constants are anonymized, fixed constants are not", "[enumeration]")
{
    Node nx(NodeType::Variable); nx.HashValue = 1;

    // 2*x and 99*x, both with an *optimizable* weight - same free-parameter family, same key.
    Node w2 = Node::Constant(2.0); w2.Optimize = true;
    Node w99 = Node::Constant(99.0); w99.Optimize = true;
    Tree const t2 = Tree({ w2, nx, Util::MakeOp<BuiltinOp::Mul>() }).UpdateNodes();
    Tree const t99 = Tree({ w99, nx, Util::MakeOp<BuiltinOp::Mul>() }).UpdateNodes();
    CHECK(CanonicalizeEnumerationTree(t2).Key == CanonicalizeEnumerationTree(t99).Key);

    // A *fixed* (non-optimizable) 2*x is a structurally distinct family from a fixed 3*x - their
    // values are part of the structural identity, not anonymized.
    Node f2 = Node::Constant(2.0); f2.Optimize = false;
    Node f3 = Node::Constant(3.0); f3.Optimize = false;
    Tree const tf2 = Tree({ f2, nx, Util::MakeOp<BuiltinOp::Mul>() }).UpdateNodes();
    Tree const tf3 = Tree({ f3, nx, Util::MakeOp<BuiltinOp::Mul>() }).UpdateNodes();
    CHECK(CanonicalizeEnumerationTree(tf2).Key != CanonicalizeEnumerationTree(tf3).Key);

    // ...and a fixed constant never collides with an optimizable one either.
    CHECK(CanonicalizeEnumerationTree(tf2).Key != CanonicalizeEnumerationTree(t2).Key);
}

TEST_CASE("CanonicalizeEnumerationTree - domain-distinct functions never share a canonical key", "[enumeration]")
{
    Node nx(NodeType::Variable); nx.HashValue = 1;
    Tree const logForm = Tree({ nx, Util::MakeOp<BuiltinOp::Log>() }).UpdateNodes();
    Tree const expForm = Tree({ nx, Util::MakeOp<BuiltinOp::Exp>() }).UpdateNodes();
    Tree const sinForm = Tree({ nx, Util::MakeOp<BuiltinOp::Sin>() }).UpdateNodes();
    auto logKey = CanonicalizeEnumerationTree(logForm).Key;
    auto expKey = CanonicalizeEnumerationTree(expForm).Key;
    auto sinKey = CanonicalizeEnumerationTree(sinForm).Key;
    CHECK(logKey != expKey);
    CHECK(logKey != sinKey);
    CHECK(expKey != sinKey);
}

TEST_CASE("CanonicalizeEnumerationTree - bounded distribution expands (a+b)*(c+d)", "[enumeration]")
{
    Node na(NodeType::Variable); na.HashValue = 1;
    Node nb(NodeType::Variable); nb.HashValue = 2;
    Node nc(NodeType::Variable); nc.HashValue = 3;
    Node nd(NodeType::Variable); nd.HashValue = 4;

    // (a+b)*(c+d)
    Tree const product = Tree({
        na, nb, Util::MakeOp<BuiltinOp::Add>(),
        nc, nd, Util::MakeOp<BuiltinOp::Add>(),
        Util::MakeOp<BuiltinOp::Mul>(),
    }).UpdateNodes();

    // a*c + a*d + b*c + b*d, built directly as a flat sum (Add's postfix arity set to 4).
    auto ac = Util::MakeOp<BuiltinOp::Mul>();
    auto ad = Util::MakeOp<BuiltinOp::Mul>();
    auto bc = Util::MakeOp<BuiltinOp::Mul>();
    auto bd = Util::MakeOp<BuiltinOp::Mul>();
    auto sum = Util::MakeOp<BuiltinOp::Add>(); sum.Arity = 4;
    Tree const expanded = Tree({
        na, nc, ac,
        na, nd, ad,
        nb, nc, bc,
        nb, nd, bd,
        sum,
    }).UpdateNodes();

    CHECK(CanonicalizeEnumerationTree(product).Key == CanonicalizeEnumerationTree(expanded).Key);
}

TEST_CASE("CanonicalizeEnumerationTree - Div/Inv normalize to the same multiplicative-inverse family", "[enumeration]")
{
    Node nx(NodeType::Variable); nx.HashValue = 1;
    Node ny(NodeType::Variable); ny.HashValue = 2;
    Node nz(NodeType::Variable); nz.HashValue = 3;

    // Div(x, y) == Mul(x, Inv(y)) - binary Div and unary Div (1/x) normalize to the same family.
    // Operon lays out binary-op children with the semantic first operand at the rightmost position
    // (immediately preceding the op node - see functions.hpp's Sub/Div/Pow and
    // pappus_backend.cpp's "operon lays out binary-op children with the semantic LEFT operand at the
    // higher index" comment), so Div(x, y) = x/y is built as [y, x, Div], not [x, y, Div].
    auto div2 = Util::MakeOp<BuiltinOp::Div>(); div2.Arity = 2;
    Tree const binaryDiv = Tree({ ny, nx, div2 }).UpdateNodes();
    auto invY = Util::MakeOp<BuiltinOp::Div>(); invY.Arity = 1;
    auto mul = Util::MakeOp<BuiltinOp::Mul>();
    Tree const mulOfInv = Tree({ ny, invY, nx, mul }).UpdateNodes();
    CHECK(CanonicalizeEnumerationTree(binaryDiv).Key == CanonicalizeEnumerationTree(mulOfInv).Key);

    // Inv(Inv(x)) == x - double inversion cancels.
    auto inv1 = Util::MakeOp<BuiltinOp::Div>(); inv1.Arity = 1;
    auto inv2 = Util::MakeOp<BuiltinOp::Div>(); inv2.Arity = 1;
    Tree const doubleInv = Tree({ nx, inv1, inv2 }).UpdateNodes();
    Tree const bare = Tree({ nx }).UpdateNodes();
    CHECK(CanonicalizeEnumerationTree(doubleInv).Key == CanonicalizeEnumerationTree(bare).Key);

    // Div(x, y*z) == Mul(x, Inv(y*z)) - a multi-factor monomial denominator is still invertible.
    // x is the numerator (semantic first operand, rightmost), y*z the denominator - pushed first.
    auto mulYZ = Util::MakeOp<BuiltinOp::Mul>();
    Tree const divByProduct = Tree({ ny, nz, mulYZ, nx, div2 }).UpdateNodes();
    auto invYZ = Util::MakeOp<BuiltinOp::Div>(); invYZ.Arity = 1;
    auto mulYZ2 = Util::MakeOp<BuiltinOp::Mul>();
    auto mul2 = Util::MakeOp<BuiltinOp::Mul>();
    Tree const mulOfInvProduct = Tree({ ny, nz, mulYZ2, invYZ, nx, mul2 }).UpdateNodes();
    CHECK(CanonicalizeEnumerationTree(divByProduct).Key == CanonicalizeEnumerationTree(mulOfInvProduct).Key);

    // A sum denominator (y+z, not a pure monomial) has no closed monomial inverse - falls back to an
    // opaque representation, which must still differ from the multiplied-out family above.
    auto addYZ = Util::MakeOp<BuiltinOp::Add>();
    Tree const divBySum = Tree({ ny, nz, addYZ, nx, div2 }).UpdateNodes();
    CHECK(CanonicalizeEnumerationTree(divBySum).Key != CanonicalizeEnumerationTree(divByProduct).Key);
}

TEST_CASE("CanonicalizeEnumerationTree - fixed-exponent Pow folds to repeated multiplication", "[enumeration]")
{
    Node nx(NodeType::Variable); nx.HashValue = 1;

    // Cube: Pow(x, 3) with a fixed (non-optimizable) exponent operand == x*x*x. Pow(base, exp) is
    // built as [exp, base, Pow] - base is the semantic first operand (rightmost).
    Node exp3 = Node::Constant(3.0); exp3.Optimize = false;
    Tree const cubeForm = Tree({ exp3, nx, Util::MakeOp<BuiltinOp::Pow>() }).UpdateNodes();
    auto mulA = Util::MakeOp<BuiltinOp::Mul>();
    auto mulB = Util::MakeOp<BuiltinOp::Mul>();
    Tree const cubeDirect = Tree({ nx, nx, mulA, nx, mulB }).UpdateNodes();
    CHECK(CanonicalizeEnumerationTree(cubeForm).Key == CanonicalizeEnumerationTree(cubeDirect).Key);

    // Pow(x, 2) folds to the same family as Square(x)/x*x.
    Node exp2 = Node::Constant(2.0); exp2.Optimize = false;
    Tree const powSquare = Tree({ exp2, nx, Util::MakeOp<BuiltinOp::Pow>() }).UpdateNodes();
    Tree const square = Tree({ nx, Util::MakeOp<BuiltinOp::Square>() }).UpdateNodes();
    CHECK(CanonicalizeEnumerationTree(powSquare).Key == CanonicalizeEnumerationTree(square).Key);

    // A general (non-fixed, optimizable) Pow exponent is not foldable - treated opaquely, and must
    // not collide with the fixed-exponent-3 family above.
    Node freeExp = Node::Constant(3.0); freeExp.Optimize = true;
    Tree const opaquePow = Tree({ freeExp, nx, Util::MakeOp<BuiltinOp::Pow>() }).UpdateNodes();
    CHECK(CanonicalizeEnumerationTree(opaquePow).Key != CanonicalizeEnumerationTree(cubeForm).Key);
}

TEST_CASE("CanonicalizeEnumerationTree - Sub is additive negation", "[enumeration]")
{
    Node na(NodeType::Variable); na.HashValue = 1;
    Node nb(NodeType::Variable); nb.HashValue = 2;

    // Sub(a, b) == Add(a, -b), where -b is unary Sub(b). Sub(a,b)=a-b is built as [b, a, Sub] - a
    // (the minuend) is the semantic first operand (rightmost).
    Tree const sub = Tree({ nb, na, Util::MakeOp<BuiltinOp::Sub>() }).UpdateNodes();
    auto negB = Util::MakeOp<BuiltinOp::Sub>(); negB.Arity = 1;
    auto add = Util::MakeOp<BuiltinOp::Add>();
    Tree const addOfNeg = Tree({ na, nb, negB, add }).UpdateNodes();
    CHECK(CanonicalizeEnumerationTree(sub).Key == CanonicalizeEnumerationTree(addOfNeg).Key);

    // Unary Sub(a) (negation) == Sub(0, a): a fixed-zero-minuend Sub collapses to the same negated
    // family as bare unary negation (both anonymize to a plain negated fixed-1 coefficient on `a`).
    // Sub(0,a)=0-a is built as [a, 0, Sub] - 0 (the minuend) is the semantic first operand (rightmost).
    auto negA = Util::MakeOp<BuiltinOp::Sub>(); negA.Arity = 1;
    Tree const unaryNeg = Tree({ na, negA }).UpdateNodes();
    Node zero = Node::Constant(0.0); zero.Optimize = false;
    Tree const zeroMinusA = Tree({ na, zero, Util::MakeOp<BuiltinOp::Sub>() }).UpdateNodes();
    CHECK(CanonicalizeEnumerationTree(unaryNeg).Key == CanonicalizeEnumerationTree(zeroMinusA).Key);
}

TEST_CASE("CanonicalizeEnumerationTree - ExpansionCap overflow falls back to a sound opaque product", "[enumeration]")
{
    // Two sums whose full distribution (17*16 = 272 monomials) exceeds ExpansionCap (256): must not
    // crash, must still be commutative-safe (reordering the outer Mul's operands - or, equivalently,
    // building the same two sums in reverse child order - yields the same key), and must not collide
    // with a smaller/different product that stays under the cap.
    std::vector<Node> lhsVars;
    lhsVars.reserve(17);
    for (Operon::Hash h = 1; h <= 17; ++h) { Node v(NodeType::Variable); v.HashValue = h; lhsVars.push_back(v); }
    std::vector<Node> rhsVars;
    rhsVars.reserve(16);
    for (Operon::Hash h = 101; h <= 116; ++h) { Node v(NodeType::Variable); v.HashValue = h; rhsVars.push_back(v); }

    auto buildFlatSum = [](std::vector<Node> const& vars) {
        std::vector<Node> nodes(vars.begin(), vars.end());
        auto add = Util::MakeOp<BuiltinOp::Add>();
        add.Arity = static_cast<uint16_t>(vars.size());
        nodes.push_back(add);
        return nodes;
    };

    auto lhsSum = buildFlatSum(lhsVars);
    auto rhsSum = buildFlatSum(rhsVars);
    std::vector<Node> product{ lhsSum.begin(), lhsSum.end() };
    product.insert(product.end(), rhsSum.begin(), rhsSum.end());
    product.push_back(Util::MakeOp<BuiltinOp::Mul>());
    Tree const overCap = Tree(product).UpdateNodes();

    std::vector<Node> productReversed{ rhsSum.begin(), rhsSum.end() };
    productReversed.insert(productReversed.end(), lhsSum.begin(), lhsSum.end());
    productReversed.push_back(Util::MakeOp<BuiltinOp::Mul>());
    Tree const overCapReversed = Tree(productReversed).UpdateNodes();

    auto key = CanonicalizeEnumerationTree(overCap).Key;
    CHECK_FALSE(key.empty());
    CHECK(key == CanonicalizeEnumerationTree(overCapReversed).Key);

    // A structurally different over-cap product (drop the last rhs variable) must not collide.
    auto rhsShort = std::vector<Node>(rhsVars.begin(), rhsVars.end() - 1);
    auto rhsShortSum = buildFlatSum(rhsShort);
    std::vector<Node> shortProduct{ lhsSum.begin(), lhsSum.end() };
    shortProduct.insert(shortProduct.end(), rhsShortSum.begin(), rhsShortSum.end());
    shortProduct.push_back(Util::MakeOp<BuiltinOp::Mul>());
    Tree const differentOverCap = Tree(shortProduct).UpdateNodes();
    CHECK(key != CanonicalizeEnumerationTree(differentOverCap).Key);
}

TEST_CASE("GrammarEnumerationAlgorithm - MDL ranking fits exactly one representative per canonical class", "[enumeration]")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    Operon::Problem problem(&ds);
    ConfigureProblem(ds, problem);

    using DTable = DispatchTable<Operon::Scalar>;
    DTable dtable;
    LBFGSOptimizer<DTable, GaussianLoss<Operon::Scalar>> optimizer{ &dtable, &problem };

    Grammar grammar(PrimitiveSet::Arithmetic, problem.GetInputs());
    EnumerationConfig config;
    config.MaxComplexity = 5;
    config.TopK = 100; // large enough to retain every canonical-class representative found
    config.Ranking = EnumerationRanking::MinimumDescriptionLength;
    config.EvaluationBufferSize = problem.TrainingRange().Size();

    Operon::RandomGenerator engineRng(42);
    auto scorer = MakeMdlScorer<DTable, GaussianLikelihood<Operon::Scalar>>(&problem, &dtable);
    GrammarEnumerationAlgorithm algo(config, grammar, &optimizer, std::move(scorer), engineRng);

    Operon::RandomGenerator fitRng(42);
    algo.Run(fitRng);

    auto best = algo.BestTrees();
    REQUIRE_FALSE(best.empty());

    // Every result's CanonicalKey is unique (one representative per canonical class - see Run's doc
    // comment) and every MDL component is finite and correctly composed: Score(bits) ==
    // NegativeLogLikelihood(nats)/ln(2) + ParameterCodeBits + StructureCodeBits.
    std::set<std::string> seenKeys;
    for (auto const& r : best) {
        CHECK(std::isfinite(r.Score));
        CHECK(std::isfinite(r.NegativeLogLikelihood));
        CHECK(std::isfinite(r.ParameterCodeBits));
        CHECK(std::isfinite(r.StructureCodeBits));
        CHECK(r.StructureCodeBits >= 0.0); // log2 of a bucket size >= 1
        auto const expectedScore = static_cast<Operon::Scalar>(r.NegativeLogLikelihood / std::log(2.0) + r.ParameterCodeBits + r.StructureCodeBits);
        CHECK(std::abs(r.Score - expectedScore) < Operon::Scalar{1e-3});
        auto const [it, inserted] = seenKeys.insert(r.CanonicalKey);
        CHECK(inserted); // no CanonicalKey repeats across BestTrees()
    }
    // ascending by Score
    for (std::size_t i = 1; i < best.size(); ++i) { CHECK(best[i - 1].Score <= best[i].Score); }
}

TEST_CASE("GrammarEnumerationAlgorithm - Cube/TenExp productions compute the correct (non-swapped) function", "[enumeration]")
{
    // Regression test: ProcessNonterminal's operand-to-postfix-position mapping must match the
    // interpreter's own convention (the FIRST semantic operand - Pow's base for Cube, Pow's base
    // for TenExp too - is read from the position immediately preceding the op node, not from
    // whichever ProductionOperand happens to be pushed first). A prior version of this code pushed
    // operands in Operands-list order (base then exponent), which silently swapped Cube into
    // computing 3^x instead of x^3, and TenExp into computing x^10 instead of 10^x - both wrong
    // functions, undetectable by the two-nonterminal-operand productions (Sub/Div/Pow/Aq) because
    // those get full symmetric (b0,b1)/(b1,b0) coverage that hides a swap, but fatal for Cube/
    // TenExp's single fixed+nonterminal-operand shape, which has no such symmetric compensation.
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto vars = ds.VariableHashes();
    std::erase(vars, ds.GetVariable("Y").value().Hash);
    vars.resize(1); // one variable is enough to pin down base vs. exponent unambiguously

    using DTable = DispatchTable<Operon::Scalar>;
    DTable dtable;
    Operon::Range const range{ 0, 1 }; // a single row is enough
    auto const x = ds.GetValues(vars.front())[0];

    // Every SimpleExpr this engine builds for a lone variable is exactly WeightPlaceholder(2.0)*x +
    // BiasPlaceholder(1.0) = 2x+1 (see enumeration.cpp's WeightPlaceholder/BiasPlaceholder) -
    // deterministic and known ahead of time, so the expected numeric value is exact, not fitted.
    auto const simpleExprValue = 2.0 * static_cast<double>(x) + 1.0;

    SECTION("Cube: Pow(SimpleExpr, 3) evaluates to SimpleExpr^3, not 3^SimpleExpr")
    {
        Grammar grammar; grammar.SetVariables(vars); grammar.Configure(ToFunctionSet(EnumerationFunction::Cube));
        Operon::RandomGenerator rng(42);
        EnumerationEngine engine(grammar, /*maxComplexity=*/6, rng);
        engine.Build();

        Tree const* cubeTree = nullptr;
        for (std::size_t b = 1; b <= engine.MaxComplexity() && !cubeTree; ++b) {
            for (auto const& t : engine.Bucket(GrammarSymbol::RecurringFactor, b)) {
                if (t.Nodes().back().IsOp<BuiltinOp::Pow>()) { cubeTree = &t; break; }
            }
        }
        REQUIRE(cubeTree != nullptr);

        Interpreter<Operon::Scalar, DTable> interpreter(&dtable, &ds, cubeTree);
        auto estimated = interpreter.Evaluate(cubeTree->GetCoefficients(), range).value();
        auto const expectedCube = std::pow(simpleExprValue, 3.0);
        auto const expectedSwapped = std::pow(3.0, simpleExprValue);
        CHECK(std::abs(static_cast<double>(estimated[0]) - expectedCube) < 1e-2);
        CHECK(std::abs(static_cast<double>(estimated[0]) - expectedSwapped) > 1e-2); // must NOT be the swapped form
    }

    SECTION("TenExp: Pow(10, SimpleExpr) evaluates to 10^SimpleExpr, not SimpleExpr^10")
    {
        Grammar grammar; grammar.SetVariables(vars); grammar.Configure(ToFunctionSet(EnumerationFunction::TenExp));
        Operon::RandomGenerator rng(42);
        EnumerationEngine engine(grammar, /*maxComplexity=*/6, rng);
        engine.Build();

        Tree const* tenExpTree = nullptr;
        for (std::size_t b = 1; b <= engine.MaxComplexity() && !tenExpTree; ++b) {
            for (auto const& t : engine.Bucket(GrammarSymbol::RecurringFactor, b)) {
                if (t.Nodes().back().IsOp<BuiltinOp::Pow>()) { tenExpTree = &t; break; }
            }
        }
        REQUIRE(tenExpTree != nullptr);

        Interpreter<Operon::Scalar, DTable> interpreter(&dtable, &ds, tenExpTree);
        auto estimated = interpreter.Evaluate(tenExpTree->GetCoefficients(), range).value();
        auto const expectedTenExp = std::pow(10.0, simpleExprValue);
        auto const expectedSwapped = std::pow(simpleExprValue, 10.0);
        CHECK(std::abs(static_cast<double>(estimated[0]) - expectedTenExp) < 1e-2);
        CHECK(std::abs(static_cast<double>(estimated[0]) - expectedSwapped) > 1e-2); // must NOT be the swapped form
    }
}

TEST_CASE("Domain analyzer classifies fixed restricted trees and policies", "[enumeration][domain]")
{
    auto negative = Node::Constant(-1.0);
    negative.Optimize = false;
    Tree negativeLog({ negative, Node::Function(static_cast<Hash>(BuiltinOp::Log), 1) });
    negativeLog.UpdateNodes();
    Dataset data({ "x" }, { { 0.F } });
    DomainContext context(data, Range(0, 1));
    CHECK(AnalyzeDomain(negativeLog, context, DomainPolicy::AllRowsFinite) == DomainStatus::Invalid);
    CHECK(AnalyzeDomain(negativeLog, context, DomainPolicy::NoFiniteRows) == DomainStatus::Invalid);
    auto positive = Node::Constant(1.0);
    positive.Optimize = false;
    Tree positiveLog({ positive, Node::Function(static_cast<Hash>(BuiltinOp::Log), 1) });
    positiveLog.UpdateNodes();
    CHECK(AnalyzeDomain(positiveLog, context, DomainPolicy::AllRowsFinite) == DomainStatus::Valid);
    CHECK(AnalyzeDomain(positiveLog, context, DomainPolicy::NoFiniteRows) == DomainStatus::Valid);
}

TEST_CASE("Domain analyzer folds ordinary children under restricted roots", "[enumeration][domain]")
{
    Dataset data({ "x" }, { { -2.F, -3.F } });
    auto const xHash = data.VariableHashes().front();
    Node x(NodeType::Variable);
    x.HashValue = x.CalculatedHashValue = xHash;
    x.Optimize = false;
    Dataset mixed({ "x" }, { { -2.F, 2.F } });
    auto offset = Node::Constant(-1.0);
    offset.Optimize = false;
    Tree logTree({ offset, x, Node::Function(static_cast<Hash>(BuiltinOp::Add), 2),
        Node::Function(static_cast<Hash>(BuiltinOp::Log), 1) });
    logTree.UpdateNodes();
    DomainContext context(data, Range(0, 2));
    CHECK(AnalyzeDomain(logTree, context, DomainPolicy::AllRowsFinite) == DomainStatus::Invalid);

    CHECK(AnalyzeDomain(logTree, DomainContext(mixed, Range(0, 2)), DomainPolicy::AllRowsFinite)
        == DomainStatus::Invalid);
    CHECK(AnalyzeDomain(logTree, DomainContext(mixed, Range(0, 2)), DomainPolicy::NoFiniteRows)
        == DomainStatus::Valid);
}

TEST_CASE("Domain analyzer retains optimized placeholders as unknown", "[enumeration][domain]")
{
    Dataset data({ "x" }, { { -1.F } });
    auto const x = data.VariableHashes().front();
    Node variable(NodeType::Variable);
    variable.HashValue = x;
    variable.Optimize = false;
    auto coefficient = Node::Constant(0.0);
    Tree logTree({ variable, coefficient, Node::Function(static_cast<Hash>(BuiltinOp::Div), 2),
        Node::Function(static_cast<Hash>(BuiltinOp::Log), 1) });
    logTree.UpdateNodes();
    CHECK(AnalyzeDomain(logTree, DomainContext(data, Range(0, 1))) == DomainStatus::Unknown);
}

} // namespace Operon::Test
