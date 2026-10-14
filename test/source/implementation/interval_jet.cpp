// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <eve/wide.hpp>

#include "interval_jet_support.hpp"
#include "operon/core/jet.hpp"
#include "operon/core/tree_diff.hpp"
#include "operon/interpreter/backend/eve/jet_math.hpp"
#include "operon/interpreter/backend/interval_jet_math.hpp"
#include "operon/interpreter/interval_evaluator.hpp"
#include "operon/interpreter/interval_jet_evaluator.hpp"

namespace Operon::Test {
namespace {
    using namespace JetSupport;
    using Status = Operon::IntervalJetStatus;
    using JE = Operon::IntervalJetEvaluator<S>;

    constexpr auto Inf = std::numeric_limits<S>::infinity();

    auto MakeEvaluator(Operon::Tree const& tree, DomainMap const& domains, std::vector<Operon::Hash> const& vars) -> JE
    {
        return JE { &tree, domains, vars };
    }

    // Random box: centre in [-3, 3], width from a mix of point / narrow / medium / wide.
    auto RandomDomains(std::mt19937_64& rng, Operon::Dataset const& ds, std::size_t nvars) -> DomainMap
    {
        constexpr std::array<double, 5> widths { 0.0, 0.05, 0.5, 2.0, 6.0 };
        DomainMap domains;
        for (std::size_t i = 1; i <= nvars; ++i) {
            auto const centre = std::uniform_real_distribution<double> { -3.0, 3.0 }(rng);
            auto const half = 0.5 * widths[std::uniform_int_distribution<std::size_t> { 0, widths.size() - 1 }(rng)];
            auto const lo = static_cast<S>(centre - half);
            auto const hi = std::max(lo, static_cast<S>(centre + half));
            domains.insert_or_assign(VarHash(ds, i), std::pair { lo, hi });
        }
        return domains;
    }

    struct SoundnessResult {
        Containment Value;
        Containment Derivative;
        std::size_t Evaluated {};
        std::size_t Certain {};
    };

    // Evaluates `expr` over `boxes` random boxes with the first `ndiff` variables differentiated and checks every
    // certain enclosure against `samples` reference points of the box.
    // T is the interval scalar of the evaluator under test: float is the production type (only outward rounded to
    // within a few ulp by pappus), double checks the formulas themselves with a tight slack.
    template <typename T = S>
    auto Soundness(std::string const& expr, std::size_t nvars, std::size_t ndiff, std::size_t boxes,
        std::size_t samples, bool randomCoefficients = false, std::uint64_t seed = 20260101) -> SoundnessResult
    {
        auto const ds = MakeDataset(nvars);
        auto const tree = InfixParser::ParseOrThrow(expr, ds);
        std::vector<Operon::Hash> vars;
        for (std::size_t i = 1; i <= ndiff; ++i) {
            vars.push_back(VarHash(ds, i));
        }
        std::mt19937_64 rng { seed };
        SoundnessResult out;
        out.Value.Slack = out.Derivative.Slack = std::same_as<T, double> ? 1e-9 : 1e-4;
        for (std::size_t b = 0; b < boxes; ++b) {
            auto const domains = RandomDomains(rng, ds, nvars);
            auto coeff = tree.GetCoefficients();
            if (randomCoefficients) {
                for (auto& c : coeff) {
                    c = static_cast<S>(std::uniform_real_distribution<double> { -2.0, 2.0 }(rng));
                }
            }
            auto const jet = Operon::IntervalJetEvaluator<T> { &tree, domains, vars }(coeff);
            REQUIRE(jet.has_value());
            ++out.Evaluated;
            auto const set = ToBoundSet(*jet);
            for (std::size_t k = 0; k < ndiff; ++k) {
                out.Certain += jet->Status[k] == Status::Certain ? 1 : 0;
            }
            SampleCheck(tree, coeff, vars, domains, set, rng, samples, out.Value, out.Derivative);
        }
        return out;
    }

    void ExpectSound(SoundnessResult const& r)
    {
        INFO("worst derivative excess " << r.Derivative.WorstExcess << ", value excess " << r.Value.WorstExcess);
        CHECK(r.Derivative.Violations == 0);
        CHECK(r.Value.Violations == 0);
        CHECK(r.Derivative.Checked > 0);
        CHECK(r.Certain > 0);
    }

    auto Relative(S a, S b) -> bool
    {
        return std::abs(a - b) <= 1e-4F * std::max({ S { 1 }, std::abs(a), std::abs(b) });
    }

    // Evaluates one expression over the given box (variables X1, X2, ... get consecutive entries).
    auto Eval(std::string const& expr, std::vector<std::pair<S, S>> const& box, std::size_t ndiff)
        -> tl::expected<IntervalJetBounds<S>, std::string>
    {
        auto const ds = MakeDataset(box.size());
        auto const tree = InfixParser::ParseOrThrow(expr, ds);
        DomainMap domains;
        std::vector<Operon::Hash> vars;
        for (std::size_t i = 0; i < box.size(); ++i) {
            domains.insert_or_assign(VarHash(ds, i + 1), box[i]);
            if (i < ndiff) {
                vars.push_back(VarHash(ds, i + 1));
            }
        }
        return JE { &tree, domains, vars }(tree.GetCoefficients());
    }
} // namespace

TEST_CASE("JetMath sqr hook: scalar, SIMD lanes and intervals", "[interval-jet]")
{
    CHECK(JetMath<double>::sqr(-3.0) == 9.0);
    CHECK(JetMath<float>::sqr(1.5F) == 2.25F);

    using W = eve::wide<float>;
    W const w { [](auto i, auto) { return static_cast<float>(i) - 2.0F; } };
    auto const sq = JetMath<W>::sqr(w);
    for (std::size_t i = 0; i < W::size(); ++i) {
        CHECK(sq.get(i) == w.get(i) * w.get(i));
    }

    // dependency-aware: x * x over [-1, 1] is [-1, 1], sqr is [0, 1]
    auto const x = Interval { -1.0F, 1.0F };
    CHECK((x * x).inf() == -1.0F);
    CHECK(JetMath<Interval>::sqr(x).inf() == 0.0F);
    CHECK(JetMath<Interval>::sqr(x).sup() == 1.0F);
}

TEST_CASE("Jet over intervals: derivative formulas use the tight square", "[interval-jet]")
{
    using JI = Operon::Jet<Interval, 1>;

    SECTION("atan on [-1, 1]: derivative about [0.5, 1], not unbounded")
    {
        auto const r = Operon::atan(JI { Interval { -1.0F, 1.0F }, 0 });
        CHECK(r.v[0].inf() >= 0.4999F);
        CHECK(r.v[0].sup() <= 1.0001F);
        CHECK(r.v[0].inf() <= 0.5F);
        CHECK(r.v[0].sup() >= 1.0F);
    }
    SECTION("tanh: 1 - tanh^2 stays within [0, 1]")
    {
        auto const r = Operon::tanh(JI { Interval { -1.0F, 1.0F }, 0 });
        CHECK(r.v[0].sup() <= 1.0001F);
        CHECK(r.v[0].inf() >= 0.0F);
    }
    SECTION("tan: 1 + tan^2 >= 1")
    {
        auto const r = Operon::tan(JI { Interval { -1.0F, 1.0F }, 0 });
        CHECK(r.v[0].inf() >= 0.9999F);
    }
    SECTION("asin/acos: 1 / sqrt(1 - x^2) on [-0.5, 0.5]")
    {
        auto const s = Operon::asin(JI { Interval { -0.5F, 0.5F }, 0 });
        CHECK(s.v[0].inf() >= 0.9999F);
        CHECK(s.v[0].sup() <= 1.1548F); // 1 / sqrt(0.75)
        auto const c = Operon::acos(JI { Interval { -0.5F, 0.5F }, 0 });
        CHECK(c.v[0].sup() <= -0.9999F);
        CHECK(c.v[0].inf() >= -1.1548F);
    }
    SECTION("integer-exponent pow(Jet, T) uses the integer power: (x^2)' = 2x over [0, 1] is [0, 2]")
    {
        auto const r = Operon::pow(JI { Interval { 0.0F, 1.0F }, 0 }, Interval { 2.0F });
        CHECK(r.a.inf() == 0.0F);
        CHECK(r.a.sup() == 1.0F);
        CHECK(r.v[0].inf() == 0.0F);
        CHECK(r.v[0].sup() == 2.0F);
        auto const neg = Operon::pow(JI { Interval { -1.0F, 1.0F }, 0 }, Interval { 3.0F });
        CHECK(neg.v[0].inf() == 0.0F);
        CHECK(neg.v[0].sup() == 3.0F);
    }
    SECTION("sgn is a sound hull")
    {
        using M = JetMath<Interval>;
        CHECK(M::sgn(Interval { 1.0F, 2.0F }) == Interval { 1.0F });
        CHECK(M::sgn(Interval { -2.0F, -1.0F }) == Interval { -1.0F });
        CHECK(M::sgn(Interval { -1.0F, 2.0F }) == Interval { -1.0F, 1.0F });
        CHECK(M::sgn(Interval { 0.0F, 2.0F }) == Interval { 0.0F, 1.0F });
    }
}

TEST_CASE("IntervalJetEvaluator: the two enclosure examples where jet and symbolic dag differ", "[interval-jet]")
{
    SECTION("pow(x, 2) on [0, 1]: jet gives [0, 2], the symbolic dag is unbounded at 0")
    {
        auto const jet = Eval("X1 ^ 2", { { 0.0F, 1.0F } }, 1);
        REQUIRE(jet);
        REQUIRE(jet->Status[0] == Status::Certain);
        CHECK(jet->Derivative[0].inf() == 0.0F);
        CHECK(jet->Derivative[0].sup() == 2.0F);
        CHECK(jet->Value.inf() == 0.0F);
        CHECK(jet->Value.sup() == 1.0F);

        auto const ds = MakeDataset(1);
        auto const tree = InfixParser::ParseOrThrow("X1 ^ 2", ds);
        auto const coeff = tree.GetCoefficients();
        auto const dag = BuildVariableGradientDag(tree, coeff);
        REQUIRE(dag.Variables.size() == 1);
        DomainMap domains;
        domains.insert_or_assign(VarHash(ds, 1), std::pair { 0.0F, 1.0F });
        IntervalEvaluator<S> ie(&tree, domains);
        ie.SetNodes(dag.Nodes);
        Interval symbolic;
        REQUIRE(ie.EvaluateAll(coeff, [&](Operon::Span<Interval const> v) { symbolic = v[dag.Roots[0]]; }));
        CHECK(!std::isfinite(symbolic.sup()));
    }

    SECTION("atan(x) on [-1, 1]: derivative about [0.5, 1]")
    {
        auto const jet = Eval("atan(X1)", { { -1.0F, 1.0F } }, 1);
        REQUIRE(jet);
        REQUIRE(jet->Status[0] == Status::Certain);
        CHECK(jet->Derivative[0].inf() >= 0.4999F);
        CHECK(jet->Derivative[0].inf() <= 0.5F);
        CHECK(jet->Derivative[0].sup() >= 1.0F);
        CHECK(jet->Derivative[0].sup() <= 1.0001F);
    }
}

TEST_CASE("IntervalJetEvaluator: derivative enclosures contain sampled true derivatives (unary ops)", "[interval-jet]")
{
    for (auto const* expr :
        { "sqrt(X1)", "cbrt(X1)", "exp(X1)", "log(X1)", "log1p(X1)", "logabs(X1)", "sin(X1)", "cos(X1)", "tan(X1)",
            "asin(X1)", "acos(X1)", "atan(X1)", "sinh(X1)", "cosh(X1)", "tanh(X1)", "square(X1)", "-X1", "1 / X1" }) {
        DYNAMIC_SECTION(expr)
        {
            ExpectSound(Soundness(expr, 1, 1, 300, 24));
            ExpectSound(Soundness<double>(expr, 1, 1, 300, 24));
        }
    }
}

TEST_CASE("IntervalJetEvaluator: derivative enclosures contain sampled true derivatives (binary ops and chains)",
    "[interval-jet]")
{
    for (auto const* expr : { "X1 + X2", "X1 - X2", "X1 * X2", "X1 / X2", "X1 ^ X2", "X1 ^ 2", "X1 ^ 3", "X1 ^ 2.5",
             "X1 ^ -1", "2 ^ X1", "X2 ^ 2 + X1", "X1 * X1 * X2", "sin(X1 * X2) + exp(X1) / (X2 ^ 2 + 1)",
             "log(X1 * X1 + X2 * X2 + 1) * atan(X1 / (X2 * X2 + 0.5))", "sqrt(X1 * X1 + X2 * X2) - tanh(X1 * X2)",
             "square(X1 - X2) / (cosh(X1) + 1)", "asin(X1 / (1 + square(X2))) + acos(tanh(X1))",
             "1 / (1 + X1 * X1) - cbrt(X2) * log1p(X1 * X1)" }) {
        DYNAMIC_SECTION(expr)
        {
            ExpectSound(Soundness(expr, 2, 2, 300, 24));
            ExpectSound(Soundness(expr, 2, 2, 150, 24, /*randomCoefficients=*/true, 99));
            ExpectSound(Soundness<double>(expr, 2, 2, 300, 24));
            ExpectSound(Soundness<double>(expr, 2, 2, 150, 24, /*randomCoefficients=*/true, 99));
        }
    }
}

TEST_CASE("IntervalJetEvaluator: 3 and 4 differentiated variables, undifferentiated variables are constants",
    "[interval-jet]")
{
    constexpr auto expr = "X1 * X2 + sin(X3) / (1 + X4 * X4) - X1 * X4 + exp(X2 * X3)";
    SECTION("K = 4")
    {
        ExpectSound(Soundness(expr, 4, 4, 300, 24));
        ExpectSound(Soundness<double>(expr, 4, 4, 300, 24));
    }
    SECTION("K = 3 (X4 is a constant over its box)")
    {
        ExpectSound(Soundness(expr, 4, 3, 300, 24));
        ExpectSound(Soundness<double>(expr, 4, 3, 300, 24));
    }
    SECTION("K = 2")
    {
        ExpectSound(Soundness(expr, 4, 2, 300, 24));
        ExpectSound(Soundness<double>(expr, 4, 2, 300, 24));
    }
}

TEST_CASE("IntervalJetEvaluator: directions are independent of K and of each other", "[interval-jet]")
{
    constexpr auto expr = "sin(X1 * X2) + exp(X1) / (X2 ^ 2 + 1) - X3 * X1 / (X2 * X2 + 2)";
    auto const ds = MakeDataset(3);
    auto const tree = InfixParser::ParseOrThrow(expr, ds);
    DomainMap domains;
    domains.insert_or_assign(VarHash(ds, 1), std::pair { -0.5F, 1.0F });
    domains.insert_or_assign(VarHash(ds, 2), std::pair { 0.25F, 2.0F });
    domains.insert_or_assign(VarHash(ds, 3), std::pair { -1.0F, 1.0F });
    auto const coeff = tree.GetCoefficients();

    auto const all = JE { &tree, domains, std::vector { VarHash(ds, 1), VarHash(ds, 2), VarHash(ds, 3) } }(coeff);
    REQUIRE(all);
    for (std::size_t k = 0; k < 3; ++k) {
        auto const single = JE { &tree, domains, std::vector { VarHash(ds, k + 1) } }(coeff);
        REQUIRE(single);
        CHECK(single->Value == all->Value);
        CHECK(single->Derivative[0] == all->Derivative[k]);
        CHECK(single->Status[0] == all->Status[k]);
    }
    // permuting the variables permutes the directions
    auto const swapped = JE { &tree, domains, std::vector { VarHash(ds, 3), VarHash(ds, 1) } }(coeff);
    REQUIRE(swapped);
    CHECK(swapped->Derivative[0] == all->Derivative[2]);
    CHECK(swapped->Derivative[1] == all->Derivative[0]);
}

TEST_CASE("IntervalJetEvaluator: a failing partial derivative does not leak into another direction", "[interval-jet]")
{
    // d sqrt(x)/dx = 1 / (2 sqrt(x)) is undefined at the point domain {0}: X2's derivative is a domain error,
    // while sqrt(0) itself and d/dX1 are fine.
    auto const r = Eval("X1 + sqrt(X2)", { { 1.0F, 2.0F }, { 0.0F, 0.0F } }, 2);
    REQUIRE(r);
    CHECK(!r->Value.is_empty());
    CHECK(r->Status[0] == Status::Certain);
    CHECK(r->Derivative[0] == Interval { 1.0F });
    CHECK(r->Status[1] == Status::Failed);
    CHECK(r->Derivative[1].is_empty());
}

TEST_CASE("IntervalJetEvaluator: weights and coefficients follow the IntervalEvaluator convention", "[interval-jet]")
{
    auto const ds = MakeDataset(2);
    auto const x1 = VarHash(ds, 1);
    auto const x2 = VarHash(ds, 2);

    SECTION("weighted variable: d(w X)/dX = w exactly")
    {
        auto tree = InfixParser::ParseOrThrow("X1", ds);
        tree.Nodes()[0].Value = S { -2.5 };
        tree.Nodes()[0].Optimize = false;
        DomainMap domains;
        domains.insert_or_assign(x1, std::pair { 1.0F, 3.0F });
        auto const r = MakeEvaluator(tree, domains, { x1 })(tree.GetCoefficients());
        REQUIRE(r);
        CHECK(r->Derivative[0] == Interval { -2.5F });
        CHECK(r->Value == Interval { -7.5F, -2.5F });
    }

    // sin(w2 * X2) * (w1 * X1), with function-node weights on both sin and Mul, all optimizable.
    auto const build = [&] {
        Node v2 = Node(NodeType::Variable, x2);
        v2.Value = S { -0.5 };
        Node sn = Node::Function(Operon::Hash(BuiltinOp::Sin), 1);
        sn.Value = S { 2.5 };
        sn.Optimize = true;
        Node v1 = Node(NodeType::Variable, x1);
        v1.Value = S { 1.5 };
        Node mul = Node::Function(Operon::Hash(BuiltinOp::Mul), 2);
        mul.Value = S { 0.75 };
        mul.Optimize = true;
        return Tree({ v2, sn, v1, mul }).UpdateNodes();
    };

    SECTION("coefficients are consumed in Optimize order and contain the sampled derivative")
    {
        auto const tree = build();
        auto coeff = tree.GetCoefficients();
        REQUIRE(coeff.size() == 4);
        coeff = { S { -0.5 }, S { 2.5 }, S { 1.5 }, S { 0.75 } };
        DomainMap domains;
        domains.insert_or_assign(x1, std::pair { -1.0F, 2.0F });
        domains.insert_or_assign(x2, std::pair { -2.0F, 1.0F });
        std::vector const vars { x1, x2 };
        auto const jet = MakeEvaluator(tree, domains, vars)(coeff);
        REQUIRE(jet);
        REQUIRE(jet->Status[0] == Status::Certain);
        REQUIRE(jet->Status[1] == Status::Certain);

        // the symbolic dag is an independent oracle for the weight convention
        auto const dag = BuildVariableGradientDag(tree, coeff, vars);
        IntervalEvaluator<S> ie(&tree, domains);
        ie.SetNodes(dag.Nodes);
        BoundSet dagSet;
        REQUIRE(ie.EvaluateAll(coeff, [&](Operon::Span<Interval const> v) {
            dagSet.Value = ToBnd(v[dag.OriginalSize - 1]);
            for (std::size_t k = 0; k < vars.size(); ++k) {
                auto const at
                    = static_cast<std::size_t>(std::ranges::find(dag.Variables, vars[k]) - dag.Variables.begin());
                dagSet.Derivative[k] = ToBnd(v[dag.Roots[at]]);
            }
        }));

        std::mt19937_64 rng { 5 };
        Containment jv { 1e-4 };
        Containment jd { 1e-4 };
        Containment dv { 1e-4 };
        Containment dd { 1e-4 };
        SampleCheck(tree, coeff, vars, domains, ToBoundSet(*jet), rng, 200, jv, jd);
        SampleCheck(tree, coeff, vars, domains, dagSet, rng, 200, dv, dd);
        CHECK(jd.Checked > 0);
        CHECK(jd.Violations == 0);
        CHECK(jv.Violations == 0);
        CHECK(dd.Violations == 0); // reference == dag convention, hence the reference is a valid oracle here
        CHECK(dv.Violations == 0);
        // value is identical to the dag's identity root
        CHECK(ToBnd(jet->Value) == *dagSet.Value);
    }

    SECTION("changing a coefficient changes the result")
    {
        auto const tree = build();
        DomainMap domains;
        domains.insert_or_assign(x1, std::pair { 1.0F, 2.0F });
        domains.insert_or_assign(x2, std::pair { 0.5F, 1.0F });
        auto const a = MakeEvaluator(tree, domains, { x1 })(Operon::Vector<S> { -0.5F, 2.5F, 1.5F, 0.75F });
        auto const b = MakeEvaluator(tree, domains, { x1 })(Operon::Vector<S> { -0.5F, 2.5F, 1.5F, 1.5F });
        REQUIRE(a);
        REQUIRE(b);
        // only the Mul weight differs: the derivative scales by exactly 2
        CHECK(Relative(b->Derivative[0].inf(), 2 * a->Derivative[0].inf()));
        CHECK(Relative(b->Derivative[0].sup(), 2 * a->Derivative[0].sup()));
    }

    SECTION("zero function weight zeroes the derivative even through an unbounded interior")
    {
        Node v = Node(NodeType::Variable, x1);
        v.Value = S { 1 };
        Node tn = Node::Function(Operon::Hash(BuiltinOp::Tan), 1);
        tn.Value = S { 0 };
        tn.Optimize = false;
        auto const tree = Tree({ v, tn }).UpdateNodes();
        DomainMap domains;
        domains.insert_or_assign(x1, std::pair { 1.0F, 2.0F }); // contains pi/2
        auto const r = MakeEvaluator(tree, domains, { x1 })(tree.GetCoefficients());
        REQUIRE(r);
        CHECK(r->Status[0] == Status::Certain);
        CHECK(r->Derivative[0] == Interval { 0.0F });
        CHECK(r->Value == Interval { 0.0F });
    }
}

TEST_CASE("IntervalJetEvaluator: Ref nodes alias their target's value and tangent", "[interval-jet]")
{
    auto const ds = MakeDataset(2);
    auto const x1 = VarHash(ds, 1);
    auto const x2 = VarHash(ds, 2);
    // sin(X1) * sin(X1) + X2, the second sin(X1) is a Ref
    Node v1 = Node(NodeType::Variable, x1);
    Node sn = Node::Function(Operon::Hash(BuiltinOp::Sin), 1);
    Node ref = Node::Ref(1);
    Node mul = Node::Function(Operon::Hash(BuiltinOp::Mul), 2);
    Node v2 = Node(NodeType::Variable, x2);
    Node add = Node::Function(Operon::Hash(BuiltinOp::Add), 2);
    auto const tree = Tree({ v1, sn, ref, mul, v2, add }).UpdateNodes();
    REQUIRE(tree.Length() == 6);

    DomainMap domains;
    domains.insert_or_assign(x1, std::pair { 0.0F, 1.0F });
    domains.insert_or_assign(x2, std::pair { -1.0F, 1.0F });
    std::vector const vars { x1, x2 };
    auto const coeff = tree.GetCoefficients();
    auto const jet = MakeEvaluator(tree, domains, vars)(coeff);
    REQUIRE(jet);
    REQUIRE(jet->Status[0] == Status::Certain);
    REQUIRE(jet->Status[1] == Status::Certain);
    CHECK(jet->Derivative[1] == Interval { 1.0F });

    std::mt19937_64 rng { 11 };
    Containment v { 1e-4 };
    Containment d { 1e-4 };
    SampleCheck(tree, coeff, vars, domains, ToBoundSet(*jet), rng, 200, v, d);
    CHECK(d.Checked > 0);
    CHECK(d.Violations == 0);
    CHECK(v.Violations == 0);
    // 2 sin(x) cos(x) over [0, 1] is in [0, sin(2) = 0.909]
    CHECK(jet->Derivative[0].sup() >= 0.9092F);
    CHECK(jet->Derivative[0].inf() <= 0.0F);
}

TEST_CASE(
    "IntervalJetEvaluator: unsupported operations are uncertain exactly where the symbolic dag is", "[interval-jet]")
{
    // X1, X2 differentiated; X3 is a constant over its box.
    auto const ds = MakeDataset(3);
    auto const V = [&](std::size_t i) { return Node(NodeType::Variable, VarHash(ds, i)); };
    auto const F = [](BuiltinOp op, std::uint16_t arity) { return Node::Function(Operon::Hash(op), arity); };
    auto const C = [](double v) { return Node::Constant(v); };
    auto const Make = [](std::vector<Node> nodes) { return Tree(std::move(nodes)).UpdateNodes(); };

    // Binary nodes list the far operand first: op(a, b) is emitted as [b, a, op].
    std::vector<std::pair<std::string, Tree>> cases;
    for (auto const* expr : { "abs(X1) + X2", "sqrtabs(X1 * X2)", "aq(X1, X2)", "powabs(X1, X2)",
             "sin(abs(X3)) * X2 + X1", "abs(X1 * X3) / (X2 * X2 + 1)", "exp(X1) + abs(X3)", "sin(X1) + cos(X2)",
             "log(X1 + X2) * X3", "X1 ^ X2 + abs(X2)", "sqrtabs(X3) + X2 * X1", "X1 * X2 / (1 + X1)" }) {
        cases.emplace_back(expr, InfixParser::ParseOrThrow(expr, ds));
    }
    cases.emplace_back("fmin(X1, X2) + 1", Make({ C(1), V(2), V(1), F(BuiltinOp::Fmin, 2), F(BuiltinOp::Add, 2) }));
    cases.emplace_back("fmax(X1, 3) * X2", Make({ V(2), C(3), V(1), F(BuiltinOp::Fmax, 2), F(BuiltinOp::Mul, 2) }));
    cases.emplace_back("floor(X1) + X2", Make({ V(2), V(1), F(BuiltinOp::Floor, 1), F(BuiltinOp::Add, 2) }));
    cases.emplace_back("X1 + ceil(X2)", Make({ V(1), V(2), F(BuiltinOp::Ceil, 1), F(BuiltinOp::Add, 2) }));
    cases.emplace_back("X1 + floor(X3)", Make({ V(1), V(3), F(BuiltinOp::Floor, 1), F(BuiltinOp::Add, 2) }));
    cases.emplace_back("X2 + fmin(X3, 2)", Make({ V(2), C(2), V(3), F(BuiltinOp::Fmin, 2), F(BuiltinOp::Add, 2) }));

    for (auto const& [name, tree] : cases) {
        DYNAMIC_SECTION(name)
        {
            std::vector const vars { VarHash(ds, 1), VarHash(ds, 2) };
            DomainMap domains;
            for (std::size_t i = 1; i <= 3; ++i) {
                domains.insert_or_assign(VarHash(ds, i), std::pair { 1.0F, 2.0F });
            }
            auto const coeff = tree.GetCoefficients();
            auto const jet = MakeEvaluator(tree, domains, vars)(coeff);
            REQUIRE(jet);
            auto const dag = BuildVariableGradientDag(tree, coeff, vars);
            for (std::size_t k = 0; k < vars.size(); ++k) {
                auto const at = std::ranges::find(dag.Variables, vars[k]);
                bool const dagCertain
                    = at == dag.Variables.end() || dag.Certain[static_cast<std::size_t>(at - dag.Variables.begin())];
                INFO("direction " << k);
                CHECK((jet->Status[k] == Status::Uncertain) == !dagCertain);
            }
        }
    }

    SECTION("concrete cases")
    {
        auto const r = Eval("abs(X1) + X2", { { 1.0F, 2.0F }, { 1.0F, 2.0F } }, 2);
        REQUIRE(r);
        CHECK(r->Status[0] == Status::Uncertain);
        CHECK(r->Status[1] == Status::Certain);
        CHECK(r->Derivative[1] == Interval { 1.0F });
        CHECK(r->Value == Interval { 2.0F, 4.0F }); // the primal of an unsupported op is still computed

        auto const both = Eval("sqrtabs(X1 * X2)", { { 1.0F, 2.0F }, { 1.0F, 2.0F } }, 2);
        REQUIRE(both);
        CHECK(both->Status[0] == Status::Uncertain);
        CHECK(both->Status[1] == Status::Uncertain);

        // an unsupported op on an independent branch does not taint the variable
        auto const clean = Eval("sin(X1) + abs(X2)", { { 1.0F, 2.0F }, { 1.0F, 2.0F } }, 1);
        REQUIRE(clean);
        CHECK(clean->Status[0] == Status::Certain);
    }
}

TEST_CASE("IntervalJetEvaluator: structural errors and domain errors are reported distinctly", "[interval-jet]")
{
    SECTION("domain error: log of a negative box is Failed (not Uncertain) and the value is empty")
    {
        auto const r = Eval("log(X1)", { { -2.0F, -1.0F } }, 1);
        REQUIRE(r);
        CHECK(r->Value.is_empty());
        CHECK(r->Status[0] == Status::Failed);
    }
    SECTION("domain error in an unsupported-op-free branch: asin outside [-1, 1]")
    {
        auto const r = Eval("asin(X1) + X2", { { 2.0F, 3.0F }, { 0.0F, 1.0F } }, 2);
        REQUIRE(r);
        CHECK(r->Status[0] == Status::Failed);
        CHECK(r->Status[1] == Status::Failed); // the whole value is empty
    }
    SECTION("NaN coefficient")
    {
        auto const ds = MakeDataset(1);
        auto const tree = InfixParser::ParseOrThrow("X1 * 2", ds);
        DomainMap domains;
        domains.insert_or_assign(VarHash(ds, 1), std::pair { 0.0F, 1.0F });
        auto coeff = tree.GetCoefficients();
        coeff.back() = std::numeric_limits<S>::quiet_NaN();
        auto const r = MakeEvaluator(tree, domains, { VarHash(ds, 1) })(coeff);
        REQUIRE(r);
        CHECK(r->Status[0] == Status::Failed);
    }
    SECTION("a variable absent from the tree has an exactly zero derivative")
    {
        auto const r = Eval("X1 * 2", { { 0.0F, 1.0F }, { 0.0F, 1.0F } }, 2);
        REQUIRE(r);
        CHECK(r->Status[1] == Status::Certain);
        CHECK(r->Derivative[1] == Interval { 0.0F });
    }
    SECTION("user-defined operations are not supported by the jet path")
    {
        auto const ds = MakeDataset(1);
        Node v = Node(NodeType::Variable, VarHash(ds, 1));
        auto const tree = Tree({ v, Node::Function(Operon::Hash { 9999 }, 1) }).UpdateNodes();
        DomainMap domains;
        domains.insert_or_assign(VarHash(ds, 1), std::pair { 0.0F, 1.0F });
        auto const r = MakeEvaluator(tree, domains, { VarHash(ds, 1) })(tree.GetCoefficients());
        REQUIRE(!r);
        CHECK(r.error().find("not supported") != std::string::npos);
    }
    SECTION("n-ary Div is not supported (the symbolic dag does not differentiate it either)")
    {
        auto const ds = MakeDataset(1);
        Node v = Node(NodeType::Variable, VarHash(ds, 1));
        Node c = Node::Constant(2.0);
        Node div = Node::Function(Operon::Hash(BuiltinOp::Div), 3);
        auto const tree = Tree({ c, c, v, div }).UpdateNodes();
        DomainMap domains;
        domains.insert_or_assign(VarHash(ds, 1), std::pair { 1.0F, 2.0F });
        auto const r = MakeEvaluator(tree, domains, { VarHash(ds, 1) })(tree.GetCoefficients());
        REQUIRE(!r);
    }
    SECTION("missing domain bound")
    {
        auto const ds = MakeDataset(2);
        auto const tree = InfixParser::ParseOrThrow("X1 + X2", ds);
        DomainMap domains;
        domains.insert_or_assign(VarHash(ds, 1), std::pair { 0.0F, 1.0F });
        auto const r = MakeEvaluator(tree, domains, { VarHash(ds, 1) })(tree.GetCoefficients());
        REQUIRE(!r);
    }
    SECTION("unsupported number of differentiated variables")
    {
        auto const ds = MakeDataset(5);
        auto const tree = InfixParser::ParseOrThrow("X1 + X2 + X3 + X4 + X5", ds);
        DomainMap domains;
        std::vector<Operon::Hash> vars;
        for (std::size_t i = 1; i <= 5; ++i) {
            domains.insert_or_assign(VarHash(ds, i), std::pair { 0.0F, 1.0F });
            vars.push_back(VarHash(ds, i));
        }
        REQUIRE(!MakeEvaluator(tree, domains, vars)(tree.GetCoefficients()));
        REQUIRE(!MakeEvaluator(tree, domains, {})(tree.GetCoefficients()));
        vars.pop_back();
        auto const four = MakeEvaluator(tree, domains, vars)(tree.GetCoefficients());
        REQUIRE(four);
        for (std::size_t k = 0; k < 4; ++k) {
            CHECK(four->Derivative[k] == Interval { 1.0F });
        }
    }
}

TEST_CASE("IntervalJetEvaluator: x^1 and x^0 differentiate exactly even over a zero base", "[interval-jet]")
{
    auto const one = Eval("X1 ^ 1", { { 0.0F, 0.0F } }, 1);
    REQUIRE(one);
    CHECK(one->Status[0] == Status::Certain);
    CHECK(one->Derivative[0] == Interval { 1.0F });

    auto const zero = Eval("X1 ^ 0", { { 1.0F, 2.0F } }, 1);
    REQUIRE(zero);
    CHECK(zero->Status[0] == Status::Certain);
    CHECK(zero->Derivative[0] == Interval { 0.0F });
}

TEST_CASE("IntervalJetEvaluator: the identity bound equals the IntervalEvaluator bound", "[interval-jet]")
{
    constexpr auto expr = "sin(X1 * X2) + exp(X1) / (X2 ^ 2 + 1) - cbrt(X1) * log(X2 + 3) + atan(X1 / (X2 * X2 + 0.5))";
    auto const ds = MakeDataset(2);
    auto const tree = InfixParser::ParseOrThrow(expr, ds);
    std::mt19937_64 rng { 3 };
    auto const coeff = tree.GetCoefficients();
    for (int i = 0; i < 100; ++i) {
        auto const domains = RandomDomains(rng, ds, 2);
        auto const jet = MakeEvaluator(tree, domains, { VarHash(ds, 1) })(coeff);
        IntervalEvaluator<S> ie(&tree, domains);
        auto const plain = ie(coeff);
        REQUIRE(jet);
        REQUIRE(plain);
        CHECK(jet->Value == *plain);
    }
}

} // namespace Operon::Test
