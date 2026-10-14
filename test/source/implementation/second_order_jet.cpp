// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "interval_jet_support.hpp"
#include "operon/core/jet.hpp"
#include "operon/core/node.hpp"
#include "operon/core/pset.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/interpreter/backend/interval_jet_math.hpp"
#include "operon/interpreter/interval_second_order_jet_evaluator.hpp"
#include "operon/operators/creator.hpp"
#include <catch2/catch_test_macros.hpp>
#include <fmt/ranges.h>
#include <pappus/pappus.hpp>

namespace Operon::Test {
namespace {
    using namespace JetSupport;

    using SJ1 = Operon::Jet<double, 1, Operon::JetOrder::SecondDiagonal>;
    using J1 = Operon::Jet<double, 1>;
    using Iv = pappus::interval<double>;
    using SIJ = Operon::Jet<Iv, 1, Operon::JetOrder::SecondDiagonal>;
    using Status = Operon::IntervalJetStatus;
    using SOE = Operon::IntervalSecondOrderJetEvaluator<S>;

    constexpr auto Inf = std::numeric_limits<S>::infinity();

    // Scalar counterparts of the Operon-specific functions, so that one generic callable below serves as the jet
    // function and as its scalar reference (see jet.cpp).
    namespace Ref {
        auto square(double x) -> double { return x * x; }
        auto logabs(double x) -> double { return std::log(std::abs(x)); }
        auto sqrtabs(double x) -> double { return std::sqrt(std::abs(x)); }
        auto aq(double x, double y) -> double { return x / std::sqrt(1.0 + y * y); }
        auto powabs(double x, double y) -> double { return std::pow(std::abs(x), y); }
    } // namespace Ref

    template <typename X, typename = void> struct ScalarOfImpl {
        using type = X;
    };
    template <typename X> struct ScalarOfImpl<X, std::void_t<typename X::Scalar>> {
        using type = typename X::Scalar;
    };
    // The plain scalar under a jet (the double itself for a double): constants inside a generic function are built
    // as ScalarOf<X>(c), which also works for interval scalars whose constructor is explicit.
    template <typename X> using ScalarOf = typename ScalarOfImpl<std::decay_t<X>>::type;

// One generic function of the argument `x`: it runs on doubles, on Jet<double, 1> (the first
// derivative), on Jet<double, 1, JetOrder::SecondDiagonal> and on Jet<interval<double>, 1, JetOrder::SecondDiagonal>.
// `C(c)` is a constant of the argument's scalar type.
#define SOJ_FN(expr)                                                                                                   \
    [](auto const& x) {                                                                                                \
        using ScalarType = ScalarOf<decltype(x)>;                                                                      \
        [[maybe_unused]] auto const C = [](double c) { return ScalarType(c); };                                        \
        using std::abs;                                                                                                \
        using std::acos;                                                                                               \
        using std::asin;                                                                                               \
        using std::atan;                                                                                               \
        using std::cbrt;                                                                                               \
        using std::cos;                                                                                                \
        using std::cosh;                                                                                               \
        using std::exp;                                                                                                \
        using std::fmax;                                                                                               \
        using std::fmin;                                                                                               \
        using std::log;                                                                                                \
        using std::log1p;                                                                                              \
        using std::pow;                                                                                                \
        using std::sin;                                                                                                \
        using std::sinh;                                                                                               \
        using std::sqrt;                                                                                               \
        using std::tan;                                                                                                \
        using std::tanh;                                                                                               \
        using namespace Ref;                                                                                           \
        return expr;                                                                                                   \
    }

    auto Near(double actual, double expected, double rtol) -> bool
    {
        if (std::isnan(actual) || std::isnan(expected)) {
            return std::isnan(actual) && std::isnan(expected);
        }
        if (std::isinf(actual) || std::isinf(expected)) {
            return actual == expected;
        }
        return std::abs(actual - expected) <= rtol * std::max(1.0, std::abs(expected));
    }

    // value, first and second derivative of `f` at x against: the scalar function, the first-order Jet (v) and
    // central differences -- of the Jet's own first derivative (accurate) and of the primal (independent of Jet).
    template <typename F> void CheckAt(char const* name, F f, double x)
    {
        INFO(name << " at x = " << x);
        auto const s = f(SJ1 { x, 0 });
        auto const j = f(J1 { x, 0 });

        CHECK(Near(s.a, f(x), 1e-12));
        CHECK(Near(s.v[0], j.v[0], 1e-12));

        auto const h = 1e-5 * std::max(1.0, std::abs(x));
        auto const fromJet = (f(J1 { x + h, 0 }).v[0] - f(J1 { x - h, 0 }).v[0]) / (2.0 * h);
        CHECK(Near(s.w[0], fromJet, 1e-6));

        auto const g = 1e-3 * std::max(1.0, std::abs(x));
        auto const fromPrimal = (f(x + g) - 2.0 * f(x) + f(x - g)) / (g * g);
        CHECK(Near(s.w[0], fromPrimal, 1e-4));
    }

    template <typename F> void CheckAll(char const* name, F f, std::vector<double> const& points)
    {
        for (auto const x : points) {
            CheckAt(name, f, x);
        }
    }

    // The interval jet over the box [lo, hi] must contain the order-two Jet<double> of every sampled point inside it.
    template <typename F> void CheckEnclosure(char const* name, F f, double lo, double hi)
    {
        INFO(name << " over [" << lo << ", " << hi << "]");
        auto const r = f(SIJ { Iv { lo, hi }, 0 });
        auto const contains = [](Iv const& bound, double value) {
            if (!std::isfinite(value)) {
                return true; // a singular/NaN sample point has nothing to enclose
            }
            auto const slack = 1e-9 * (1.0 + std::abs(value));
            return !bound.is_empty() && bound.inf() - slack <= value && value <= bound.sup() + slack;
        };
        constexpr int Samples = 33;
        for (int i = 0; i < Samples; ++i) {
            auto const x = lo + (hi - lo) * static_cast<double>(i) / (Samples - 1);
            auto const p = f(SJ1 { x, 0 });
            CHECK(contains(r.a, p.a));
            CHECK(contains(r.v[0], p.v[0]));
            CHECK(contains(r.w[0], p.w[0]));
        }
    }

    // Two sampled boxes around a point: narrow and wide.
    template <typename F> void CheckEnclosures(char const* name, F f, std::vector<double> const& centres)
    {
        for (auto const c : centres) {
            CheckEnclosure(name, f, c - 0.05, c + 0.05);
            CheckEnclosure(name, f, c - 0.2, c + 0.15);
        }
    }

    // A tree over the box: variables X1, X2, ... get consecutive entries, the first `ndiff` are differentiated.
    auto EvalTree(Operon::Tree const& tree, Operon::Dataset const& ds, std::vector<std::pair<S, S>> const& box,
        std::size_t ndiff) -> tl::expected<IntervalSecondOrderJetBounds<S>, std::string>
    {
        DomainMap domains;
        std::vector<Operon::Hash> vars;
        for (std::size_t i = 0; i < box.size(); ++i) {
            domains.insert_or_assign(VarHash(ds, i + 1), box[i]);
            if (i < ndiff) {
                vars.push_back(VarHash(ds, i + 1));
            }
        }
        return SOE { &tree, domains, vars }(tree.GetCoefficients());
    }

    auto Eval(std::string const& expr, std::vector<std::pair<S, S>> const& box, std::size_t ndiff)
        -> tl::expected<IntervalSecondOrderJetBounds<S>, std::string>
    {
        auto const ds = MakeDataset(box.size());
        auto const tree = InfixParser::ParseOrThrow(expr, ds);
        return EvalTree(tree, ds, box, ndiff);
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

    struct SecondOrderSoundness {
        std::size_t Evaluated {};
        std::size_t Certain {};
        std::size_t Checked {}; // value, first and second derivative comparisons against a regular reference point
        std::size_t Violations {};
        double WorstExcess {};
    };

    // Evaluates `expr` over `boxes` random boxes with the first `ndiff` variables differentiated and checks every
    // certain enclosure (value, first and second derivative) against `samples` reference points of the box. The
    // reference is the double-precision first-order Jet tree evaluator of the interval-jet tests: the second
    // derivative is the central difference of its first derivative along the variable. T is the interval scalar of
    // the evaluator under test: float is the production type, double checks the formulas with a tight slack.
    template <typename T = S>
    auto Soundness(std::string const& expr, std::size_t nvars, std::size_t ndiff, std::size_t boxes,
        std::size_t samples, bool randomCoefficients = false, std::uint64_t seed = 20260101) -> SecondOrderSoundness
    {
        auto const ds = MakeDataset(nvars);
        auto const tree = InfixParser::ParseOrThrow(expr, ds);
        std::vector<Operon::Hash> vars;
        for (std::size_t i = 1; i <= ndiff; ++i) {
            vars.push_back(VarHash(ds, i));
        }
        std::mt19937_64 rng { seed };
        auto const slack = std::same_as<T, double> ? 1e-5 : 2e-3;
        SecondOrderSoundness out;
        for (std::size_t b = 0; b < boxes; ++b) {
            auto const domains = RandomDomains(rng, ds, nvars);
            auto coeff = tree.GetCoefficients();
            if (randomCoefficients) {
                for (auto& c : coeff) {
                    c = static_cast<S>(std::uniform_real_distribution<double> { -2.0, 2.0 }(rng));
                }
            }
            auto const bounds = Operon::IntervalSecondOrderJetEvaluator<T> { &tree, domains, vars }(coeff);
            REQUIRE(bounds.has_value());
            ++out.Evaluated;

            for (std::size_t k = 0; k < ndiff; ++k) {
                out.Certain += bounds->Status[k] == Status::Certain ? 1 : 0;
            }
            for (std::size_t s = 0; s < samples; ++s) {
                Operon::Map<Operon::Hash, double> point;
                for (std::size_t i = 1; i <= nvars; ++i) {
                    auto const [lo, hi] = domains.at(VarHash(ds, i));
                    auto const u = std::uniform_real_distribution<double> { 0.1, 0.9 }(rng);
                    point[VarHash(ds, i)] = lo + (static_cast<double>(hi) - static_cast<double>(lo)) * u;
                }
                bool regular = true;
                auto const centre = Reference(tree, coeff, vars, point, regular);
                if (!regular) {
                    continue;
                }
                for (std::size_t k = 0; k < ndiff; ++k) {
                    if (bounds->Status[k] != Status::Certain) {
                        continue;
                    }
                    auto const [lo, hi] = domains.at(vars[k]);
                    auto const h = 1e-4 * std::max(static_cast<double>(hi) - static_cast<double>(lo), 1e-2);
                    auto plus = point;
                    auto minus = point;
                    plus[vars[k]] += h;
                    minus[vars[k]] -= h;
                    bool regularPlus = true;
                    bool regularMinus = true;
                    auto const up = Reference(tree, coeff, vars, plus, regularPlus);
                    auto const down = Reference(tree, coeff, vars, minus, regularMinus);
                    if (!regularPlus || !regularMinus) {
                        continue;
                    }
                    auto const second = (up.v[k] - down.v[k]) / (2.0 * h);
                    auto const check = [&](auto const& bound, double reference) {
                        if (!std::isfinite(reference)) {
                            return;
                        }
                        ++out.Checked;
                        auto const lower = static_cast<double>(bound.inf());
                        auto const upper = static_cast<double>(bound.sup());
                        auto const excess = !(lower <= upper) ? std::numeric_limits<double>::infinity()
                                                              : std::max(lower - reference, reference - upper);
                        if (excess > slack * (1.0 + std::abs(reference))) {
                            ++out.Violations;
                            out.WorstExcess = std::max(out.WorstExcess, excess);
                        }
                    };
                    check(bounds->Value, centre.a);
                    check(bounds->First[k], centre.v[k]);
                    check(bounds->Second[k], second);
                }
            }
        }
        return out;
    }

    void ExpectSound(SecondOrderSoundness const& r)
    {
        INFO("worst excess " << r.WorstExcess);
        CHECK(r.Violations == 0);
        CHECK(r.Checked > 0);
        CHECK(r.Certain > 0);
    }
} // namespace

TEST_CASE("Jet<T, K, JetOrder::SecondDiagonal>: construction and primal comparisons", "[second-order-jet]")
{
    Jet<double, 3, Operon::JetOrder::SecondDiagonal> const zero;
    CHECK(zero.a == 0.0);
    for (std::size_t k = 0; k < 3; ++k) {
        CHECK(zero.v[k] == 0.0);
        CHECK(zero.w[k] == 0.0);
    }

    Jet<double, 3, Operon::JetOrder::SecondDiagonal> const constant { 2.5 };
    CHECK(constant.a == 2.5);
    CHECK(constant.v[1] == 0.0);
    CHECK(constant.w[1] == 0.0);

    Jet<double, 3, Operon::JetOrder::SecondDiagonal> const seeded { 4.0, 1 };
    CHECK(seeded.a == 4.0);
    CHECK(seeded.v[0] == 0.0);
    CHECK(seeded.v[1] == 1.0);
    CHECK(seeded.v[2] == 0.0);
    CHECK(seeded.w[1] == 0.0);

    Jet<double, 2, Operon::JetOrder::SecondDiagonal> const explicitJet { 1.0, { 2.0, 3.0 }, { 4.0, 5.0 } };
    CHECK(explicitJet.v[1] == 3.0);
    CHECK(explicitJet.w[0] == 4.0);

    // comparisons look at the primal only
    Jet<double, 1, Operon::JetOrder::SecondDiagonal> const a { 1.0, { 5.0 }, { 7.0 } };
    Jet<double, 1, Operon::JetOrder::SecondDiagonal> const b { 1.0, { 6.0 }, { 8.0 } };
    CHECK(a == b);
    CHECK(a == 1.0);
    CHECK(2.0 > a);
    CHECK(a <= b);
    CHECK(a != Jet<double, 1, Operon::JetOrder::SecondDiagonal> { 2.0 });
}

TEST_CASE("Jet<T, K, JetOrder::SecondDiagonal>: arithmetic matches finite differences", "[second-order-jet]")
{
    CheckAll("x + x", SOJ_FN(x + x), { -1.0, 0.5 });
    CheckAll("x - 3", SOJ_FN(x - C(3.0)), { -1.0, 0.5 });
    CheckAll("3 - x*x", SOJ_FN(C(3.0) - x * x), { -1.0, 0.5 });
    CheckAll("x * x * x", SOJ_FN(x * x * x), { -1.5, 0.0, 2.0 });
    CheckAll("2 * x * x", SOJ_FN(C(2.0) * x * x), { -1.5, 0.0, 2.0 });
    CheckAll("x * 2", SOJ_FN(x * C(2.0)), { -1.5, 0.0, 2.0 });
    CheckAll("(x + 1) / (x * x + 2)", SOJ_FN((x + C(1.0)) / (x * x + C(2.0))), { -2.0, -0.3, 0.8, 3.0 });
    CheckAll("2 / x", SOJ_FN(C(2.0) / x), { -2.0, 0.4, 3.0 });
    CheckAll("x / 3", SOJ_FN(x / C(3.0)), { -2.0, 0.4, 3.0 });
    CheckAll("1 / (1 + x*x)", SOJ_FN(C(1.0) / (C(1.0) + x * x)), { -1.5, 0.0, 0.7 });
    CheckAll("sin(x) / x", SOJ_FN(sin(x) / x), { -2.0, 0.5, 3.0 });
    CheckAll("-x * x", SOJ_FN(-x * x), { -2.0, 0.5 });
    CheckAll("x * sin(x)", SOJ_FN(x * sin(x)), { -2.0, 0.5, 3.0 });

    SECTION("compound assignment")
    {
        SJ1 x { 1.7, 0 };
        auto const reference = ((x + 2.0) * x - 1.0) / (x * x + 1.0);
        SJ1 y = x;
        y += 2.0;
        y *= x;
        y -= 1.0;
        y /= x * x + 1.0;
        CHECK(Near(y.a, reference.a, 1e-14));
        CHECK(Near(y.v[0], reference.v[0], 1e-14));
        CHECK(Near(y.w[0], reference.w[0], 1e-14));
        SJ1 z = x;
        z += x;
        z -= SJ1 { 0.2 };
        z *= 3.0;
        z /= 2.0;
        auto const expectedZ = (x + x - 0.2) * 3.0 / 2.0;
        CHECK(Near(z.w[0], expectedZ.w[0], 1e-14));
        CHECK(Near(z.v[0], 3.0, 1e-14));
    }
}

TEST_CASE("Jet<T, K, JetOrder::SecondDiagonal>: elementary functions match finite differences", "[second-order-jet]")
{
    CheckAll("sin", SOJ_FN(sin(x)), { -2.1, -0.3, 0.7, 3.0 });
    CheckAll("cos", SOJ_FN(cos(x)), { -2.1, -0.3, 0.7, 3.0 });
    CheckAll("tan", SOJ_FN(tan(x)), { -1.2, -0.4, 0.5, 1.1 });
    CheckAll("exp", SOJ_FN(exp(x)), { -1.5, 0.2, 2.0 });
    CheckAll("log", SOJ_FN(log(x)), { 0.2, 1.0, 4.5 });
    CheckAll("log1p", SOJ_FN(log1p(x)), { -0.5, 0.3, 3.0 });
    CheckAll("sqrt", SOJ_FN(sqrt(x)), { 0.3, 1.0, 7.0 });
    CheckAll("cbrt", SOJ_FN(cbrt(x)), { -8.0, -0.5, 0.4, 5.0 });
    CheckAll("asin", SOJ_FN(asin(x)), { -0.7, 0.1, 0.8 });
    CheckAll("acos", SOJ_FN(acos(x)), { -0.7, 0.1, 0.8 });
    CheckAll("atan", SOJ_FN(atan(x)), { -3.0, 0.2, 1.5 });
    CheckAll("sinh", SOJ_FN(sinh(x)), { -2.0, 0.3, 1.2 });
    CheckAll("cosh", SOJ_FN(cosh(x)), { -2.0, 0.3, 1.2 });
    CheckAll("tanh", SOJ_FN(tanh(x)), { -2.0, 0.3, 1.2 });
    CheckAll("square", SOJ_FN(square(x)), { -2.0, 0.0, 1.2 });
    CheckAll("logabs", SOJ_FN(logabs(x)), { -2.5, -0.2, 0.9, 3.0 });
    CheckAll("sqrtabs", SOJ_FN(sqrtabs(x)), { -4.0, -0.3, 0.6, 2.0 });
    CheckAll("abs (a kink: only the operand's curvature propagates)", SOJ_FN(abs(x)), { -1.5, 0.7 });
    CheckAll("abs(x * x - 1)", SOJ_FN(abs(x * x - C(1.0))), { -2.0, 0.3, 1.7 });
}

TEST_CASE("Jet<T, K, JetOrder::SecondDiagonal>: composition follows the second-order chain rule", "[second-order-jet]")
{
    CheckAll("sin(x * x)", SOJ_FN(sin(x * x)), { -1.3, 0.4, 1.9 });
    CheckAll("exp(sin(x))", SOJ_FN(exp(sin(x))), { -2.0, 0.5, 2.5 });
    CheckAll("log(1 + x * x)", SOJ_FN(log(C(1.0) + x * x)), { -2.0, 0.0, 1.5 });
    CheckAll("sqrt(x * x + 1)", SOJ_FN(sqrt(x * x + C(1.0))), { -2.0, 0.0, 1.5 });
    CheckAll("tanh(x) * atan(x)", SOJ_FN(tanh(x) * atan(x)), { -2.0, 0.3, 1.5 });
    CheckAll("x / (1 + x * x)", SOJ_FN(x / (C(1.0) + x * x)), { -2.0, 0.3, 1.5 });
    CheckAll("cos(x) / (2 + sin(x))", SOJ_FN(cos(x) / (C(2.0) + sin(x))), { -2.0, 0.3, 2.5 });
    CheckAll("exp(-x * x)", SOJ_FN(exp(-x * x)), { -1.5, 0.2, 0.9 });
    CheckAll("atan(x * x - 1)", SOJ_FN(atan(x * x - C(1.0))), { -1.5, 0.2, 0.9 });
    CheckAll("asin(x / 2) + acos(x / 3)", SOJ_FN(asin(x / C(2.0)) + acos(x / C(3.0))), { -1.2, 0.2, 0.9 });
    CheckAll("sinh(x) * cosh(x)", SOJ_FN(sinh(x) * cosh(x)), { -1.2, 0.2, 0.9 });
    CheckAll("log(x) * sqrt(x) / (1 + x)", SOJ_FN(log(x) * sqrt(x) / (C(1.0) + x)), { 0.3, 1.2, 4.0 });
    CheckAll("cbrt(x * x + 2) + tan(x / 3)", SOJ_FN(cbrt(x * x + C(2.0)) + tan(x / C(3.0))), { -2.0, 0.3, 1.5 });
}

TEST_CASE("Jet<T, K, JetOrder::SecondDiagonal>: binary functions (both operands depend on the direction)",
    "[second-order-jet]")
{
    SECTION("pow with a constant exponent")
    {
        CheckAll("x^2.5", SOJ_FN(pow(x, C(2.5))), { 0.5, 1.7, 4.0 });
        CheckAll("x^3", SOJ_FN(pow(x, C(3.0))), { -1.5, 0.4, 2.0 });
        CheckAll("x^2", SOJ_FN(pow(x, C(2.0))), { -1.2, 0.0, 0.7 });
        CheckAll("x^1", SOJ_FN(pow(x, C(1.0))), { -1.2, 0.0, 0.7 });
        CheckAll("x^0", SOJ_FN(pow(x, C(0.0))), { 0.5, 1.5 });
        CheckAll("x^-2", SOJ_FN(pow(x, C(-2.0))), { -1.5, 0.5, 2.0 });
    }
    SECTION("pow with a constant base")
    {
        CheckAll("2^x", SOJ_FN(pow(C(2.0), x)), { -1.0, 0.5, 2.0 });
        CheckAll("0.5^(x*x)", SOJ_FN(pow(C(0.5), x * x)), { -1.0, 0.5, 2.0 });
    }
    SECTION("pow with both operands carrying derivatives")
    {
        CheckAll("x^x", SOJ_FN(pow(x, x)), { 0.5, 1.3, 3.0 });
        CheckAll("x^(sin(x) + 2)", SOJ_FN(pow(x, sin(x) + C(2.0))), { 0.4, 1.5, 2.5 });
        CheckAll("(x*x + 1)^x", SOJ_FN(pow(x * x + C(1.0), x)), { -1.2, 0.3, 1.8 });
        CheckAll("(2 + cos(x))^(x*x/4)", SOJ_FN(pow(C(2.0) + cos(x), x * x / C(4.0))), { -1.2, 0.3, 1.8 });
    }
    SECTION("powabs")
    {
        CheckAll("|x|^2.5", SOJ_FN(powabs(x, C(2.5))), { -2.0, -0.4, 0.9, 3.0 });
        CheckAll("|-3|^x", SOJ_FN(powabs(C(-3.0), x)), { -1.0, 0.5, 2.0 });
        CheckAll("|x|^x", SOJ_FN(powabs(x, x)), { -1.3, 0.6, 2.2 });
        CheckAll("|x|^(sin(x) + 2)", SOJ_FN(powabs(x, sin(x) + C(2.0))), { -1.7, 0.8, 2.4 });
    }
    SECTION("aq")
    {
        CheckAll("aq(x, 2)", SOJ_FN(aq(x, C(2.0))), { -1.5, 0.3, 2.0 });
        CheckAll("aq(1.5, x)", SOJ_FN(aq(C(1.5), x)), { -1.5, 0.3, 2.0 });
        CheckAll("aq(x, x)", SOJ_FN(aq(x, x)), { -1.5, 0.3, 2.0 });
        CheckAll("aq(x * x, cos(x) + 0.5)", SOJ_FN(aq(x * x, cos(x) + C(0.5))), { -1.5, 0.3, 2.0 });
    }
    SECTION("fmin / fmax: weights are constant between ties, only the operands' curvature propagates")
    {
        CheckAll("fmin(x, 0.5 x^2 + 0.3)", SOJ_FN(fmin(x, C(0.5) * x * x + C(0.3))), { -1.0, 0.8, 2.0 });
        CheckAll("fmax(x, sin(x))", SOJ_FN(fmax(x, sin(x))), { -1.0, 1.0, 2.5 });
        CheckAll("fmin(x * x, 1)", SOJ_FN(fmin(x * x, C(1.0))), { -2.0, 0.4, 3.0 });
        CheckAll("fmax(0.2, x * x)", SOJ_FN(fmax(C(0.2), x * x)), { -2.0, 0.1, 3.0 });
    }
}

TEST_CASE("Jet<T, K, JetOrder::SecondDiagonal>: exact second derivative of constant exponents at a zero base",
    "[second-order-jet]")
{
    // x^2 is the line with curvature 2 even at 0 (0 * 0^-1 is not evaluated), x^1 and x^0 have none.
    auto const square = Operon::pow(SJ1 { 0.0, 0 }, 2.0);
    CHECK(square.a == 0.0);
    CHECK(square.v[0] == 0.0);
    CHECK(square.w[0] == 2.0);
    auto const line = Operon::pow(SJ1 { 0.0, 0 }, 1.0);
    CHECK(line.v[0] == 1.0);
    CHECK(line.w[0] == 0.0);
    auto const constant = Operon::pow(SJ1 { 0.0, 0 }, 0.0);
    CHECK(constant.a == 1.0);
    CHECK(constant.v[0] == 0.0);
    CHECK(constant.w[0] == 0.0);

    // the same through pow(Jet<T, K, 2>, Jet<T, K, 2>) with an exponent that carries no derivative
    for (auto const base : { 0.0, 1.5 }) {
        INFO("base " << base);
        auto const squareJ = Operon::pow(SJ1 { base, 0 }, SJ1 { 2.0 });
        CHECK(squareJ.v[0] == 2.0 * base);
        CHECK(squareJ.w[0] == 2.0);
        auto const lineJ = Operon::pow(SJ1 { base, 0 }, SJ1 { 1.0 });
        CHECK(lineJ.a == base);
        CHECK(lineJ.v[0] == 1.0);
        CHECK(lineJ.w[0] == 0.0);
        auto const constantJ = Operon::pow(SJ1 { base, 0 }, SJ1 { 0.0 });
        CHECK(constantJ.a == 1.0);
        CHECK(constantJ.v[0] == 0.0);
        CHECK(constantJ.w[0] == 0.0);
    }

    // an exponent that does depend on the direction keeps the full formula away from the exact cases
    CheckAll("x^(x/2 + 1)", SOJ_FN(pow(x, x / C(2.0) + C(1.0))), { 0.5, 1.7 });

    // intervals: a degenerate exponent interval takes the exact cases too, over a box that contains 0
    using IvJet = SIJ;
    auto const box = IvJet { Iv { -0.5, 0.5 }, 0 };
    auto const squareI = Operon::pow(box, IvJet { Iv { 2.0 } });
    CHECK(squareI.w[0].inf() == 2.0);
    CHECK(squareI.w[0].sup() == 2.0);
    auto const lineI = Operon::pow(box, IvJet { Iv { 1.0 } });
    CHECK(lineI.w[0].inf() == 0.0);
    CHECK(lineI.w[0].sup() == 0.0);
    auto const constantI = Operon::pow(IvJet { Iv { 0.0, 0.0 }, 0 }, IvJet { Iv { 0.0 } });
    CHECK(!constantI.w[0].is_empty());
    CHECK(constantI.w[0].inf() == 0.0);
    CHECK(constantI.w[0].sup() == 0.0);
}

TEST_CASE(
    "Jet<T, K, JetOrder::SecondDiagonal>: directions are independent pure second derivatives", "[second-order-jet]")
{
    using SJ3 = Operon::Jet<double, 3, Operon::JetOrder::SecondDiagonal>;
    SJ3 const x { 1.3, 0 };
    SJ3 const y { -0.7, 1 };
    SJ3 const z { 2.1, 2 };

    // f = x * y + sin(z) * x + y^3: f_xx = 0, f_yy = 6 y, f_zz = -sin(z) x -- no mixed terms leak into the diagonal
    auto const f = x * y + sin(z) * x + y * y * y;
    CHECK(Near(f.a, 1.3 * -0.7 + std::sin(2.1) * 1.3 + -0.7 * -0.7 * -0.7, 1e-14));
    CHECK(Near(f.v[0], -0.7 + std::sin(2.1), 1e-14));
    CHECK(Near(f.v[1], 1.3 + 3.0 * 0.7 * 0.7, 1e-14));
    CHECK(Near(f.v[2], std::cos(2.1) * 1.3, 1e-14));
    CHECK(Near(f.w[0], 0.0, 1e-14));
    CHECK(Near(f.w[1], 6.0 * -0.7, 1e-14));
    CHECK(Near(f.w[2], -std::sin(2.1) * 1.3, 1e-14));

    // a direction does not change when others are added: the same function in a 1-direction jet
    auto const single = SJ1 { 2.1, 0 };
    auto const g = Operon::sin(single) * 1.3;
    CHECK(Near(g.w[0], f.w[2], 1e-14));

    // a shared operand between the two operands of a product contributes the cross term 2 f' g'
    auto const sq = x * x;
    CHECK(Near(sq.w[0], 2.0, 1e-14));
    CHECK(Near(sq.w[1], 0.0, 1e-14));
}

TEST_CASE("Jet<T, K, JetOrder::SecondDiagonal> over intervals: enclosures contain sampled order-two jets",
    "[second-order-jet]")
{
    SECTION("arithmetic and chains")
    {
        CheckEnclosures("x * x * x", SOJ_FN(x * x * x), { -1.5, 0.0, 2.0 });
        CheckEnclosures("(x + 1) / (x * x + 2)", SOJ_FN((x + C(1.0)) / (x * x + C(2.0))), { -2.0, 0.8, 3.0 });
        CheckEnclosures("2 / x", SOJ_FN(C(2.0) / x), { -2.0, 3.0 });
        CheckEnclosures("sin(x * x)", SOJ_FN(sin(x * x)), { -1.3, 0.4, 1.9 });
        CheckEnclosures("exp(sin(x))", SOJ_FN(exp(sin(x))), { -2.0, 0.5, 2.5 });
        CheckEnclosures("x / (1 + x * x)", SOJ_FN(x / (C(1.0) + x * x)), { -2.0, 0.3, 1.5 });
        CheckEnclosures("x * sin(x)", SOJ_FN(x * sin(x)), { -2.0, 0.5, 3.0 });
    }
    SECTION("elementary functions")
    {
        CheckEnclosures("sin", SOJ_FN(sin(x)), { -2.1, 0.7, 3.0 });
        CheckEnclosures("cos", SOJ_FN(cos(x)), { -2.1, 0.7, 3.0 });
        CheckEnclosures("tan", SOJ_FN(tan(x)), { -0.9, 0.5 });
        CheckEnclosures("exp", SOJ_FN(exp(x)), { -1.5, 2.0 });
        CheckEnclosures("log", SOJ_FN(log(x)), { 0.5, 4.5 });
        CheckEnclosures("log1p", SOJ_FN(log1p(x)), { -0.5, 3.0 });
        CheckEnclosures("sqrt", SOJ_FN(sqrt(x)), { 0.5, 7.0 });
        CheckEnclosures("cbrt", SOJ_FN(cbrt(x)), { -8.0, 0.5, 5.0 });
        CheckEnclosures("asin", SOJ_FN(asin(x)), { -0.5, 0.1, 0.6 });
        CheckEnclosures("acos", SOJ_FN(acos(x)), { -0.5, 0.1, 0.6 });
        CheckEnclosures("atan", SOJ_FN(atan(x)), { -3.0, 0.2, 1.5 });
        CheckEnclosures("sinh", SOJ_FN(sinh(x)), { -2.0, 1.2 });
        CheckEnclosures("cosh", SOJ_FN(cosh(x)), { -2.0, 1.2 });
        CheckEnclosures("tanh", SOJ_FN(tanh(x)), { -2.0, 0.3, 1.2 });
        CheckEnclosures("square", SOJ_FN(square(x)), { -2.0, 0.0, 1.2 });
        CheckEnclosures("logabs", SOJ_FN(logabs(x)), { -2.5, 0.9, 3.0 });
        CheckEnclosures("sqrtabs", SOJ_FN(sqrtabs(x)), { -4.0, 0.9, 2.0 });
        CheckEnclosures("abs", SOJ_FN(abs(x)), { -1.5, 0.7 });
    }
    SECTION("pow, powabs, aq, fmin, fmax")
    {
        CheckEnclosures("x^2.5", SOJ_FN(pow(x, C(2.5))), { 0.5, 1.7, 4.0 });
        CheckEnclosures("x^3", SOJ_FN(pow(x, C(3.0))), { -1.5, 0.0, 2.0 });
        CheckEnclosures("x^2", SOJ_FN(pow(x, C(2.0))), { -1.2, 0.0, 0.7 });
        CheckEnclosures("x^1", SOJ_FN(pow(x, C(1.0))), { -1.2, 0.0, 0.7 });
        CheckEnclosures("x^0", SOJ_FN(pow(x, C(0.0))), { -0.5, 0.5, 1.5 });
        CheckEnclosures("2^x", SOJ_FN(pow(C(2.0), x)), { -1.0, 0.5, 2.0 });
        CheckEnclosures("x^x", SOJ_FN(pow(x, x)), { 0.6, 1.3, 3.0 });
        CheckEnclosures("x^(sin(x) + 2)", SOJ_FN(pow(x, sin(x) + C(2.0))), { 0.5, 1.5, 2.5 });
        CheckEnclosures("|x|^2.5", SOJ_FN(powabs(x, C(2.5))), { -2.0, 0.9, 3.0 });
        CheckEnclosures("|x|^x", SOJ_FN(powabs(x, x)), { -1.3, 0.6, 2.2 });
        CheckEnclosures("aq(x, 2)", SOJ_FN(aq(x, C(2.0))), { -1.5, 0.3, 2.0 });
        CheckEnclosures("aq(1.5, x)", SOJ_FN(aq(C(1.5), x)), { -1.5, 0.3, 2.0 });
        CheckEnclosures("aq(x * x, cos(x) + 0.5)", SOJ_FN(aq(x * x, cos(x) + C(0.5))), { -1.5, 0.3, 2.0 });
        CheckEnclosures("fmin(x, 0.5 x^2 + 0.3)", SOJ_FN(fmin(x, C(0.5) * x * x + C(0.3))), { -1.0, 0.8, 2.0 });
        CheckEnclosures("fmax(x, sin(x))", SOJ_FN(fmax(x, sin(x))), { -1.0, 1.0, 2.5 });
    }
}

TEST_CASE("Jet<T, K, JetOrder::SecondDiagonal> over intervals: the squared first derivatives stay tight",
    "[second-order-jet]")
{
    SECTION("square: the curvature is exactly [2, 2]")
    {
        auto const r = Operon::square(SIJ { Iv { -1.0, 3.0 }, 0 });
        CHECK(r.w[0].inf() == 2.0);
        CHECK(r.w[0].sup() == 2.0);
        CHECK(r.v[0].inf() == -2.0);
        CHECK(r.v[0].sup() == 6.0);
    }
    SECTION("x^2 over a zero-straddling box: the curvature is exactly [2, 2]")
    {
        auto const r = Operon::pow(SIJ { Iv { -1.0, 1.0 }, 0 }, Iv { 2.0 });
        CHECK(r.w[0].inf() == 2.0);
        CHECK(r.w[0].sup() == 2.0);
    }
    SECTION("exp: f'' = exp over the box")
    {
        auto const r = Operon::exp(SIJ { Iv { 0.0, 1.0 }, 0 });
        CHECK(r.w[0].inf() <= 1.0);
        CHECK(r.w[0].inf() >= 1.0 - 1e-12);
        CHECK(r.w[0].sup() >= std::exp(1.0));
        CHECK(r.w[0].sup() <= std::exp(1.0) + 1e-12);
    }
    SECTION("1 / x over [1, 2]: f'' = 2 / x^3 in [0.25, 2]")
    {
        auto const r = Iv { 1.0 } / SIJ { Iv { 1.0, 2.0 }, 0 };
        CHECK(r.w[0].inf() <= 0.25 + 1e-12);
        CHECK(r.w[0].inf() >= 0.25 - 0.05);
        CHECK(r.w[0].sup() >= 2.0);
    }
}

TEST_CASE("IntervalSecondOrderJetEvaluator: known enclosures", "[second-order-jet]")
{
    SECTION("x^2 over [0, 1] and over the point 0: first derivative [0, 2] / [0, 0], second derivative exactly 2")
    {
        auto const wide = Eval("X1 ^ 2", { { 0.0F, 1.0F } }, 1);
        REQUIRE(wide);
        REQUIRE(wide->Status[0] == Status::Certain);
        CHECK(wide->First[0].inf() == 0.0F);
        CHECK(wide->First[0].sup() == 2.0F);
        CHECK(wide->Second[0] == Interval { 2.0F });
        CHECK(wide->Value.inf() == 0.0F);
        CHECK(wide->Value.sup() == 1.0F);

        auto const point = Eval("X1 ^ 2", { { 0.0F, 0.0F } }, 1);
        REQUIRE(point);
        REQUIRE(point->Status[0] == Status::Certain);
        CHECK(point->Second[0] == Interval { 2.0F });
    }
    SECTION("x^3 over [-1, 2]: second derivative 6 x in [-6, 12]")
    {
        auto const r = Eval("X1 ^ 3", { { -1.0F, 2.0F } }, 1);
        REQUIRE(r);
        REQUIRE(r->Status[0] == Status::Certain);
        CHECK(r->Second[0].inf() <= -6.0F);
        CHECK(r->Second[0].sup() >= 12.0F);
        CHECK(r->Second[0].inf() >= -6.01F);
        CHECK(r->Second[0].sup() <= 12.01F);
    }
    SECTION("(X1 - 2)^2 * X2 over [1, 5]^2: d2/dX1^2 = 2 X2 in [2, 10]; the other direction is the plain jet")
    {
        auto const r = Eval("(X1 - 2) * (X1 - 2) * X2", { { 1.0F, 5.0F }, { 1.0F, 5.0F } }, 2);
        REQUIRE(r);
        REQUIRE(r->Status[0] == Status::Certain);
        REQUIRE(r->Status[1] == Status::Certain);
        CHECK(r->Second[0].inf() <= 2.0F);
        CHECK(r->Second[0].sup() >= 10.0F);
        // linear in X2: no curvature, exactly
        CHECK(r->Second[1] == Interval { 0.0F });
    }
    SECTION("a variable that is not in the tree has zero derivatives")
    {
        auto const r = Eval("X1 * X1", { { 1.0F, 2.0F }, { 3.0F, 4.0F } }, 2);
        REQUIRE(r);
        CHECK(r->Status[1] == Status::Certain);
        CHECK(r->First[1] == Interval { 0.0F });
        CHECK(r->Second[1] == Interval { 0.0F });
    }
    SECTION("atan(x) over [0, 1]: f'' = -2x / (1 + x^2)^2 stays within [-2, 0]")
    {
        auto const r = Eval("atan(X1)", { { 0.0F, 1.0F } }, 1);
        REQUIRE(r);
        REQUIRE(r->Status[0] == Status::Certain);
        CHECK(r->Second[0].inf() <= -0.65F);
        CHECK(r->Second[0].inf() >= -2.0001F);
        CHECK(r->Second[0].sup() >= 0.0F);
        CHECK(r->Second[0].sup() <= 0.0001F);
    }
}

TEST_CASE("IntervalSecondOrderJetEvaluator: second-derivative enclosures contain sampled true values (unary ops)",
    "[second-order-jet]")
{
    for (auto const* expr :
        { "sqrt(X1)", "cbrt(X1)", "exp(X1)", "log(X1)", "log1p(X1)", "logabs(X1)", "sin(X1)", "cos(X1)", "tan(X1)",
            "asin(X1)", "acos(X1)", "atan(X1)", "sinh(X1)", "cosh(X1)", "tanh(X1)", "square(X1)", "-X1", "1 / X1" }) {
        DYNAMIC_SECTION(expr)
        {
            ExpectSound(Soundness(expr, 1, 1, 300, 12));
            ExpectSound(Soundness<double>(expr, 1, 1, 300, 12));
        }
    }
}

TEST_CASE("IntervalSecondOrderJetEvaluator: second-derivative enclosures contain sampled true values (binary ops)",
    "[second-order-jet]")
{
    for (auto const* expr : { "X1 + X2", "X1 - X2", "X1 * X2", "X1 / X2", "X1 ^ X2", "X1 ^ 2", "X1 ^ 3", "X1 ^ 2.5",
             "X1 ^ -1", "2 ^ X1", "X2 ^ 2 + X1", "X1 * X1 * X2", "sin(X1 * X2) + exp(X1) / (X2 ^ 2 + 1)",
             "log(X1 * X1 + X2 * X2 + 1) * atan(X1 / (X2 * X2 + 0.5))", "sqrt(X1 * X1 + X2 * X2) - tanh(X1 * X2)",
             "square(X1 - X2) / (cosh(X1) + 1)", "asin(X1 / (1 + square(X2))) + acos(tanh(X1))",
             "1 / (1 + X1 * X1) - cbrt(X2) * log1p(X1 * X1)" }) {
        DYNAMIC_SECTION(expr)
        {
            ExpectSound(Soundness(expr, 2, 2, 300, 12));
            ExpectSound(Soundness(expr, 2, 2, 150, 12, /*randomCoefficients=*/true, 99));
            ExpectSound(Soundness<double>(expr, 2, 2, 300, 12));
            ExpectSound(Soundness<double>(expr, 2, 2, 150, 12, /*randomCoefficients=*/true, 99));
        }
    }
}

TEST_CASE("IntervalSecondOrderJetEvaluator: 3 and 4 differentiated variables, undifferentiated variables are constants",
    "[second-order-jet]")
{
    constexpr auto expr = "X1 * X2 + sin(X3) / (1 + X4 * X4) - X1 * X4 + exp(X2 * X3)";
    SECTION("K = 4")
    {
        ExpectSound(Soundness(expr, 4, 4, 200, 12));
        ExpectSound(Soundness<double>(expr, 4, 4, 200, 12));
    }
    SECTION("K = 3 (X4 is a constant over its box)")
    {
        ExpectSound(Soundness(expr, 4, 3, 200, 12));
        ExpectSound(Soundness<double>(expr, 4, 3, 200, 12));
    }
    SECTION("K = 2")
    {
        ExpectSound(Soundness(expr, 4, 2, 200, 12));
        ExpectSound(Soundness<double>(expr, 4, 2, 200, 12));
    }
}

TEST_CASE("IntervalSecondOrderJetEvaluator: directions are independent of K and of each other", "[second-order-jet]")
{
    constexpr auto expr = "sin(X1 * X2) + exp(X1) / (X2 ^ 2 + 1) - X3 * X1 / (X2 * X2 + 2)";
    auto const ds = MakeDataset(3);
    auto const tree = InfixParser::ParseOrThrow(expr, ds);
    DomainMap domains;
    domains.insert_or_assign(VarHash(ds, 1), std::pair { -0.5F, 1.0F });
    domains.insert_or_assign(VarHash(ds, 2), std::pair { 0.25F, 2.0F });
    domains.insert_or_assign(VarHash(ds, 3), std::pair { -1.0F, 1.0F });
    auto const coeff = tree.GetCoefficients();

    auto const all = SOE { &tree, domains, std::vector { VarHash(ds, 1), VarHash(ds, 2), VarHash(ds, 3) } }(coeff);
    REQUIRE(all);
    for (std::size_t k = 0; k < 3; ++k) {
        auto const single = SOE { &tree, domains, std::vector { VarHash(ds, k + 1) } }(coeff);
        REQUIRE(single);
        CHECK(single->Value == all->Value);
        CHECK(single->First[0] == all->First[k]);
        CHECK(single->Second[0] == all->Second[k]);
        CHECK(single->Status[0] == all->Status[k]);
    }
    // permuting the variables permutes the directions
    auto const swapped = SOE { &tree, domains, std::vector { VarHash(ds, 3), VarHash(ds, 1) } }(coeff);
    REQUIRE(swapped);
    CHECK(swapped->Second[0] == all->Second[2]);
    CHECK(swapped->Second[1] == all->Second[0]);
    CHECK(swapped->First[1] == all->First[0]);
}

TEST_CASE("IntervalSecondOrderJetEvaluator: non-smooth operations are uncertain, per direction", "[second-order-jet]")
{
    auto const ds = MakeDataset(2);
    auto const V = [&](std::size_t i) { return Node(NodeType::Variable, VarHash(ds, i)); };
    auto const F = [](BuiltinOp op, std::uint16_t arity) { return Node::Function(Operon::Hash(op), arity); };
    auto const C = [](double v) { return Node::Constant(v); };
    auto const Make = [](std::vector<Node> nodes) { return Tree(std::move(nodes)).UpdateNodes(); };
    // `a + X2 * X2` for the postfix nodes of a subtree `a` over X1
    auto const WithSquare = [&](std::vector<Node> const& a) {
        std::vector<Node> nodes { V(2), V(2), F(BuiltinOp::Mul, 2) };
        nodes.insert(nodes.end(), a.begin(), a.end());
        nodes.push_back(F(BuiltinOp::Add, 2));
        return Make(std::move(nodes));
    };

    SECTION("each non-smooth built-in flags exactly the directions it depends on")
    {
        std::vector<std::pair<std::string, Tree>> cases;
        for (auto const* expr : { "abs(X1) + X2 * X2", "sqrtabs(X1) + X2 * X2", "aq(X1, 2) + X2 * X2",
                 "powabs(X1, 2) + X2 * X2", "sin(abs(X1)) + X2 * X2" }) {
            cases.emplace_back(expr, InfixParser::ParseOrThrow(expr, ds));
        }
        cases.emplace_back("floor(X1) + X2 * X2", WithSquare({ V(1), F(BuiltinOp::Floor, 1) }));
        cases.emplace_back("ceil(X1) + X2 * X2", WithSquare({ V(1), F(BuiltinOp::Ceil, 1) }));
        cases.emplace_back("fmin(X1, 3) + X2 * X2", WithSquare({ C(3), V(1), F(BuiltinOp::Fmin, 2) }));
        cases.emplace_back("fmax(X1, 0) + X2 * X2", WithSquare({ C(0), V(1), F(BuiltinOp::Fmax, 2) }));
        cases.emplace_back("exp(floor(X1) * X1) + X2 * X2",
            WithSquare({ V(1), V(1), F(BuiltinOp::Floor, 1), F(BuiltinOp::Mul, 2), F(BuiltinOp::Exp, 1) }));
        for (auto const& [name, tree] : cases) {
            DYNAMIC_SECTION(name)
            {
                auto const r = EvalTree(tree, ds, { { 1.0F, 2.0F }, { 1.0F, 2.0F } }, 2);
                REQUIRE(r);
                CHECK(r->Status[0] == Status::Uncertain);
                REQUIRE(r->Status[1] == Status::Certain);
                CHECK(r->Second[1] == Interval { 2.0F });
            }
        }
    }
    SECTION("an operation on both variables flags both directions")
    {
        std::vector<std::pair<std::string, Tree>> cases;
        for (auto const* expr : { "aq(X1, X2)", "powabs(X1, X2)", "abs(X1 * X2)" }) {
            cases.emplace_back(expr, InfixParser::ParseOrThrow(expr, ds));
        }
        cases.emplace_back("fmin(X1, X2)", Make({ V(2), V(1), F(BuiltinOp::Fmin, 2) }));
        cases.emplace_back("fmax(X1, X2)", Make({ V(2), V(1), F(BuiltinOp::Fmax, 2) }));
        for (auto const& [name, tree] : cases) {
            DYNAMIC_SECTION(name)
            {
                auto const r = EvalTree(tree, ds, { { 1.0F, 2.0F }, { 1.0F, 2.0F } }, 2);
                REQUIRE(r);
                CHECK(r->Status[0] == Status::Uncertain);
                CHECK(r->Status[1] == Status::Uncertain);
            }
        }
    }
    SECTION("a non-smooth operation on an independent variable does not taint the direction")
    {
        // |X2| does not depend on X1: d2/dX1^2 of |X2| * X1^2 is 2 |X2|, certain
        auto const r = Eval("abs(X2) * X1 * X1", { { 1.0F, 2.0F }, { 1.0F, 2.0F } }, 2);
        REQUIRE(r);
        CHECK(r->Status[0] == Status::Certain);
        CHECK(r->Second[0].inf() <= 2.0F);
        CHECK(r->Second[0].sup() >= 4.0F);
        CHECK(r->Status[1] == Status::Uncertain);
    }
    SECTION("a non-smooth operation outside the differentiated variables is a constant")
    {
        auto const r = Eval("abs(X2) + X1 * X1", { { 1.0F, 2.0F }, { -1.0F, 2.0F } }, 1);
        REQUIRE(r);
        CHECK(r->Status[0] == Status::Certain);
        CHECK(r->Second[0] == Interval { 2.0F });
    }
    SECTION("the uncertain direction never reports a finite claim as Certain")
    {
        auto const r = Eval("abs(X1)", { { 1.0F, 2.0F } }, 1);
        REQUIRE(r);
        CHECK(r->Status[0] == Status::Uncertain);
        CHECK(r->First[0].inf() == -Inf);
        CHECK(r->First[0].sup() == Inf);
        CHECK(r->Second[0].inf() == -Inf);
        CHECK(r->Second[0].sup() == Inf);
    }
}

TEST_CASE(
    "IntervalSecondOrderJetEvaluator: a failing derivative does not leak into another direction", "[second-order-jet]")
{
    // d2 sqrt(x)/dx2 = -1 / (4 sqrt(x)^3) is undefined at the point domain {0}: X2's derivatives are domain errors,
    // while sqrt(0) itself and d/dX1, d2/dX1^2 are fine.
    auto const r = Eval("X1 * X1 + sqrt(X2)", { { 1.0F, 2.0F }, { 0.0F, 0.0F } }, 2);
    REQUIRE(r);
    CHECK(!r->Value.is_empty());
    CHECK(r->Status[0] == Status::Certain);
    CHECK(r->Second[0] == Interval { 2.0F });
    CHECK(r->Status[1] == Status::Failed);
    CHECK(r->Second[1].is_empty());
}

TEST_CASE("IntervalSecondOrderJetEvaluator: weights and coefficients follow the IntervalEvaluator convention",
    "[second-order-jet]")
{
    auto const ds = MakeDataset(2);
    auto const x1 = VarHash(ds, 1);

    // 0.75 * (2.5 * sin(X1) * 1.5 X1 ... ) built by hand: w * f(X) with an Optimize weight on the function node
    Node v1 = Node(NodeType::Variable, x1);
    v1.Value = S { 1.5 };
    v1.Optimize = false;
    Node sn = Node::Function(Operon::Hash(BuiltinOp::Sin), 1);
    sn.Value = S { 2.5 };
    sn.Optimize = true;
    auto const tree = Tree({ v1, sn }).UpdateNodes();

    DomainMap domains;
    domains.insert_or_assign(x1, std::pair { 0.5F, 0.6F });

    // 2.5 * sin(1.5 x)'' = -2.5 * 2.25 sin(1.5 x)
    auto const run = [&](S weight) {
        std::vector<S> coeff { weight };
        return SOE { &tree, domains, std::vector { x1 } }(coeff);
    };
    auto const r = run(2.5F);
    REQUIRE(r);
    REQUIRE(r->Status[0] == Status::Certain);
    for (auto const x : { 0.5, 0.55, 0.6 }) {
        auto const exact = -2.5 * 2.25 * std::sin(1.5 * x);
        CHECK(r->Second[0].inf() <= exact + 1e-5);
        CHECK(r->Second[0].sup() >= exact - 1e-5);
    }

    // a different coefficient is honoured; a zero weight gives exactly zero derivatives
    auto const scaled = run(-1.0F);
    REQUIRE(scaled);
    CHECK(scaled->Second[0].inf() > 0.0F);
    auto const zero = run(0.0F);
    REQUIRE(zero);
    CHECK(zero->Status[0] == Status::Certain);
    CHECK(zero->First[0] == Interval { 0.0F });
    CHECK(zero->Second[0] == Interval { 0.0F });
}

TEST_CASE("IntervalSecondOrderJetEvaluator: shared (Ref) subtrees and tree retargeting", "[second-order-jet]")
{
    auto const ds = MakeDataset(2);
    auto const x1 = VarHash(ds, 1);
    auto const x2 = VarHash(ds, 2);
    Node v1 = Node(NodeType::Variable, x1);
    Node sn = Node::Function(Operon::Hash(BuiltinOp::Sin), 1);
    Node ref = Node::Ref(1);
    Node mul = Node::Function(Operon::Hash(BuiltinOp::Mul), 2);
    Node v2 = Node(NodeType::Variable, x2);
    Node add = Node::Function(Operon::Hash(BuiltinOp::Add), 2);
    auto const tree = Tree({ v1, sn, ref, mul, v2, add }).UpdateNodes();
    REQUIRE(tree.Length() == 6);

    DomainMap domains;
    domains.insert_or_assign(x1, std::pair { 0.5F, 1.0F });
    domains.insert_or_assign(x2, std::pair { 1.0F, 2.0F });

    // sin(X1)^2 + X2: d2/dX1^2 = 2 cos(2 X1) in [2 cos(2), 2 cos(1)]
    SOE evaluator { &tree, domains, std::vector { x1, x2 } };
    auto const r = evaluator(tree.GetCoefficients());
    REQUIRE(r);
    REQUIRE(r->Status[0] == Status::Certain);
    CHECK(r->Second[0].inf() <= 2.0 * std::cos(2.0) + 1e-5);
    CHECK(r->Second[0].sup() >= 2.0 * std::cos(1.0) - 1e-5);
    CHECK(r->Second[1] == Interval { 0.0F });

    // SetTree retargets the same evaluator, whose scratch is reused
    auto const other = InfixParser::ParseOrThrow("X1 * X1 * X2", ds);
    evaluator.SetTree(&other);
    auto const o = evaluator(other.GetCoefficients());
    REQUIRE(o);
    CHECK(o->Second[0].inf() <= 2.0F);
    CHECK(o->Second[0].sup() >= 4.0F);
    // the same evaluator, evaluated again on the first tree, reproduces the first result
    evaluator.SetTree(&tree);
    auto const again = evaluator(tree.GetCoefficients());
    REQUIRE(again);
    CHECK(again->Second[0] == r->Second[0]);
    CHECK(again->Second[1] == r->Second[1]);
}

TEST_CASE("IntervalSecondOrderJetEvaluator: errors", "[second-order-jet]")
{
    SECTION("user-defined operations are not supported by the jet path")
    {
        auto const ds = MakeDataset(1);
        Node v = Node(NodeType::Variable, VarHash(ds, 1));
        auto const tree = Tree({ v, Node::Function(Operon::Hash { 9999 }, 1) }).UpdateNodes();
        DomainMap domains;
        domains.insert_or_assign(VarHash(ds, 1), std::pair { 0.0F, 1.0F });
        auto const r = SOE { &tree, domains, std::vector { VarHash(ds, 1) } }(tree.GetCoefficients());
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
        REQUIRE(!(SOE { &tree, domains, std::vector { VarHash(ds, 1) } }(tree.GetCoefficients())));
    }
    SECTION("missing domain bound")
    {
        auto const ds = MakeDataset(1);
        auto const tree = InfixParser::ParseOrThrow("X1 * X1", ds);
        DomainMap domains;
        REQUIRE(!(SOE { &tree, domains, std::vector { VarHash(ds, 1) } }(tree.GetCoefficients())));
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
        REQUIRE(!(SOE { &tree, domains, vars }(tree.GetCoefficients())));
        REQUIRE(!(SOE { &tree, domains, std::vector<Operon::Hash> {} }(tree.GetCoefficients())));
        vars.pop_back();
        auto const four = SOE { &tree, domains, vars }(tree.GetCoefficients());
        REQUIRE(four);
        CHECK(four->Directions == 4);
        for (std::size_t k = 0; k < 4; ++k) {
            CHECK(four->First[k] == Interval { 1.0F });
            CHECK(four->Second[k] == Interval { 0.0F });
            CHECK(four->Status[k] == Status::Certain);
        }
    }
}

// The audit counterexample: a finite-difference containment harness once reported d2/dX1^2 values of about
// [-0.31, 0.31] against a [0, 0] Certain bound on this tree and blamed the Pow curvature rule. X1 enters only
// through X1 ^ X3 (X3's box straddles 1, so the true curvature takes both signs); X4 enters linearly, so its own
// direction is curvature-free -- the audit had compared X1's finite differences against the wrong direction's
// bound. The regression pins all three facts: the X1 enclosure contains the analytic curvature grid, the linear
// direction stays exactly zero, and an unevaluated direction can no longer be read as a certified [0, 0].
TEST_CASE("IntervalSecondOrderJetEvaluator: pow curvature stays with its own direction", "[second-order-jet]")
{
    auto const ds = MakeDataset(5);
    auto const tree = InfixParser::ParseOrThrow(
        "((exp((1 * (1 * X2))) ^ log(((1 * X2) * 1))) + (((1 * X4) * (1 * X3)) - ((1 * X1) ^ (1 * X3))))", ds);
    DomainMap domains;
    domains.insert_or_assign(VarHash(ds, 1), std::pair { S { 0.6F }, S { 1.4F } });
    domains.insert_or_assign(VarHash(ds, 2), std::pair { S { 0.5F }, S { 2.0F } });
    domains.insert_or_assign(VarHash(ds, 3), std::pair { S { 0.75F }, S { 1.45F } });
    domains.insert_or_assign(VarHash(ds, 4), std::pair { S { 0.7F }, S { 2.9F } });
    auto const coeff = tree.GetCoefficients();

    // d2/dX1^2 = -X3 (X3 - 1) X1^(X3 - 2) over an interior grid of the box
    auto const alongX1 = SOE { &tree, domains, std::vector { VarHash(ds, 1) } }(coeff);
    REQUIRE(alongX1);
    REQUIRE(alongX1->Status[0] == Status::Certain);
    for (int i = 0; i <= 6; ++i) {
        for (int j = 0; j <= 6; ++j) {
            auto const x1 = 0.6 + (1.4 - 0.6) * static_cast<double>(i) / 6.0;
            auto const x3 = 0.75 + (1.45 - 0.75) * static_cast<double>(j) / 6.0;
            auto const second = -x3 * (x3 - 1.0) * std::pow(x1, x3 - 2.0);
            INFO("x1 = " << x1 << ", x3 = " << x3);
            CHECK(alongX1->Second[0].inf() <= second + 1e-5);
            CHECK(alongX1->Second[0].sup() >= second - 1e-5);
        }
    }
    // X3 straddles 1: the enclosure is a genuine two-sided range, not a point
    CHECK(alongX1->Second[0].inf() < 0.0);
    CHECK(alongX1->Second[0].sup() > 0.0);

    // The reported direct finite difference is stable when evaluated along the same X1 direction.  The terms
    // preceding the subtraction do not depend on X1, so this is also an independent scalar oracle for the tree.
    auto const x1 = 1.1;
    auto const x3 = 1.35;
    auto const analytic = -x3 * (x3 - 1.0) * std::pow(x1, x3 - 2.0);
    auto const valueAlongX1 = [x3](double x) { return -std::pow(x, x3); };
    for (auto const step : { 1e-2, 2e-3, 5e-4 }) {
        auto const fd = (valueAlongX1(x1 + step) - 2.0 * valueAlongX1(x1) + valueAlongX1(x1 - step)) / (step * step);
        INFO("step = " << step);
        CHECK(std::abs(fd - analytic) < 1e-4);
        CHECK(alongX1->Second[0].inf() <= fd + 1e-5);
        CHECK(alongX1->Second[0].sup() >= fd - 1e-5);
    }

    // X4 is linear in the tree: its own direction carries no curvature
    auto const alongX4 = SOE { &tree, domains, std::vector { VarHash(ds, 4) } }(coeff);
    REQUIRE(alongX4);
    REQUIRE(alongX4->Status[0] == Status::Certain);
    CHECK(alongX4->Second[0] == Interval { 0.0F });
    // ... and that zero must not be readable as another direction's bound
    CHECK(!(alongX4->Second[0] == alongX1->Second[0]));

    // directions at or beyond Directions read Failed, never a default-Certain [0, 0]
    for (auto k = alongX4->Directions; k < alongX4->MaxDirections; ++k) {
        CHECK(alongX4->Status[k] == Status::Failed);
    }
}

namespace {
    // Deterministic TypeCoherent+tanh corpus over positive boxes (narrow/wide/point), random coefficients (a
    // quarter of trees in integer mode), 1..4 directions: every Certain second-derivative enclosure must contain
    // the central finite difference of the scalar reference at well-interior sample points. The finite-difference
    // signal is only meaningful while ulp(|f|) / h^2 stays well below the curvature scale: a sample whose tree
    // value is so large that double-precision cancellation could dominate is skipped (an unguarded harness
    // produces exactly the false alarms this suite guards against, e.g. fd == 0 or quantized noise against a
    // sound bound on a tree whose value is ~1e15).
    struct FiniteDifferenceSoundness {
        std::size_t Certain {};
        std::size_t Checked {};
        std::size_t Violations {};
    };

    auto ScalarFiniteDifferenceSoundness(std::uint64_t seed, std::size_t trees) -> FiniteDifferenceSoundness
    {
        auto const ds = MakeDataset(5);
        std::vector<Operon::Hash> allVars;
        for (std::size_t i = 1; i <= 5; ++i) {
            allVars.push_back(VarHash(ds, i));
        }
        PrimitiveSet pset { PrimitiveSet::TypeCoherent | BuiltinOp::Tanh };
        for (auto op : { BuiltinOp::Add, BuiltinOp::Sub, BuiltinOp::Mul, BuiltinOp::Div }) {
            pset.SetFrequency(Operon::Hash(op), 6);
        }
        pset.SetMinMaxArity(Operon::Hash(BuiltinOp::Add), 2, 4);
        pset.SetMinMaxArity(Operon::Hash(BuiltinOp::Mul), 2, 4);
        pset.SetMinMaxArity(Operon::Hash(BuiltinOp::Sub), 1, 2);
        pset.SetMinMaxArity(Operon::Hash(BuiltinOp::Div), 1, 2);
        pset.SetFrequency(Node(NodeType::Variable).HashValue, 3);

        auto const randomBox = [](std::mt19937_64& rng) -> std::pair<S, S> {
            auto const u = [&](double a, double b) { return std::uniform_real_distribution<double> { a, b }(rng); };
            // narrow positive / wide positive / positive point
            switch (std::uniform_int_distribution<int> { 0, 3 }(rng)) {
            case 0:
            case 1: {
                auto const lo = u(0.1, 3.0);
                return { static_cast<S>(lo), static_cast<S>(lo + u(0.01, 0.5)) };
            }
            case 2: {
                auto const lo = u(0.01, 1.0);
                return { static_cast<S>(lo), static_cast<S>(lo + u(1.0, 6.0)) };
            }
            default: {
                auto const lo = u(0.2, 3.0);
                return { static_cast<S>(lo), static_cast<S>(lo) };
            }
            }
        };

        Operon::RandomGenerator treeRng { seed };
        std::mt19937_64 rng { seed ^ 0x9E3779B97F4A7C15ULL };
        FiniteDifferenceSoundness out;
        std::size_t made = 0;
        while (made < trees) {
            auto const nvars = std::uniform_int_distribution<std::size_t> { 2, 5 }(rng);
            std::vector<Operon::Hash> vars(allVars.begin(), allVars.begin() + static_cast<std::ptrdiff_t>(nvars));
            auto const length = std::uniform_int_distribution<std::size_t> { 5, 40 }(rng);
            auto tree = BalancedTreeCreator { &pset, vars, 0.0, 40 }(treeRng, length, 0, 40);
            auto present = std::vector<Operon::Hash> {};
            for (auto const& n : tree.Nodes()) {
                if (n.IsVariable() && std::ranges::find(present, n.HashValue) == present.end()) {
                    present.push_back(n.HashValue);
                }
            }
            if (present.empty()) {
                continue;
            }
            std::ranges::shuffle(present, rng);
            present.resize(
                std::uniform_int_distribution<std::size_t> { 1, std::min<std::size_t>(4, present.size()) }(rng));
            auto coeff = tree.GetCoefficients();
            auto const integerMode = std::uniform_real_distribution<double> { 0.0, 1.0 }(rng) < 0.25;
            for (auto& c : coeff) {
                auto v = std::uniform_real_distribution<double> { -2.0, 2.0 }(rng);
                if (integerMode) {
                    v = std::round(v);
                    v = v == 0.0 ? 1.0 : v;
                }
                c = static_cast<S>(v);
            }
            DomainMap domains;
            for (auto const h : vars) {
                domains.insert_or_assign(h, randomBox(rng));
            }
            ++made;

            auto const bounds = SOE { &tree, domains, present }(coeff);
            if (!bounds) {
                continue;
            }
            for (std::size_t k = 0; k < present.size(); ++k) {
                if (bounds->Status[k] != Status::Certain || bounds->Second[k].is_empty()) {
                    continue;
                }
                ++out.Certain;
                for (std::size_t s = 0; s < 16; ++s) {
                    Operon::Map<Operon::Hash, double> point;
                    for (auto const& [h, b] : domains) {
                        auto const u = std::uniform_real_distribution<double> { 0.15, 0.85 }(rng);
                        point[h] = static_cast<double>(b.first)
                            + (static_cast<double>(b.second) - static_cast<double>(b.first)) * u;
                    }
                    auto const [lo, hi] = domains.at(present[k]);
                    auto const x = point[present[k]];
                    auto const hstep = std::min(0.02 * std::max(1.0, std::abs(x)),
                        0.25 * std::min(x - static_cast<double>(lo), static_cast<double>(hi) - x));
                    if (!(hstep > 0.0)) {
                        continue;
                    }
                    bool regular = true;
                    bool rp = true;
                    bool rm = true;
                    auto const centre = Reference(tree, coeff, present, point, regular);
                    auto plus = point;
                    auto minus = point;
                    plus[present[k]] += hstep;
                    minus[present[k]] -= hstep;
                    auto const up = Reference(tree, coeff, present, plus, rp);
                    auto const dn = Reference(tree, coeff, present, minus, rm);
                    if (!regular || !rp || !rm) {
                        continue;
                    }
                    auto const fd = (up.a - 2.0 * centre.a + dn.a) / (hstep * hstep);
                    if (!std::isfinite(fd)) {
                        continue;
                    }
                    if (std::abs(centre.a) * 2.3e-16 / (hstep * hstep) > 0.02 * (1.0 + std::abs(fd))) {
                        continue; // cancellation noise of |f| could dominate the curvature signal
                    }
                    ++out.Checked;
                    auto const lower = static_cast<double>(bounds->Second[k].inf());
                    auto const upper = static_cast<double>(bounds->Second[k].sup());
                    auto const contained
                        = fd >= lower - 0.1 * (1.0 + std::abs(fd)) && fd <= upper + 0.1 * (1.0 + std::abs(fd));
                    if (!contained) {
                        ++out.Violations;
                        INFO("finite difference " << fd << " outside [" << lower << ", " << upper << "] of "
                                                  << fmt::format("{}", Operon::Fmt::TreeFormatArgs { tree, ds, 7 }));
                    }
                    CHECK(contained);
                }
            }
        }
        return out;
    }
} // namespace

TEST_CASE("IntervalSecondOrderJetEvaluator: certain second derivatives contain scalar finite differences over a "
          "deterministic corpus",
    "[second-order-jet]")
{
    auto const r = ScalarFiniteDifferenceSoundness(20260612, 600);
    INFO("certain " << r.Certain << ", checked " << r.Checked << ", violations " << r.Violations);
    CHECK(r.Violations == 0);
    CHECK(r.Checked > 0);
    CHECK(r.Certain > 0);
}

} // namespace Operon::Test
