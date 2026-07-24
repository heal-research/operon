// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
//
// TightenRange / TightenRangeBisected validated against real physics
// formulas, not just synthetic hand-built trees or GP-evolved ones.
//
// The 19 synthetic problems are those used in Kronberger, de Franca,
// Burlacu, Haider & Kommenda, "Shape-constrained Symbolic Regression -
// Improving Extrapolation with Prior Knowledge" (arXiv:2103.15624): 16
// drawn from the AI Feynman Symbolic Regression Database (Udrescu &
// Tegmark, 2020) with their published variable domains, plus 3 fluid
// dynamics engineering formulas (Aircraft lift, Flow psi, Fuel flow) whose
// exact published domains are in that paper's supplementary material
// (not available in this session) - those three use an approximate,
// clearly-marked reconstruction instead. Several of the Feynman formulas
// have genuine repeated-variable structure (e.g. I.9.18's coordinate
// differences, I.15.3x/I.15.3t's repeated `u`, Flow psi's repeated `r`/`R`),
// making them good real-world dependency-problem test cases.

#include <catch2/catch_test_macros.hpp>

#include <random>
#include <string>
#include <vector>

#include "../operon_test.hpp"

#include "operon/core/dataset.hpp"
#include "operon/core/tree.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/interpreter/interval_evaluator.hpp"
#include "operon/interpreter/range_tightening.hpp"
#include "operon/parser/infix.hpp"

namespace Operon::Test {

namespace {

struct Var {
    std::string name;
    double lo;
    double hi;
};

struct Problem {
    std::string name;
    std::string formula; // operon infix syntax: ^ for pow, asin/acos, pi already substituted
    std::vector<Var> vars;
};

// clang-format off
auto const PI = 3.14159265358979323846;

auto Problems() -> std::vector<Problem> const&
{
    static std::vector<Problem> const problems{
        // --- FDE (fluid dynamics engineering); approximate domains, not
        // from the paper's supplementary material - see file header. ---
        {"Aircraft lift (approx.)",
         "C_La * (alpha + alpha0) + C_Lde * de * (SHT / Sref)",
         {{"C_La", 0.05, 0.15}, {"alpha", -0.1, 0.3}, {"alpha0", -0.1, 0.1},
          {"C_Lde", 0.01, 0.05}, {"de", -0.3, 0.3}, {"SHT", 1, 5}, {"Sref", 10, 30}}},
        {"Flow psi (approx.)",
         "Vinf * r * sin(theta / (2 * " + std::to_string(PI) + ")) * (1 - (R / r) ^ 2)"
         " + (Gamma / (2 * " + std::to_string(PI) + ")) * log(r / R)",
         {{"Vinf", 1, 5}, {"r", 2, 5}, {"theta", 0, 6.28318530718}, {"R", 1, 2}, {"Gamma", 1, 5}}},
        {"Fuel flow (approx., standard choked-nozzle form)",
         "p0 * Astar / sqrt(T0) * sqrt(gamma / Rgas) * (2 / (gamma + 1)) ^ ((gamma + 1) / (2 * (gamma - 1)))",
         {{"p0", 1, 5}, {"Astar", 0.01, 0.1}, {"T0", 250, 350}, {"gamma", 1.2, 1.4}, {"Rgas", 280, 300}}},

        // --- AI Feynman Symbolic Regression Database (exact formulas and
        // domains from FeynmanEquations.csv / BonusEquations.csv) ---
        {"I.6.2", "exp(((theta / sigma) ^ 2) / (-2)) / (sqrt(2 * " + std::to_string(PI) + ") * sigma)",
         {{"sigma", 1, 3}, {"theta", 1, 3}}},
        {"I.9.18", "G * m1 * m2 / ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2)",
         {{"m1", 1, 2}, {"m2", 1, 2}, {"G", 1, 2}, {"x1", 3, 4}, {"x2", 1, 2},
          {"y1", 3, 4}, {"y2", 1, 2}, {"z1", 3, 4}, {"z2", 1, 2}}},
        {"I.15.3x", "(x - u * t) / sqrt(1 - u ^ 2 / c ^ 2)",
         {{"x", 5, 10}, {"u", 1, 2}, {"c", 3, 20}, {"t", 1, 2}}},
        {"I.15.3t", "(t - u * x / c ^ 2) / sqrt(1 - u ^ 2 / c ^ 2)",
         {{"x", 1, 5}, {"c", 3, 10}, {"u", 1, 2}, {"t", 1, 5}}},
        {"I.30.5", "asin(lambd / (n * d))",
         {{"lambd", 1, 2}, {"d", 2, 5}, {"n", 1, 5}}},
        {"I.32.17", "(1 / 2 * epsilon * c * Ef ^ 2) * (8 * " + std::to_string(PI) + " * r ^ 2 / 3)"
         " * (omega ^ 4 / (omega ^ 2 - omega_0 ^ 2) ^ 2)",
         {{"epsilon", 1, 2}, {"c", 1, 2}, {"Ef", 1, 2}, {"r", 1, 2}, {"omega", 1, 2}, {"omega_0", 3, 5}}},
        {"I.41.16", "h / (2 * " + std::to_string(PI) + ") * omega ^ 3 / (" + std::to_string(PI) + " ^ 2 * c ^ 2"
         " * (exp((h / (2 * " + std::to_string(PI) + ")) * omega / (kb * T)) - 1))",
         {{"omega", 1, 5}, {"T", 1, 5}, {"h", 1, 5}, {"kb", 1, 5}, {"c", 1, 5}}},
        {"I.48.2", "m * c ^ 2 / sqrt(1 - v ^ 2 / c ^ 2)",
         {{"m", 1, 5}, {"v", 1, 2}, {"c", 3, 10}}},
        {"II.6.15a", "p_d / (4 * " + std::to_string(PI) + " * epsilon) * 3 * z / r ^ 5 * sqrt(x ^ 2 + y ^ 2)",
         {{"epsilon", 1, 3}, {"p_d", 1, 3}, {"r", 1, 3}, {"x", 1, 3}, {"y", 1, 3}, {"z", 1, 3}}},
        {"II.11.27", "n * alpha / (1 - (n * alpha / 3)) * epsilon * Ef",
         {{"n", 0, 1}, {"alpha", 0, 1}, {"epsilon", 1, 2}, {"Ef", 1, 2}}},
        {"II.11.28", "1 + n * alpha / (1 - (n * alpha / 3))",
         {{"n", 0, 1}, {"alpha", 0, 1}}},
        {"II.35.21", "n_rho * mom * tanh(mom * B / (kb * T))",
         {{"n_rho", 1, 5}, {"mom", 1, 5}, {"B", 1, 5}, {"kb", 1, 5}, {"T", 1, 5}}},
        {"III.9.52", "(p_d * Ef * t / (h / (2 * " + std::to_string(PI) + ")))"
         " * sin((omega - omega_0) * t / 2) ^ 2 / ((omega - omega_0) * t / 2) ^ 2",
         {{"p_d", 1, 3}, {"Ef", 1, 3}, {"t", 1, 3}, {"h", 1, 3}, {"omega", 1, 5}, {"omega_0", 1, 5}}},
        {"III.10.19", "mom * sqrt(Bx ^ 2 + By ^ 2 + Bz ^ 2)",
         {{"mom", 1, 5}, {"Bx", 1, 5}, {"By", 1, 5}, {"Bz", 1, 5}}},
        {"Jackson 2.11", "q / (4 * " + std::to_string(PI) + " * epsilon * y ^ 2)"
         " * (4 * " + std::to_string(PI) + " * epsilon * Volt * d - q * d * y ^ 3 / (y ^ 2 - d ^ 2) ^ 2)",
         {{"q", 1, 5}, {"y", 1, 3}, {"Volt", 1, 5}, {"d", 4, 6}, {"epsilon", 1, 5}}},
        {"Wave power", "-32.0 / 5 * G ^ 4 / c ^ 5 * (m1 * m2) ^ 2 * (m1 + m2) / r ^ 5",
         {{"G", 1, 2}, {"c", 1, 2}, {"m1", 1, 5}, {"m2", 1, 5}, {"r", 1, 2}}},
    };
    return problems;
}
// clang-format on

} // namespace

TEST_CASE("Feynman benchmark suite - TightenRange/TightenRangeBisected soundness and tightness", "[range_tightening][feynman]")
{
    using DTable = DispatchTable<Operon::Scalar>;
    using Interp = Interpreter<Operon::Scalar, DTable>;
    DTable dtable;

    constexpr int nSamples = 200;
    constexpr auto tol = 1e-3F;
    Operon::RandomGenerator rng(2103'15624UL);

    std::size_t totalSoundnessChecks = 0;
    std::size_t totalViolations = 0;

    for (auto const& p : Problems()) {
        INFO("problem: " << p.name);

        std::vector<std::string> names;
        names.reserve(p.vars.size());
        for (auto const& v : p.vars) { names.push_back(v.name); }
        std::vector<std::vector<Operon::Scalar>> dummyData(
            p.vars.size(), std::vector<Operon::Scalar>(1, Operon::Scalar{1}));
        Dataset const ds(names, dummyData);

        auto tree = Operon::InfixParser::Parse(p.formula, ds);
        auto const coeff = tree.GetCoefficients();

        IntervalEvaluator::DomainMap domains;
        for (auto const& v : p.vars) {
            auto const hash = ds.GetVariable(v.name)->Hash;
            domains[hash] = {static_cast<Operon::Scalar>(v.lo), static_cast<Operon::Scalar>(v.hi)};
        }

        auto const naive     = IntervalEvaluator(&tree, domains).Evaluate(coeff);
        auto const tightened = TightenRange(tree, domains, coeff);
        auto const bisected  = TightenRangeBisected(tree, domains, coeff, 3);

        if (naive.is_empty()) {
            // A pre-existing IntervalEvaluator limitation, not a
            // TightenRange bug: Pow always dispatches through the general
            // interval^interval overload, which domain-restricts the base
            // to non-negative even when the exponent is a constant integer
            // (e.g. squaring an always-negative subtraction like
            // (omega^2 - omega_0^2)^2, mathematically fine, but rejected
            // here). Affects Jackson 2.11 and I.32.17 in this suite.
            fmt::print("{:<40} naive interval empty (Pow domain limitation, not a TightenRange bug)\n", p.name);
            continue;
        }

        auto const naiveWidth = naive.diameter();
        if (std::isfinite(naiveWidth) && naiveWidth > 0) {
            auto const tightPct = 100.0 * (1.0 - static_cast<double>(tightened.diameter()) / static_cast<double>(naiveWidth));
            auto const bisPct   = 100.0 * (1.0 - static_cast<double>(bisected.diameter()) / static_cast<double>(naiveWidth));
            fmt::print("{:<40} naive=[{:.4g},{:.4g}] tightened: {:5.1f}%  bisected: {:5.1f}%\n",
                p.name, naive.inf(), naive.sup(), tightPct, bisPct);
        } else {
            fmt::print("{:<40} naive=[{:.4g},{:.4g}] (unbounded)\n", p.name, naive.inf(), naive.sup());
        }

        // TightenRange/TightenRangeBisected must never be looser than naive.
        CHECK(tightened.inf() >= naive.inf() - tol);
        CHECK(tightened.sup() <= naive.sup() + tol);
        CHECK(bisected.inf() >= tightened.inf() - tol);
        CHECK(bisected.sup() <= tightened.sup() + tol);

        if (bisected.is_empty()) { continue; } // domain edge somewhere in the formula; nothing to sample

        // Soundness: dense random point sampling must never fall outside
        // the tightened enclosure.
        std::vector<std::vector<Operon::Scalar>> pointData(p.vars.size(), std::vector<Operon::Scalar>(nSamples));
        for (std::size_t vi = 0; vi < p.vars.size(); ++vi) {
            std::uniform_real_distribution<Operon::Scalar> pd(
                static_cast<Operon::Scalar>(p.vars[vi].lo), static_cast<Operon::Scalar>(p.vars[vi].hi));
            for (auto& v : pointData[vi]) { v = pd(rng); }
        }
        Dataset const pointDs(names, pointData);
        Range const range{0, nSamples};
        auto const values = Interp::Evaluate(tree, pointDs, range, Operon::Span<Operon::Scalar const>(coeff));

        for (auto v : values) {
            if (!std::isfinite(v)) { continue; }
            ++totalSoundnessChecks;
            if (v < bisected.inf() - tol || v > bisected.sup() + tol) { ++totalViolations; }
        }
    }

    INFO("total soundness violations: " << totalViolations << " / " << totalSoundnessChecks);
    CHECK(totalViolations == 0);
    CHECK(totalSoundnessChecks > 0);
}

} // namespace Operon::Test
