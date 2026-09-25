// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "../operon_test.hpp"

#include "operon/core/dataset.hpp"
#include "operon/core/tree.hpp"
#include "operon/interpreter/interval_evaluator.hpp"
#include "operon/parser/infix.hpp"

namespace nb = ankerl::nanobench;

namespace {

using Scalar = Operon::Scalar;
using Interval = pappus::interval<Scalar>;
using DomainMap = Operon::IntervalEvaluator<Scalar>::DomainMap;

struct CorpusCase {
    std::string name;
    std::vector<std::pair<std::string, std::pair<Scalar, Scalar>>> domains;
    std::string expression;
};

// Standard dependency expressions, two published Feynman formulae, and two
// verbatim Operon GP trees exported from the shape-bound correctness corpus.
auto const kCorpus = std::vector<CorpusCase> {
    { "ia_x_squared_minus_x", { { "X1", { -1, 1 } } }, "X1 ^ 2 - X1" },
    { "ia_difference_square", { { "X1", { 0, 10 } }, { "X2", { 0, 10 } } }, "(X1 - X2) ^ 2" },
    { "ia_product_cancellation", { { "X1", { 0, 10 } }, { "X2", { 0, 10 } } }, "X1 * X2 - X1 * X2" },
    { "ia_difference_of_squares", { { "X1", { 0, 10 } }, { "X2", { 0, 10 } } }, "(X1 - X2) * (X1 + X2)" },
    { "feynman_gravitation", { { "G", { 1, 2 } }, { "m1", { 1, 2 } }, { "m2", { 1, 2 } }, { "x1", { 3, 4 } }, { "x2", { 1, 2 } }, { "y1", { 3, 4 } }, { "y2", { 1, 2 } }, { "z1", { 3, 4 } }, { "z2", { 1, 2 } } },
        "G * m1 * m2 / ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2)" },
    { "feynman_planck", { { "omega", { 1, 5 } }, { "T", { 1, 5 } }, { "h", { 1, 5 } }, { "kb", { 1, 5 } }, { "c", { 1, 5 } } },
        "h * omega ^ 3 / (3.141592653589793 ^ 2 * c ^ 2 * (exp(h * omega / (kb * T)) - 1))" },
    { "operon_gp_i_48_20", { { "c", { 3, 10 } }, { "v", { 1, 2 } }, { "m", { 1, 5 } } },
        "((-1.833329) + (0.854424 * ((((((-0.396240) * c) ^ 2) * (((2.610423 ^ 2) / cos((((-1.972425) * v) / (tanh((((0.572300 * v) / (tanh((0.199622 * c)) + ((-0.739473) * c))) ^ 2)) + ((-1.927949) * c))))) * (1.093432 * m))) + ((0.572300 * v) / ((tanh(((-0.739473) * c)) * (((tanh((-1.058995)) / ((-0.469899) * m)) + ((((-0.411143) * v) / ((-0.469899) * m)) / (0.199622 * c))) + ((0.572300 * v) / ((0.376507 * c) ^ 2)))) + ((-0.396240) * c)))) + (0.846632 / exp(tanh(((-0.739473) * c)))))))" },
    { "operon_gp_fuel_flow", { { "Astar", { 0.5, 1.5 } }, { "T0", { 250, 260 } }, { "p0", { 400000, 600000 } } },
        "((-0.026927) + (0.026880 * ((((tanh((cos((((0.449930 * Astar) / (1.366455 * T0)) * (6.154615 * p0))) / tanh(tanh(tanh(tanh(exp(((-0.760431) * T0)))))))) / ((cos(tanh(tanh(((-1.112441) * T0)))) / ((-0.836549) * T0)) * (5.931292 * p0))) + (((0.449930 * Astar) / (1.366455 * T0)) * (6.154615 * p0))) * (sqrt((1.354045 * T0)) * cos(cos(tanh((-0.544384)))))) + cos(tanh(tanh(tanh((((0.254493 * Astar) / cos(tanh(exp(((-0.760431) * T0))))) * 0.189909))))))))" },
};

struct Outcome {
    std::optional<Interval> bound;
    std::size_t evaluations {};
};

struct PreparedCase {
    Operon::Dataset dataset;
    Operon::Tree tree;
    DomainMap domains;
};

auto Prepare(CorpusCase const& c) -> PreparedCase
{
    std::vector<std::string> names;
    std::vector<std::vector<Scalar>> values;
    for (auto const& [name, range] : c.domains) {
        names.push_back(name);
        values.push_back({ range.first, range.second });
    }
    Operon::Dataset dataset(names, values);
    auto tree = Operon::InfixParser::ParseOrThrow(c.expression, dataset);
    DomainMap domains;
    for (auto const& [name, range] : c.domains) {
        domains.emplace(dataset.GetVariable(name)->Hash, range);
    }
    return { std::move(dataset), std::move(tree), std::move(domains) };
}

auto Evaluate(Operon::Tree const& tree, DomainMap const& domains, std::size_t& evaluations) -> std::optional<Interval>
{
    ++evaluations;
    try {
        Operon::IntervalEvaluator<Scalar> evaluator(&tree, domains);
        auto const bound = evaluator.Evaluate(tree.GetCoefficients());
        if (!std::isfinite(bound.inf()) || !std::isfinite(bound.sup())) {
            return std::nullopt;
        }
        return bound;
    } catch (std::exception const&) {
        return std::nullopt;
    }
}

auto ReferencedAxes(Operon::Tree const& tree, DomainMap const& domains) -> std::vector<Operon::Hash>
{
    std::vector<Operon::Hash> axes;
    for (auto const& node : tree.Nodes()) {
        if (!node.IsVariable() || !domains.contains(node.HashValue)
            || std::ranges::find(axes, node.HashValue) != axes.end()) {
            continue;
        }
        axes.push_back(node.HashValue);
    }
    return axes;
}

auto Occurrences(Operon::Tree const& tree, Operon::Hash axis) -> std::size_t
{
    return static_cast<std::size_t>(std::count_if(tree.Nodes().begin(), tree.Nodes().end(),
        [axis](auto const& node) { return node.IsVariable() && node.HashValue == axis; }));
}

auto Width(Interval const& bound) -> double
{
    return static_cast<double>(bound.sup()) - static_cast<double>(bound.inf());
}

// A schedule supplies one bisection axis per bit. Every strategy therefore
// spends exactly 2^depth leaves, even when a tree references many variables.
auto EvaluateGrid(Operon::Tree const& tree, DomainMap const& root, std::vector<Operon::Hash> const& axes,
    std::vector<std::size_t> const& schedule) -> Outcome
{
    std::size_t evaluations = 0;
    if (schedule.empty()) {
        return { Evaluate(tree, root, evaluations), evaluations };
    }

    std::vector<int> splits(axes.size());
    for (auto axis : schedule) {
        ++splits[axis];
    }

    std::optional<Interval> result;
    auto const leaves = std::size_t { 1 } << schedule.size();
    for (std::size_t leaf = 0; leaf < leaves; ++leaf) {
        DomainMap leafDomains = root;
        std::vector<int> cell(axes.size());
        std::vector<int> cellBits(axes.size());
        for (std::size_t bit = 0; bit < schedule.size(); ++bit) {
            auto const axis = schedule[bit];
            cell[axis] |= static_cast<int>((leaf >> bit) & 1U) << cellBits[axis]++;
        }
        for (std::size_t axis = 0; axis < axes.size(); ++axis) {
            auto const [lo, hi] = root.at(axes[axis]);
            auto const step = (hi - lo) / static_cast<Scalar>(std::size_t { 1 } << splits[axis]);
            leafDomains[axes[axis]] = { lo + static_cast<Scalar>(cell[axis]) * step,
                lo + static_cast<Scalar>(cell[axis] + 1) * step };
        }
        auto const bound = Evaluate(tree, leafDomains, evaluations);
        if (!bound) {
            return { std::nullopt, evaluations };
        }
        result = result ? (*result | *bound) : bound;
    }
    return { result, evaluations };
}

auto RepeatedAxisSchedule(Operon::Tree const& tree, DomainMap const& domains, int depth, bool occurrenceWeighted)
    -> std::vector<std::size_t>
{
    auto const axes = ReferencedAxes(tree, domains);
    if (axes.empty()) {
        return {};
    }
    auto score = [&](std::size_t axis) -> double {
        if (occurrenceWeighted) {
            return static_cast<double>(Occurrences(tree, axes[axis]));
        }
        auto const [lo, hi] = domains.at(axes[axis]);
        return static_cast<double>(hi - lo);
    };
    auto selected = std::size_t { 0 };
    for (std::size_t axis = 1; axis < axes.size(); ++axis) {
        if (score(axis) > score(selected)) {
            selected = axis;
        }
    }
    return std::vector<std::size_t>(static_cast<std::size_t>(depth), selected);
}

auto RoundRobinSchedule(Operon::Tree const& tree, DomainMap const& domains, int depth) -> std::vector<std::size_t>
{
    auto const axes = ReferencedAxes(tree, domains);
    std::vector<std::size_t> schedule;
    schedule.reserve(static_cast<std::size_t>(depth));
    for (int bit = 0; bit < depth && !axes.empty(); ++bit) {
        schedule.push_back(static_cast<std::size_t>(bit) % axes.size());
    }
    return schedule;
}

// Exhaustively scores one candidate schedule per axis at each added bit. It is
// an oracle-quality fixed-budget selector, deliberately included to quantify
// benefit before proposing a cheaper adaptive approximation.
auto GreedyGridSchedule(Operon::Tree const& tree, DomainMap const& domains, int depth)
    -> std::pair<std::vector<std::size_t>, std::size_t>
{
    auto const axes = ReferencedAxes(tree, domains);
    std::vector<std::size_t> schedule;
    std::size_t evaluations = 0;
    for (int bit = 0; bit < depth && !axes.empty(); ++bit) {
        auto bestAxis = std::size_t { 0 };
        auto bestWidth = std::numeric_limits<double>::infinity();
        for (std::size_t axis = 0; axis < axes.size(); ++axis) {
            auto candidate = schedule;
            candidate.push_back(axis);
            auto const outcome = EvaluateGrid(tree, domains, axes, candidate);
            evaluations += outcome.evaluations;
            if (outcome.bound && Width(*outcome.bound) < bestWidth) {
                bestWidth = Width(*outcome.bound);
                bestAxis = axis;
            }
        }
        schedule.push_back(bestAxis);
    }
    return { std::move(schedule), evaluations };
}

auto SimplifiedWidest(Operon::Tree const& tree, DomainMap const& domains, int depth) -> Outcome
{
    auto simplified = tree;
    simplified.Reduce().Simplify();
    auto const axes = ReferencedAxes(simplified, domains);
    return EvaluateGrid(simplified, domains, axes, RepeatedAxisSchedule(simplified, domains, depth, false));
}

struct Method {
    char const* name;
    std::function<Outcome(Operon::Tree const&, DomainMap const&, int)> run;
};

auto const kMethods = std::vector<Method> {
    { "widest", [](auto const& tree, auto const& domains, int depth) {
         auto const axes = ReferencedAxes(tree, domains);
         return EvaluateGrid(tree, domains, axes, RepeatedAxisSchedule(tree, domains, depth, false));
     } },
    { "occurrence", [](auto const& tree, auto const& domains, int depth) {
         auto const axes = ReferencedAxes(tree, domains);
         return EvaluateGrid(tree, domains, axes, RepeatedAxisSchedule(tree, domains, depth, true));
     } },
    { "round_robin", [](auto const& tree, auto const& domains, int depth) {
         auto const axes = ReferencedAxes(tree, domains);
         return EvaluateGrid(tree, domains, axes, RoundRobinSchedule(tree, domains, depth));
     } },
    { "greedy_grid", [](auto const& tree, auto const& domains, int depth) {
         auto const axes = ReferencedAxes(tree, domains);
         auto [schedule, probes] = GreedyGridSchedule(tree, domains, depth);
         auto outcome = EvaluateGrid(tree, domains, axes, schedule);
         outcome.evaluations += probes;
         return outcome;
     } },
    { "simplify_widest", [](auto const& tree, auto const& domains, int depth) {
         return SimplifiedWidest(tree, domains, depth);
     } },
};

void RunCase(nb::Bench& bench, CorpusCase const& corpus, int depth)
{
    auto prepared = Prepare(corpus);
    std::size_t directEvaluations = 0;
    auto const direct = Evaluate(prepared.tree, prepared.domains, directEvaluations);
    REQUIRE(direct);
    auto const directWidth = Width(*direct);

    fmt::print("\n{} nodes={} depth={} leaves={} direct=[{:.8g},{:.8g}] width={:.8g}\n",
        corpus.name, prepared.tree.Length(), depth, std::size_t { 1 } << depth,
        static_cast<double>(direct->inf()), static_cast<double>(direct->sup()), directWidth);
    for (auto const& method : kMethods) {
        Outcome latest;
        bench.run(fmt::format("{} {} d{}", corpus.name, method.name, depth), [&] {
            latest = method.run(prepared.tree, prepared.domains, depth);
            nb::doNotOptimizeAway(latest.evaluations);
        });
        REQUIRE(latest.bound);
        CHECK(latest.bound->inf() >= direct->inf());
        CHECK(latest.bound->sup() <= direct->sup());
        auto const width = Width(*latest.bound);
        auto const reduction = directWidth > 0 ? 100.0 * (directWidth - width) / directWidth : 0.0;
        auto const ns = bench.results().back().average(nb::Result::Measure::elapsed) * 1e9;
        fmt::print("  {:16s} width={:12.8g} reduction={:7.3f}% evals={:5d} time={:11.1f}ns\n",
            method.name, width, reduction, latest.evaluations, ns);
    }
}

} // namespace

TEST_CASE("Bisection strategy corpus: tightness and runtime", "[performance][shape-constraints][bisection][strategy]")
{
    nb::Bench bench;
    bench.title("bisection-strategy-corpus").batch(1);
    for (auto const& corpus : kCorpus) {
        RunCase(bench, corpus, 3);
        RunCase(bench, corpus, 6);
    }
    bench.render(nb::templates::csv(), std::cout);
}
