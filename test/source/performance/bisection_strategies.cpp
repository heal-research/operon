// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "../operon_test.hpp"

#include "operon/core/dataset.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/tree_diff.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/interpreter/interval_evaluator.hpp"
#include "operon/operators/creator.hpp"
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
    { "feynman_gravitation",
        { { "G", { 1, 2 } }, { "m1", { 1, 2 } }, { "m2", { 1, 2 } }, { "x1", { 3, 4 } }, { "x2", { 1, 2 } },
            { "y1", { 3, 4 } }, { "y2", { 1, 2 } }, { "z1", { 3, 4 } }, { "z2", { 1, 2 } } },
        "G * m1 * m2 / ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2)" },
    { "feynman_planck",
        { { "omega", { 1, 5 } }, { "T", { 1, 5 } }, { "h", { 1, 5 } }, { "kb", { 1, 5 } }, { "c", { 1, 5 } } },
        "h * omega ^ 3 / (3.141592653589793 ^ 2 * c ^ 2 * (exp(h * omega / (kb * T)) - 1))" },
    { "operon_gp_i_48_20", { { "c", { 3, 10 } }, { "v", { 1, 2 } }, { "m", { 1, 5 } } },
        "((-1.833329) + (0.854424 * ((((((-0.396240) * c) ^ 2) * (((2.610423 ^ 2) / cos((((-1.972425) * v) / "
        "(tanh((((0.572300 * v) / (tanh((0.199622 * c)) + ((-0.739473) * c))) ^ 2)) + ((-1.927949) * c))))) * "
        "(1.093432 * m))) + ((0.572300 * v) / ((tanh(((-0.739473) * c)) * (((tanh((-1.058995)) / ((-0.469899) * m)) + "
        "((((-0.411143) * v) / ((-0.469899) * m)) / (0.199622 * c))) + ((0.572300 * v) / ((0.376507 * c) ^ 2)))) + "
        "((-0.396240) * c)))) + (0.846632 / exp(tanh(((-0.739473) * c)))))))" },
    { "operon_gp_fuel_flow", { { "Astar", { 0.5, 1.5 } }, { "T0", { 250, 260 } }, { "p0", { 400000, 600000 } } },
        "((-0.026927) + (0.026880 * ((((tanh((cos((((0.449930 * Astar) / (1.366455 * T0)) * (6.154615 * p0))) / "
        "tanh(tanh(tanh(tanh(exp(((-0.760431) * T0)))))))) / ((cos(tanh(tanh(((-1.112441) * T0)))) / ((-0.836549) * "
        "T0)) * (5.931292 * p0))) + (((0.449930 * Astar) / (1.366455 * T0)) * (6.154615 * p0))) * (sqrt((1.354045 * "
        "T0)) * cos(cos(tanh((-0.544384)))))) + cos(tanh(tanh(tanh((((0.254493 * Astar) / cos(tanh(exp(((-0.760431) * "
        "T0))))) * 0.189909))))))))" },
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
    Operon::IntervalEvaluator<Scalar> evaluator(&tree, domains);
    auto const bound = evaluator(tree.GetCoefficients());
    if (!bound || !std::isfinite(bound->inf()) || !std::isfinite(bound->sup())) {
        return std::nullopt;
    }
    return *bound;
}

auto ReferencedAxes(Operon::Tree const& tree, DomainMap const& domains) -> std::vector<Operon::Hash>
{
    std::vector<Operon::Hash> axes;
    for (auto const& node : tree.Nodes()) {
        if (!node.IsVariable() || !domains.contains(node.HashValue)
            || std::ranges::find(axes, node.HashValue) != axes.end()) {
            continue;
        }
        auto const [lo, hi] = domains.at(node.HashValue);
        if (hi <= lo) {
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
            leafDomains[axes[axis]]
                = { lo + static_cast<Scalar>(cell[axis]) * step, lo + static_cast<Scalar>(cell[axis] + 1) * step };
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

// The production level allocation (IntervalSubdivisionPlan::Make, source/operators/shape_constrained_evaluator.cpp):
// each level bisects the currently widest axis, the strict '>' lets the axis that comes first in `order` win ties,
// and the selected axis's width is halved. `order` is the axis encounter order (identity in production).
auto WidestFirstSplits(std::vector<double> score, std::span<std::size_t const> order, int depth) -> std::vector<int>
{
    std::vector<int> splits(score.size());
    if (order.empty()) {
        return splits;
    }
    for (int level = 0; level < depth; ++level) {
        auto selected = order.front();
        for (auto axis : order.subspan(1)) {
            if (score[axis] > score[selected]) {
                selected = axis;
            }
        }
        ++splits[selected];
        score[selected] /= 2.0;
    }
    return splits;
}

auto SplitsToSchedule(std::vector<int> const& splits) -> std::vector<std::size_t>
{
    std::vector<std::size_t> schedule;
    for (std::size_t axis = 0; axis < splits.size(); ++axis) {
        schedule.insert(schedule.end(), static_cast<std::size_t>(splits[axis]), axis);
    }
    return schedule;
}

// Production allocation, optionally with the axis encounter order reversed (the only difference: ties go to the
// axis that appears last in the tree instead of first).
auto ProductionOutcome(Operon::Tree const& tree, DomainMap const& domains, int depth, bool reversed) -> Outcome
{
    auto const axes = ReferencedAxes(tree, domains);
    std::vector<double> widths;
    for (auto axis : axes) {
        auto const [lo, hi] = domains.at(axis);
        widths.push_back(static_cast<double>(hi - lo));
    }
    std::vector<std::size_t> order(axes.size());
    std::iota(order.begin(), order.end(), std::size_t { 0 });
    if (reversed) {
        std::ranges::reverse(order);
    }
    return EvaluateGrid(tree, domains, axes, SplitsToSchedule(WidestFirstSplits(widths, order, depth)));
}

struct Method {
    char const* name;
    std::function<Outcome(Operon::Tree const&, DomainMap const&, int)> run;
};

auto const kMethods = std::vector<Method> {
    { "widest",
        [](auto const& tree, auto const& domains, int depth) {
            auto const axes = ReferencedAxes(tree, domains);
            return EvaluateGrid(tree, domains, axes, RepeatedAxisSchedule(tree, domains, depth, false));
        } },
    { "occurrence",
        [](auto const& tree, auto const& domains, int depth) {
            auto const axes = ReferencedAxes(tree, domains);
            return EvaluateGrid(tree, domains, axes, RepeatedAxisSchedule(tree, domains, depth, true));
        } },
    { "round_robin",
        [](auto const& tree, auto const& domains, int depth) {
            auto const axes = ReferencedAxes(tree, domains);
            return EvaluateGrid(tree, domains, axes, RoundRobinSchedule(tree, domains, depth));
        } },
    { "greedy_grid",
        [](auto const& tree, auto const& domains, int depth) {
            auto const axes = ReferencedAxes(tree, domains);
            auto [schedule, probes] = GreedyGridSchedule(tree, domains, depth);
            auto outcome = EvaluateGrid(tree, domains, axes, schedule);
            outcome.evaluations += probes;
            return outcome;
        } },
    { "production",
        [](auto const& tree, auto const& domains, int depth) {
            return ProductionOutcome(tree, domains, depth, false);
        } },
    { "production_rev",
        [](auto const& tree, auto const& domains, int depth) {
            return ProductionOutcome(tree, domains, depth, true);
        } },
    { "simplify_widest",
        [](auto const& tree, auto const& domains, int depth) { return SimplifiedWidest(tree, domains, depth); } },
};

void RunCase(nb::Bench& bench, CorpusCase const& corpus, int depth)
{
    auto prepared = Prepare(corpus);
    std::size_t directEvaluations = 0;
    auto const direct = Evaluate(prepared.tree, prepared.domains, directEvaluations);
    REQUIRE(direct);
    auto const directWidth = Width(*direct);

    fmt::print("\n{} nodes={} depth={} leaves={} direct=[{:.8g},{:.8g}] width={:.8g}\n", corpus.name,
        prepared.tree.Length(), depth, std::size_t { 1 } << depth, static_cast<double>(direct->inf()),
        static_cast<double>(direct->sup()), directWidth);
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
        fmt::print("  {:16s} width={:12.8g} reduction={:7.3f}% evals={:5d} time={:11.1f}ns\n", method.name, width,
            reduction, latest.evaluations, ns);
    }
}

// ===========================================================================
// Exhaustive allocation oracle
//
// For k bisection axes and depth d, every allocation of the d splits across the axes (a composition of d into k
// non-negative parts, C(d+k-1,k-1) of them) is evaluated; the order of bits within an axis does not change the grid.
// The oracle is a measuring stick only: it costs C(d+k-1,k-1) * 2^d leaf evaluations.
// ===========================================================================

enum class BoundKind : std::uint8_t { Value, FirstDerivative, SecondDerivative };

constexpr std::array kBoundKinds { BoundKind::Value, BoundKind::FirstDerivative, BoundKind::SecondDerivative };
constexpr int kMinOracleDepth = 2;
constexpr int kMaxOracleDepth = 6;
// Skip the oracle at a depth when it would need more leaf evaluations than this.
constexpr std::size_t kOracleLeafBudget = 4'000'000;
// Regret at or below this counts as "optimal" (float32 grids that differ only by rounding are ties).
constexpr double kOptimalTolerance = 1e-3;
constexpr std::size_t kThresholdsPerSide = 16;

auto KindName(BoundKind kind) -> char const*
{
    switch (kind) {
    case BoundKind::Value:
        return "value";
    case BoundKind::FirstDerivative:
        return "d1";
    case BoundKind::SecondDerivative:
        return "d2";
    }
    return "?";
}

struct Axis {
    Operon::Hash Hash {};
    Scalar Lo {};
    Scalar Hi {};
    Scalar Width {};
    double Occurrences {}; // in the expanded expression (Ref nodes expanded)
    double RawOccurrences {}; // Variable nodes in the node array (a shared subtree counts once)
};

// Number of occurrences of every axis in the fully expanded expression (a Ref node counts as the subtree it
// points at, so a DAG-shared variable counts once per use).
void FillOccurrences(Operon::Tree const& tree, std::vector<Axis>& axes)
{
    auto const& nodes = tree.Nodes();
    auto const k = axes.size();
    if (nodes.empty() || k == 0) {
        return;
    }
    std::vector<double> counts(nodes.size() * k, 0.0);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        auto const& node = nodes[i];
        auto* out = counts.data() + (i * k);
        if (node.IsRef()) {
            std::copy_n(counts.data() + (static_cast<std::size_t>(node.RefTo) * k), k, out);
        } else if (node.IsVariable()) {
            auto const it = std::ranges::find(axes, node.HashValue, &Axis::Hash);
            if (it != axes.end()) {
                out[static_cast<std::size_t>(it - axes.begin())] = 1.0;
            }
        } else if (!node.IsLeaf()) {
            auto child = i - 1;
            for (std::size_t a = 0; a < node.Arity; ++a) {
                for (std::size_t j = 0; j < k; ++j) {
                    out[j] += counts[(child * k) + j];
                }
                if (a + 1 < node.Arity) {
                    child -= static_cast<std::size_t>(nodes[child].Length) + 1;
                }
            }
        }
    }
    auto const* root = counts.data() + ((nodes.size() - 1) * k);
    for (std::size_t j = 0; j < k; ++j) {
        axes[j].Occurrences = root[j];
    }
}

// The axes IntervalSubdivisionPlan::Make would build: variables in order of first appearance, zero-width excluded.
auto BisectionAxes(Operon::Tree const& tree, DomainMap const& domains) -> std::vector<Axis>
{
    std::vector<Axis> axes;
    for (auto const& node : tree.Nodes()) {
        if (!node.IsVariable() || std::ranges::find(axes, node.HashValue, &Axis::Hash) != axes.end()) {
            continue;
        }
        auto const it = domains.find(node.HashValue);
        if (it == domains.end()) {
            continue;
        }
        auto const width = it->second.second - it->second.first;
        if (width <= Scalar { 0 }) {
            continue;
        }
        axes.push_back({ node.HashValue, it->second.first, it->second.second, width, 0.0, 0.0 });
    }
    FillOccurrences(tree, axes);
    for (auto const& node : tree.Nodes()) {
        if (node.IsVariable()) {
            if (auto const it = std::ranges::find(axes, node.HashValue, &Axis::Hash); it != axes.end()) {
                it->RawOccurrences += 1.0;
            }
        }
    }
    return axes;
}

// Memoized grid bound for an allocation (splits per axis). One IntervalEvaluator is reused; the per-leaf sub-box is
// supplied through LaneOverride so no DomainMap is copied.
class GridEvaluator {
public:
    using Evaluator = Operon::IntervalEvaluator<Scalar>;

    GridEvaluator(Operon::Tree const& tree, DomainMap const& domains, std::vector<Axis> const& axes)
        : coeff_(tree.GetCoefficients())
        , evaluator_(&tree, domains)
        , axes_(&axes)
        , lo_(axes.size())
        , hi_(axes.size())
    {
        for (std::size_t i = 0; i < axes.size(); ++i) {
            overrides_.push_back({ axes[i].Hash, &lo_[i], &hi_[i] });
        }
    }
    GridEvaluator(GridEvaluator const&) = delete;
    auto operator=(GridEvaluator const&) -> GridEvaluator& = delete;

    auto Bound(std::vector<int> const& splits) -> std::optional<Interval>
    {
        if (auto const it = memo_.find(splits); it != memo_.end()) {
            return it->second;
        }
        auto result = Compute(splits);
        memo_.emplace(splits, result);
        return result;
    }

    [[nodiscard]] auto Leaves() const -> std::size_t { return leaves_; }

private:
    auto Compute(std::vector<int> const& splits) -> std::optional<Interval>
    {
        auto const k = axes_->size();
        std::vector<std::size_t> counts(k);
        std::vector<std::size_t> cells(k, 0);
        std::size_t nLeaves = 1;
        for (std::size_t a = 0; a < k; ++a) {
            counts[a] = std::size_t { 1 } << splits[a];
            nLeaves *= counts[a];
        }
        auto inf = std::numeric_limits<Scalar>::infinity();
        auto sup = -std::numeric_limits<Scalar>::infinity();
        for (std::size_t leaf = 0; leaf < nLeaves; ++leaf) {
            for (std::size_t a = 0; a < k; ++a) {
                auto const& axis = (*axes_)[a];
                auto const step = axis.Width / static_cast<Scalar>(counts[a]);
                lo_[a] = cells[a] == 0 ? axis.Lo : axis.Lo + (static_cast<Scalar>(cells[a]) * step);
                hi_[a] = cells[a] + 1 == counts[a] ? axis.Hi : axis.Lo + (static_cast<Scalar>(cells[a] + 1) * step);
            }
            ++leaves_;
            auto const bound = evaluator_(coeff_, overrides_);
            if (!bound || bound->is_empty() || !std::isfinite(bound->inf()) || !std::isfinite(bound->sup())) {
                return std::nullopt;
            }
            inf = std::min(inf, bound->inf());
            sup = std::max(sup, bound->sup());
            for (std::size_t a = 0; a < k; ++a) {
                if (++cells[a] < counts[a]) {
                    break;
                }
                cells[a] = 0;
            }
        }
        return Interval(inf, sup);
    }

    std::vector<Scalar> coeff_;
    Evaluator evaluator_;
    std::vector<Axis> const* axes_;
    std::vector<Scalar> lo_;
    std::vector<Scalar> hi_;
    std::vector<Evaluator::LaneOverride> overrides_;
    std::map<std::vector<int>, std::optional<Interval>> memo_;
    std::size_t leaves_ {};
};

void EnumerateCompositions(
    std::vector<int>& current, std::size_t axis, int remaining, std::vector<std::vector<int>>& out)
{
    if (axis + 1 == current.size()) {
        current[axis] = remaining;
        out.push_back(current);
        return;
    }
    for (int n = 0; n <= remaining; ++n) {
        current[axis] = n;
        EnumerateCompositions(current, axis + 1, remaining - n, out);
    }
}

auto AllocationCount(std::size_t k, int depth) -> std::size_t
{
    // C(depth + k - 1, k - 1)
    std::size_t result = 1;
    for (std::size_t i = 1; i < k; ++i) {
        result = result * (static_cast<std::size_t>(depth) + i) / i;
    }
    return result;
}

// ---- allocation strategies -------------------------------------------------------------------------------------

struct StrategyContext {
    std::vector<Axis> const& Axes;
    std::optional<std::size_t> DiffAxis; // axis index of the differentiated variable, if it is an axis
    std::vector<double> const& Smear; // width * max|d/dx_axis| at the root box; empty when unavailable
    GridEvaluator& Grid;
    int Depth;
};

struct Allocation {
    std::vector<int> Splits;
    std::size_t ProbeLeaves {}; // extra leaf evaluations spent choosing the allocation
    std::size_t AuxEvaluations {}; // extra whole-tree evaluations (gradient trees) spent choosing it
};

struct StrategySpec {
    char const* Name;
    bool Cheap; // a fixed rule (no grid probing)
    std::function<Allocation(StrategyContext const&)> Allocate;
};

auto AxisWidths(std::vector<Axis> const& axes) -> std::vector<double>
{
    std::vector<double> widths;
    widths.reserve(axes.size());
    for (auto const& axis : axes) {
        widths.push_back(static_cast<double>(axis.Width));
    }
    return widths;
}

auto IdentityOrder(std::size_t n) -> std::vector<std::size_t>
{
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t { 0 });
    return order;
}

auto ArgMax(std::vector<double> const& values) -> std::size_t
{
    std::size_t selected = 0;
    for (std::size_t i = 1; i < values.size(); ++i) {
        if (values[i] > values[selected]) {
            selected = i;
        }
    }
    return selected;
}

auto ProductionAllocation(StrategyContext const& c, bool reversed) -> Allocation
{
    auto order = IdentityOrder(c.Axes.size());
    if (reversed) {
        std::ranges::reverse(order);
    }
    return { WidestFirstSplits(AxisWidths(c.Axes), order, c.Depth), 0, 0 };
}

// Greedy grid probing for the first `probeLevels` levels (every axis is tried at each level and the one giving the
// narrowest grid bound wins); any remaining levels are allocated widest-first on the post-probing widths.
auto GreedyProbeAllocation(StrategyContext const& c, int probeLevels) -> Allocation
{
    Allocation result;
    result.Splits.assign(c.Axes.size(), 0);
    auto const probed = std::min(probeLevels, c.Depth);
    for (int level = 1; level <= probed; ++level) {
        std::size_t best = 0;
        auto bestWidth = std::numeric_limits<double>::infinity();
        for (std::size_t axis = 0; axis < c.Axes.size(); ++axis) {
            auto candidate = result.Splits;
            ++candidate[axis];
            result.ProbeLeaves += std::size_t { 1 } << level;
            auto const bound = c.Grid.Bound(candidate);
            if (bound && Width(*bound) < bestWidth) {
                bestWidth = Width(*bound);
                best = axis;
            }
        }
        ++result.Splits[best];
    }
    if (probed < c.Depth) {
        auto score = AxisWidths(c.Axes);
        for (std::size_t axis = 0; axis < score.size(); ++axis) {
            score[axis] /= static_cast<double>(std::size_t { 1 } << result.Splits[axis]);
        }
        auto const rest = WidestFirstSplits(score, IdentityOrder(c.Axes.size()), c.Depth - probed);
        for (std::size_t axis = 0; axis < rest.size(); ++axis) {
            result.Splits[axis] += rest[axis];
        }
    }
    return result;
}

// Score vector for the "dependency" rules: variables that occur at most once cannot suffer the dependency
// problem; they are masked out unless every axis occurs at most once.
auto MaskSingleOccurrence(std::vector<double> score, std::vector<Axis> const& axes) -> std::vector<double>
{
    if (!std::ranges::any_of(axes, [](auto const& a) { return a.Occurrences > 1.0; })) {
        return score;
    }
    for (std::size_t i = 0; i < score.size(); ++i) {
        if (axes[i].Occurrences <= 1.0) {
            score[i] = -std::numeric_limits<double>::infinity();
        }
    }
    return score;
}

auto MakeStrategies() -> std::vector<StrategySpec>
{
    std::vector<StrategySpec> strategies;
    // Production: widest axis first, ties to the first-encountered axis.
    strategies.push_back(
        { "production", true, [](StrategyContext const& c) { return ProductionAllocation(c, false); } });
    // Production with the encounter order reversed: isolates the arbitrary tie-break.
    strategies.push_back(
        { "production_rev", true, [](StrategyContext const& c) { return ProductionAllocation(c, true); } });
    // (a) the differentiated variable gets every split (production when no differentiated axis exists).
    strategies.push_back({ "diff_all", true, [](StrategyContext const& c) {
                              if (!c.DiffAxis) {
                                  return ProductionAllocation(c, false);
                              }
                              Allocation result;
                              result.Splits.assign(c.Axes.size(), 0);
                              result.Splits[*c.DiffAxis] = c.Depth;
                              return result;
                          } });
    // (a') the differentiated variable gets the first ceil(d/2) splits, production allocates the rest.
    strategies.push_back({ "diff_half", true, [](StrategyContext const& c) {
                              if (!c.DiffAxis) {
                                  return ProductionAllocation(c, false);
                              }
                              auto const head = (c.Depth + 1) / 2;
                              auto score = AxisWidths(c.Axes);
                              score[*c.DiffAxis] /= static_cast<double>(std::size_t { 1 } << head);
                              Allocation result;
                              result.Splits = WidestFirstSplits(score, IdentityOrder(c.Axes.size()), c.Depth - head);
                              result.Splits[*c.DiffAxis] += head;
                              return result;
                          } });
    // (b) variables that occur more than once get priority; widest-first among them (all axes if none repeats).
    strategies.push_back(
        { "multi_occ", true, [](StrategyContext const& c) {
             auto const score = MaskSingleOccurrence(AxisWidths(c.Axes), c.Axes);
             return Allocation { WidestFirstSplits(score, IdentityOrder(c.Axes.size()), c.Depth), 0, 0 };
         } });
    // (b') dependency weighting: widest-first on width * (occurrences - 1), single-occurrence axes masked.
    strategies.push_back(
        { "dep_weighted", true, [](StrategyContext const& c) {
             auto score = AxisWidths(c.Axes);
             if (std::ranges::any_of(c.Axes, [](auto const& a) { return a.Occurrences > 1.0; })) {
                 for (std::size_t i = 0; i < score.size(); ++i) {
                     score[i] *= std::max(c.Axes[i].Occurrences - 1.0, 0.0);
                 }
             }
             return Allocation { WidestFirstSplits(score, IdentityOrder(c.Axes.size()), c.Depth), 0, 0 };
         } });
    // (c) occurrence-weighted width: widest-first on width * occurrences.
    strategies.push_back(
        { "occ_weighted", true, [](StrategyContext const& c) {
             auto score = AxisWidths(c.Axes);
             for (std::size_t i = 0; i < score.size(); ++i) {
                 score[i] *= c.Axes[i].Occurrences;
             }
             return Allocation { WidestFirstSplits(score, IdentityOrder(c.Axes.size()), c.Depth), 0, 0 };
         } });
    // (e) smear (Kearfott): widest-first on width * max|df/dx| over the root box; costs one gradient-tree
    // evaluation per axis. Falls back to production when a gradient bound is unavailable.
    strategies.push_back(
        { "smear", true, [](StrategyContext const& c) {
             if (c.Smear.empty()) {
                 return ProductionAllocation(c, false);
             }
             Allocation result { WidestFirstSplits(c.Smear, IdentityOrder(c.Axes.size()), c.Depth), 0, c.Axes.size() };
             return result;
         } });
    // dependency weighting by the raw Variable-node count (no Ref expansion): is the DAG expansion needed?
    strategies.push_back(
        { "dep_raw", true, [](StrategyContext const& c) {
             auto score = AxisWidths(c.Axes);
             if (std::ranges::any_of(c.Axes, [](auto const& a) { return a.RawOccurrences > 1.0; })) {
                 for (std::size_t i = 0; i < score.size(); ++i) {
                     score[i] *= std::max(c.Axes[i].RawOccurrences - 1.0, 0.0);
                 }
             }
             return Allocation { WidestFirstSplits(score, IdentityOrder(c.Axes.size()), c.Depth), 0, 0 };
         } });
    // dependency weighting squared: width * (occurrences - 1)^2.
    strategies.push_back(
        { "dep_sq", true, [](StrategyContext const& c) {
             auto score = AxisWidths(c.Axes);
             if (std::ranges::any_of(c.Axes, [](auto const& a) { return a.Occurrences > 1.0; })) {
                 for (std::size_t i = 0; i < score.size(); ++i) {
                     auto const extra = std::max(c.Axes[i].Occurrences - 1.0, 0.0);
                     score[i] *= extra * extra;
                 }
             }
             return Allocation { WidestFirstSplits(score, IdentityOrder(c.Axes.size()), c.Depth), 0, 0 };
         } });
    // smear weighted by (occurrences - 1): sensitivity times dependency.
    strategies.push_back(
        { "smear_dep", true, [](StrategyContext const& c) {
             if (c.Smear.empty()) {
                 return ProductionAllocation(c, false);
             }
             auto score = c.Smear;
             if (std::ranges::any_of(c.Axes, [](auto const& a) { return a.Occurrences > 1.0; })) {
                 for (std::size_t i = 0; i < score.size(); ++i) {
                     score[i] *= std::max(c.Axes[i].Occurrences - 1.0, 0.0);
                 }
             }
             return Allocation { WidestFirstSplits(score, IdentityOrder(c.Axes.size()), c.Depth), 0, c.Axes.size() };
         } });
    // smear restricted to axes that occur more than once.
    strategies.push_back(
        { "smear_multi", true, [](StrategyContext const& c) {
             if (c.Smear.empty()) {
                 return ProductionAllocation(c, false);
             }
             auto const score = MaskSingleOccurrence(c.Smear, c.Axes);
             return Allocation { WidestFirstSplits(score, IdentityOrder(c.Axes.size()), c.Depth), 0, c.Axes.size() };
         } });
    // (d) greedy grid probing (the existing probe-based strategy), and a variant that only probes the first 3 levels.
    strategies.push_back(
        { "greedy_probe", false, [](StrategyContext const& c) { return GreedyProbeAllocation(c, 99); } });
    strategies.push_back(
        { "greedy_probe3", false, [](StrategyContext const& c) { return GreedyProbeAllocation(c, 3); } });
    strategies.push_back({ "round_robin", true, [](StrategyContext const& c) {
                              Allocation result;
                              result.Splits.assign(c.Axes.size(), 0);
                              for (int bit = 0; bit < c.Depth; ++bit) {
                                  ++result.Splits[static_cast<std::size_t>(bit) % c.Axes.size()];
                              }
                              return result;
                          } });
    strategies.push_back({ "widest_all", true, [](StrategyContext const& c) {
                              Allocation result;
                              result.Splits.assign(c.Axes.size(), 0);
                              result.Splits[ArgMax(AxisWidths(c.Axes))] = c.Depth;
                              return result;
                          } });
    strategies.push_back({ "occ_all", true, [](StrategyContext const& c) {
                              std::vector<double> occ;
                              for (auto const& axis : c.Axes) {
                                  occ.push_back(axis.Occurrences);
                              }
                              Allocation result;
                              result.Splits.assign(c.Axes.size(), 0);
                              result.Splits[ArgMax(occ)] = c.Depth;
                              return result;
                          } });
    return strategies;
}

auto const kStrategies = MakeStrategies();

auto StrategyIndex(char const* name) -> std::size_t
{
    auto const it
        = std::ranges::find_if(kStrategies, [name](auto const& s) { return std::string_view(s.Name) == name; });
    return static_cast<std::size_t>(it - kStrategies.begin());
}

// ---- per-instance analysis -------------------------------------------------------------------------------------

struct TreeSpec {
    std::string Name;
    std::string Scenario;
    Operon::Tree Tree;
    DomainMap Domains;
    Operon::Hash DiffVar {};
    std::string Expression;
};

struct StrategyResult {
    double Width {};
    double Inf {};
    double Sup {};
    std::vector<int> Splits;
    std::size_t Leaves {};
    std::size_t Aux {};
    bool FellBack {};
};

struct DepthRecord {
    int Depth {};
    double WBest {};
    double WWorst {};
    double InfBest {}; // largest lower bound over all allocations (what the oracle certifies for "f >= L")
    double SupBest {}; // smallest upper bound over all allocations (what the oracle certifies for "f <= U")
    std::size_t Allocations {};
    std::size_t OracleLeaves {};
    std::vector<int> BestSplits;
    std::vector<StrategyResult> Strategies; // aligned with kStrategies
    double LotteryMin {}; // width of production allocation over every axis encounter order: best, worst, mean
    double LotteryMax {};
    double LotteryMean {};
};

struct InstanceRecord {
    std::string Name;
    std::string Scenario;
    std::string Expression;
    BoundKind Kind {};
    std::size_t Axes {};
    std::size_t TreeLength {};
    bool EqualWidths {};
    double W0 {};
    double Inf0 {};
    double Sup0 {};
    std::vector<DepthRecord> Depths;
};

struct SpecResult {
    std::vector<InstanceRecord> Records;
    std::map<std::string, std::size_t> Skipped;
};

constexpr auto NoGradRoot = std::numeric_limits<std::size_t>::max();

// Same derivative construction as BoundFor in shape_constrained_evaluator.cpp. nullopt: identically zero or not
// certain (reason written to `why`).
auto DerivativeTree(Operon::Tree const& tree, Operon::Hash variable, BoundKind kind, std::string& why)
    -> std::optional<Operon::Tree>
{
    auto differentiate
        = [&](Operon::Tree const& source, Operon::Span<Scalar const> coeff) -> std::optional<Operon::Tree> {
        auto dag = Operon::BuildVariableGradientDag(source, coeff, Operon::Span<Operon::Hash const> { &variable, 1 });
        auto const it = std::ranges::find(dag.Variables, variable);
        if (it == dag.Variables.end()) {
            why = "variable_absent";
            return std::nullopt;
        }
        auto const index = static_cast<std::size_t>(it - dag.Variables.begin());
        if (!dag.Certain[index]) {
            why = "no_diff_rule";
            return std::nullopt;
        }
        if (dag.Roots[index] == NoGradRoot) {
            why = "zero_derivative";
            return std::nullopt;
        }
        return Operon::SliceDagRoot(dag.Nodes, dag.Roots[index]);
    };
    auto const coefficients = tree.GetCoefficients();
    auto d1 = differentiate(tree, coefficients);
    if (!d1 || kind == BoundKind::FirstDerivative) {
        return d1;
    }
    return differentiate(*d1, {});
}

// Smear of every axis: width * max|d bounded / d axis| over the root box. Empty when unavailable.
auto SmearScores(Operon::Tree const& tree, DomainMap const& domains, std::vector<Axis> const& axes)
    -> std::vector<double>
{
    std::vector<Operon::Hash> hashes;
    for (auto const& axis : axes) {
        hashes.push_back(axis.Hash);
    }
    auto const coeff = tree.GetCoefficients();
    auto const dag = Operon::BuildVariableGradientDag(tree, coeff, hashes);
    std::vector<double> scores;
    for (auto const& axis : axes) {
        auto const it = std::ranges::find(dag.Variables, axis.Hash);
        if (it == dag.Variables.end()) {
            return {};
        }
        auto const index = static_cast<std::size_t>(it - dag.Variables.begin());
        if (!dag.Certain[index]) {
            return {};
        }
        if (dag.Roots[index] == NoGradRoot) {
            scores.push_back(0.0);
            continue;
        }
        auto const slice = Operon::SliceDagRoot(dag.Nodes, dag.Roots[index]);
        Operon::IntervalEvaluator<Scalar> evaluator(&slice, domains);
        auto const bound = evaluator(slice.GetCoefficients());
        if (!bound || bound->is_empty() || !std::isfinite(bound->inf()) || !std::isfinite(bound->sup())) {
            return {};
        }
        auto const magnitude
            = std::max(std::abs(static_cast<double>(bound->inf())), std::abs(static_cast<double>(bound->sup())));
        scores.push_back(static_cast<double>(axis.Width) * magnitude);
    }
    return scores;
}

auto Analyse(TreeSpec const& spec) -> SpecResult
{
    SpecResult out;
    for (auto const kind : kBoundKinds) {
        auto const skip = [&](char const* reason) { ++out.Skipped[fmt::format("{}:{}", KindName(kind), reason)]; };

        std::optional<Operon::Tree> bounded;
        if (kind == BoundKind::Value) {
            bounded = spec.Tree;
        } else {
            std::string why;
            bounded = DerivativeTree(spec.Tree, spec.DiffVar, kind, why);
            if (!bounded) {
                skip(why.c_str());
                continue;
            }
        }

        auto const axes = BisectionAxes(*bounded, spec.Domains);
        if (axes.size() < 2) {
            skip("fewer_than_2_axes");
            continue;
        }
        GridEvaluator grid(*bounded, spec.Domains, axes);
        auto const zeros = std::vector<int>(axes.size(), 0);
        auto const direct = grid.Bound(zeros);
        if (!direct) {
            skip("direct_bound_not_finite");
            continue;
        }
        auto const w0 = Width(*direct);
        if (w0 <= 0.0) {
            skip("zero_width_direct");
            continue;
        }

        InstanceRecord record;
        record.Name = spec.Name;
        record.Scenario = spec.Scenario;
        record.Expression = spec.Expression;
        record.Kind = kind;
        record.Axes = axes.size();
        record.TreeLength = bounded->Length();
        record.EqualWidths = std::ranges::all_of(axes, [&](auto const& a) { return a.Width == axes.front().Width; });
        record.W0 = w0;
        record.Inf0 = static_cast<double>(direct->inf());
        record.Sup0 = static_cast<double>(direct->sup());

        std::optional<std::size_t> diffAxis;
        if (kind != BoundKind::Value) {
            auto const it = std::ranges::find(axes, spec.DiffVar, &Axis::Hash);
            if (it != axes.end()) {
                diffAxis = static_cast<std::size_t>(it - axes.begin());
            }
        }
        auto const smear = SmearScores(*bounded, spec.Domains, axes);

        for (int depth = kMinOracleDepth; depth <= kMaxOracleDepth; ++depth) {
            auto const allocations = AllocationCount(axes.size(), depth);
            if (allocations * (std::size_t { 1 } << depth) > kOracleLeafBudget) {
                skip("oracle_budget");
                continue;
            }
            std::vector<std::vector<int>> all;
            std::vector<int> current(axes.size(), 0);
            EnumerateCompositions(current, 0, depth, all);

            DepthRecord rec;
            rec.Depth = depth;
            rec.Allocations = all.size();
            rec.WBest = std::numeric_limits<double>::infinity();
            rec.WWorst = 0.0;
            rec.InfBest = -std::numeric_limits<double>::infinity();
            rec.SupBest = std::numeric_limits<double>::infinity();
            // A grid with a failing leaf makes production fall back to the direct bound; the oracle sees the same.
            bool anyValid = false;
            for (auto const& splits : all) {
                auto const grown = grid.Bound(splits);
                anyValid = anyValid || grown.has_value();
                auto const& bound = grown ? *grown : *direct;
                auto const width = Width(bound);
                if (width < rec.WBest) {
                    rec.WBest = width;
                    rec.BestSplits = splits;
                }
                rec.WWorst = std::max(rec.WWorst, width);
                rec.InfBest = std::max(rec.InfBest, static_cast<double>(bound.inf()));
                rec.SupBest = std::min(rec.SupBest, static_cast<double>(bound.sup()));
            }
            rec.OracleLeaves = all.size() * (std::size_t { 1 } << depth);
            if (!anyValid) {
                skip("no_valid_allocation");
                continue;
            }

            StrategyContext const context { axes, diffAxis, smear, grid, depth };
            auto const widthOf = [&](std::vector<int> const& splits) {
                auto const bound = grid.Bound(splits);
                return bound ? Width(*bound) : w0;
            };
            for (auto const& strategy : kStrategies) {
                auto allocation = strategy.Allocate(context);
                auto const bound = grid.Bound(allocation.Splits);
                StrategyResult result;
                result.FellBack = !bound;
                auto const& effective = bound ? *bound : *direct;
                result.Width = Width(effective);
                result.Inf = static_cast<double>(effective.inf());
                result.Sup = static_cast<double>(effective.sup());
                result.Leaves = (std::size_t { 1 } << depth) + allocation.ProbeLeaves;
                result.Aux = allocation.AuxEvaluations;
                result.Splits = std::move(allocation.Splits);
                rec.Strategies.push_back(std::move(result));
            }

            // Production allocation under every axis encounter order.
            auto order = IdentityOrder(axes.size());
            auto const widths = AxisWidths(axes);
            double sum = 0.0;
            std::size_t count = 0;
            rec.LotteryMin = std::numeric_limits<double>::infinity();
            do {
                auto const width = widthOf(WidestFirstSplits(widths, order, depth));
                rec.LotteryMin = std::min(rec.LotteryMin, width);
                rec.LotteryMax = std::max(rec.LotteryMax, width);
                sum += width;
                ++count;
            } while (std::ranges::next_permutation(order).found);
            rec.LotteryMean = sum / static_cast<double>(count);
            record.Depths.push_back(std::move(rec));
        }
        if (!record.Depths.empty()) {
            out.Records.push_back(std::move(record));
        }
    }
    return out;
}

// ---- corpora ---------------------------------------------------------------------------------------------------

// The tie-break example: the same function with the encounter order of X and Y swapped.
auto const kTieBreakCorpus = std::vector<CorpusCase> {
    { "tiebreak_x_first", { { "X", { 0.75F, 1.25F } }, { "Y", { 0.75F, 1.25F } } }, "X + Y * square(square(X - 0.9))" },
    { "tiebreak_y_first", { { "X", { 0.75F, 1.25F } }, { "Y", { 0.75F, 1.25F } } }, "Y * square(square(X - 0.9)) + X" },
};

auto CorpusSpec(CorpusCase const& c) -> TreeSpec
{
    auto prepared = Prepare(c);
    // differentiate with respect to the first listed variable
    auto const diffVar = prepared.dataset.GetVariable(c.domains.front().first)->Hash;
    return { c.name, "corpus", std::move(prepared.tree), std::move(prepared.domains), diffVar, c.expression };
}

constexpr std::array<char const*, 4> kScenarioNames { "pos_equal", "pos_unequal", "mixed_equal", "mixed_unequal" };

auto RandomDomains(Operon::RandomGenerator& rd, std::span<Operon::Hash const> variables, std::size_t scenario)
    -> DomainMap
{
    static constexpr std::array<Scalar, 3> equalWidths { 0.5F, 1.0F, 2.0F };
    static constexpr std::array<Scalar, 4> unequalWidths { 0.25F, 0.5F, 1.0F, 2.0F };
    static constexpr std::array<Scalar, 6> positiveLo { 0.25F, 0.5F, 0.75F, 1.0F, 1.5F, 2.0F };
    static constexpr std::array<Scalar, 3> mixedFraction { 0.25F, 0.5F, 0.75F };
    auto const pick = [&rd](auto const& values) {
        return values[std::uniform_int_distribution<std::size_t>(0, values.size() - 1)(rd)];
    };
    auto const positive = scenario < 2;
    auto const equal = scenario % 2 == 0;
    auto const commonWidth = pick(equalWidths);
    DomainMap domains;
    for (auto const variable : variables) {
        auto const width = equal ? commonWidth : pick(unequalWidths);
        auto const lo = positive ? pick(positiveLo) : -width * pick(mixedFraction);
        domains.emplace(variable, std::pair<Scalar, Scalar> { lo, lo + width });
    }
    return domains;
}

auto RandomSpecs(std::size_t count, std::uint64_t seed) -> std::vector<TreeSpec>
{
    constexpr std::size_t maxVariables = 4;
    constexpr std::size_t maxLength = 30;
    std::vector<std::string> names;
    std::vector<std::vector<Scalar>> values;
    for (std::size_t i = 0; i < maxVariables; ++i) {
        names.push_back(fmt::format("X{}", i + 1));
        values.push_back({ 0, 1 });
    }
    Operon::Dataset dataset(names, values);
    std::vector<Operon::Hash> allVariables;
    for (auto const& name : names) {
        allVariables.push_back(dataset.GetVariable(name)->Hash);
    }
    Operon::PrimitiveSet pset { Operon::PrimitiveSet::Full };
    // Aq has no symbolic derivative rule; keeping it would only turn derivative instances into skips.
    pset.RemovePrimitive(Operon::Hash(Operon::BuiltinOp::Aq));
    Operon::RandomGenerator rd(seed);
    std::uniform_real_distribution<Scalar> magnitude(0.25F, 2.0F);

    std::vector<TreeSpec> specs;
    for (std::size_t i = 0; i < count; ++i) {
        auto const nVariables = std::uniform_int_distribution<std::size_t>(2, maxVariables)(rd);
        auto const length = std::uniform_int_distribution<std::size_t>(5, maxLength)(rd);
        auto const scenario = std::uniform_int_distribution<std::size_t>(0, kScenarioNames.size() - 1)(rd);
        std::vector<Operon::Hash> variables(
            allVariables.begin(), allVariables.begin() + static_cast<std::ptrdiff_t>(nVariables));
        Operon::BalancedTreeCreator creator { &pset, variables, 0.0, maxLength };
        auto tree = creator(rd, length, 1, 10);

        // leaf weights: half stay 1, the rest random sign and magnitude
        auto coefficients = tree.GetCoefficients();
        for (auto& coefficient : coefficients) {
            if (std::bernoulli_distribution(0.5)(rd)) {
                coefficient = (std::bernoulli_distribution(0.5)(rd) ? Scalar { 1 } : Scalar { -1 }) * magnitude(rd);
            }
        }
        tree.SetCoefficients(coefficients);

        std::vector<Operon::Hash> present;
        for (auto const& node : tree.Nodes()) {
            if (node.IsVariable() && std::ranges::find(present, node.HashValue) == present.end()) {
                present.push_back(node.HashValue);
            }
        }
        if (present.empty()) {
            continue;
        }
        auto const diffVar = present[std::uniform_int_distribution<std::size_t>(0, present.size() - 1)(rd)];
        auto domains = RandomDomains(rd, variables, scenario);
        auto expression = fmt::format("{:infix}", Operon::Fmt::TreeFormatArgs { tree, dataset, 3 });
        specs.push_back({ fmt::format("rnd{}", i), kScenarioNames[scenario], std::move(tree), std::move(domains),
            diffVar, std::move(expression) });
    }
    return specs;
}

auto EnvSize(char const* name, std::size_t fallback) -> std::size_t
{
    auto const* value = std::getenv(name); // NOLINT(concurrency-mt-unsafe)
    return value != nullptr ? static_cast<std::size_t>(std::strtoull(value, nullptr, 10)) : fallback;
}

// ---- statistics and reporting ----------------------------------------------------------------------------------

struct Group {
    std::optional<BoundKind> Kind;
    std::optional<int> Depth;
    std::optional<bool> Equal;
};

template <typename F> void ForEachDepth(std::vector<InstanceRecord> const& records, Group const& group, F&& fn)
{
    for (auto const& r : records) {
        if ((group.Kind && *group.Kind != r.Kind) || (group.Equal && *group.Equal != r.EqualWidths)) {
            continue;
        }
        for (auto const& d : r.Depths) {
            if (group.Depth && *group.Depth != d.Depth) {
                continue;
            }
            fn(r, d);
        }
    }
}

struct Summary {
    double Mean {};
    double Median {};
    double P95 {};
    double Max {};
};

auto Summarize(std::vector<double> values) -> Summary
{
    if (values.empty()) {
        return {};
    }
    std::ranges::sort(values);
    auto const n = values.size();
    auto const rank = [&](double p) {
        return values[std::min(n - 1, static_cast<std::size_t>(std::ceil(p * static_cast<double>(n))) - 1)];
    };
    return { std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(n), rank(0.5), rank(0.95),
        values.back() };
}

auto Denominator(InstanceRecord const& r, DepthRecord const& d) -> double { return std::max(d.WBest, 1e-6 * r.W0); }

auto RegretOf(double width, InstanceRecord const& r, DepthRecord const& d) -> double
{
    return std::max(0.0, (width - d.WBest) / Denominator(r, d));
}

// Instances where the allocation matters at all: the worst allocation is >1% wider than the best.
auto Informative(InstanceRecord const& r, DepthRecord const& d) -> bool
{
    return (d.WWorst - d.WBest) > 0.01 * Denominator(r, d);
}

auto GroupLabel(Group const& g) -> std::string
{
    return fmt::format("kind={} depth={} widths={}", g.Kind ? KindName(*g.Kind) : "all",
        g.Depth ? std::to_string(*g.Depth) : "2-6", g.Equal ? (*g.Equal ? "equal" : "unequal") : "any");
}

void PrintRegretTable(std::vector<InstanceRecord> const& records, Group const& group)
{
    struct Acc {
        std::vector<double> Regret;
        std::vector<double> RegretInformative;
        std::size_t Optimal {};
        std::size_t OptimalInformative {};
        double Position {};
        std::size_t PositionCount {};
        double Reduction {};
        double Leaves {};
        double Aux {};
    };
    std::vector<Acc> acc(kStrategies.size());
    std::size_t n = 0;
    std::size_t informative = 0;
    double oracleReduction = 0.0;
    double worstReduction = 0.0;
    double oracleLeaves = 0.0;
    ForEachDepth(records, group, [&](InstanceRecord const& r, DepthRecord const& d) {
        ++n;
        auto const isInformative = Informative(r, d);
        informative += isInformative ? 1 : 0;
        oracleReduction += 1.0 - (d.WBest / r.W0);
        worstReduction += 1.0 - (d.WWorst / r.W0);
        oracleLeaves += static_cast<double>(d.OracleLeaves);
        for (std::size_t i = 0; i < kStrategies.size(); ++i) {
            auto const& s = d.Strategies[i];
            auto const regret = RegretOf(s.Width, r, d);
            auto& a = acc[i];
            a.Regret.push_back(regret);
            a.Optimal += regret <= kOptimalTolerance ? 1 : 0;
            if (isInformative) {
                a.RegretInformative.push_back(regret);
                a.OptimalInformative += regret <= kOptimalTolerance ? 1 : 0;
                a.Position += std::clamp((s.Width - d.WBest) / (d.WWorst - d.WBest), 0.0, 1.0);
                ++a.PositionCount;
            }
            a.Reduction += 1.0 - (s.Width / r.W0);
            a.Leaves += static_cast<double>(s.Leaves);
            a.Aux += static_cast<double>(s.Aux);
        }
    });
    if (n == 0) {
        return;
    }
    auto const dn = static_cast<double>(n);
    auto const dInf = static_cast<double>(std::max<std::size_t>(informative, 1));
    fmt::print("\n[{}] n={} informative={} ({:.1f}%)\n", GroupLabel(group), n, informative,
        100.0 * static_cast<double>(informative) / dn);
    fmt::print("  regret = W/W_oracle - 1 in %; 'all' = every (instance,depth); 'inf' = allocation matters (worst >1% "
               "wider than best); geo = geometric mean of W/W_oracle - 1; cap = mean of min(regret,100%); pos = "
               "(W-Wbest)/(Wworst-Wbest); red = mean 1-W/W0\n");
    fmt::print("  {:<15} {:>6} {:>7} {:>8} {:>10} | {:>6} {:>7} {:>8} {:>7} {:>6} {:>6} {:>6} | {:>6} {:>8} {:>5}\n",
        "strategy", "opt%", "med%", "p95%", "max%", "optI%", "medI%", "p95I%", "geoI%", "capI%", "gt100I", "pos",
        "red%", "leaves", "aux");
    for (std::size_t i = 0; i < kStrategies.size(); ++i) {
        auto const& a = acc[i];
        auto const all = Summarize(a.Regret);
        auto const inf = Summarize(a.RegretInformative);
        auto const capped = [&] {
            double sum = 0.0;
            for (auto r : a.RegretInformative) {
                sum += std::min(r, 1.0);
            }
            return sum / dInf;
        }();
        auto const above
            = static_cast<double>(std::ranges::count_if(a.RegretInformative, [](double r) { return r > 1.0; }));
        auto const geo = [&] {
            double sum = 0.0;
            for (auto r : a.RegretInformative) {
                sum += std::log1p(r);
            }
            return std::expm1(sum / dInf);
        }();
        fmt::print(
            "  {:<15} {:6.1f} {:7.2f} {:8.2f} {:10.3g} | {:6.1f} {:7.2f} {:8.2f} {:7.2f} {:6.1f} {:6.1f} {:6.3f} | "
            "{:6.1f} {:8.1f} {:5.1f}\n",
            kStrategies[i].Name, 100.0 * static_cast<double>(a.Optimal) / dn, 100.0 * all.Median, 100.0 * all.P95,
            100.0 * all.Max, 100.0 * static_cast<double>(a.OptimalInformative) / dInf, 100.0 * inf.Median,
            100.0 * inf.P95, 100.0 * geo, 100.0 * capped, 100.0 * above / dInf,
            a.Position / static_cast<double>(std::max<std::size_t>(a.PositionCount, 1)), 100.0 * a.Reduction / dn,
            a.Leaves / dn, a.Aux / dn);
    }
    fmt::print("  {:<15} {:>6} {:>7} {:>8} {:>10} | {:>6} {:>7} {:>8} {:>7} {:>6} {:>6} {:>6} | {:6.1f} {:8.1f}\n",
        "oracle", "", "", "", "", "", "", "", "", "", "", "", 100.0 * oracleReduction / dn, oracleLeaves / dn);
    fmt::print("  {:<15} {:>6} {:>7} {:>8} {:>10} | {:>6} {:>7} {:>8} {:>7} {:>6} {:>6} {:>6} | {:6.1f}\n",
        "worst alloc", "", "", "", "", "", "", "", "", "", "", "", 100.0 * worstReduction / dn);
}

void PrintTieBreakRow(std::vector<InstanceRecord> const& records, Group const& group)
{
    auto const prod = StrategyIndex("production");
    auto const rev = StrategyIndex("production_rev");
    std::size_t n = 0;
    std::size_t differ = 0;
    std::size_t informative = 0;
    std::size_t prodBetter = 0;
    std::size_t prodWorse = 0;
    std::vector<double> delta; // |W_prod / W_rev - 1| over informative instances
    double logProd = 0.0;
    double logRev = 0.0;
    double logOrderMean = 0.0;
    double logOrderBest = 0.0;
    double logOrderWorst = 0.0;
    ForEachDepth(records, group, [&](InstanceRecord const& r, DepthRecord const& d) {
        ++n;
        auto const& p = d.Strategies[prod];
        auto const& q = d.Strategies[rev];
        differ += p.Splits != q.Splits ? 1 : 0;
        if (!Informative(r, d)) {
            return;
        }
        ++informative;
        delta.push_back(std::abs((p.Width / q.Width) - 1.0));
        prodBetter += p.Width < q.Width * 0.99 ? 1 : 0;
        prodWorse += p.Width > q.Width * 1.01 ? 1 : 0;
        logProd += std::log1p(RegretOf(p.Width, r, d));
        logRev += std::log1p(RegretOf(q.Width, r, d));
        logOrderMean += std::log1p(RegretOf(d.LotteryMean, r, d));
        logOrderBest += std::log1p(RegretOf(d.LotteryMin, r, d));
        logOrderWorst += std::log1p(RegretOf(d.LotteryMax, r, d));
    });
    if (n == 0) {
        return;
    }
    auto const dn = static_cast<double>(n);
    auto const di = static_cast<double>(std::max<std::size_t>(informative, 1));
    auto const s = Summarize(delta);
    auto const geo = [&](double logSum) { return 100.0 * std::expm1(logSum / di); };
    fmt::print("  {:<44} n={:<6} inf={:<6} alloc-differs={:5.1f}% | informative: prod<rev by>1%:{:5.1f}% prod>rev "
               "by>1%:{:5.1f}% "
               "|W_prod/W_rev-1| med={:6.1f}% p95={:7.1f}% max={:10.1f}% | geo-regret%: prod={:7.2f} rev={:7.2f} "
               "order-mean={:7.2f} order-best={:7.2f} order-worst={:7.2f}\n",
        GroupLabel(group), n, informative, 100.0 * static_cast<double>(differ) / dn,
        100.0 * static_cast<double>(prodBetter) / di, 100.0 * static_cast<double>(prodWorse) / di, 100.0 * s.Median,
        100.0 * s.P95, 100.0 * s.Max, geo(logProd), geo(logRev), geo(logOrderMean), geo(logOrderBest),
        geo(logOrderWorst));
}

// Feasibility: for each (instance, depth) a ladder of deterministic thresholds between the bound the oracle reaches
// at the deepest depth (a proxy for the true extremum) and the depth-0 bound. A strategy "falsely rejects" a
// threshold when its own bound violates it while the best allocation at that depth satisfies it.
void PrintFeasibilityMatrix(std::vector<InstanceRecord> const& records, std::optional<BoundKind> kind)
{
    struct Acc {
        std::size_t Pairs {};
        std::size_t DirectReject {};
        std::size_t OracleReject {};
        std::vector<std::size_t> Reject = std::vector<std::size_t>(kStrategies.size());
        std::vector<std::size_t> FalseReject = std::vector<std::size_t>(kStrategies.size());
    };
    std::array<Acc, kMaxOracleDepth + 1> byDepth;
    for (auto const& r : records) {
        if ((kind && *kind != r.Kind) || r.Depths.empty() || r.Depths.back().Depth != kMaxOracleDepth) {
            continue;
        }
        auto const refSup = r.Depths.back().SupBest;
        auto const refInf = r.Depths.back().InfBest;
        for (auto const& d : r.Depths) {
            auto& a = byDepth[static_cast<std::size_t>(d.Depth)];
            for (int side = 0; side < 2; ++side) {
                auto const upper = side == 0;
                auto const gap = upper ? r.Sup0 - refSup : refInf - r.Inf0;
                if (gap <= 0.0) {
                    continue;
                }
                for (std::size_t j = 0; j < kThresholdsPerSide; ++j) {
                    auto const u = (static_cast<double>(j) + 0.5) / static_cast<double>(kThresholdsPerSide);
                    auto const threshold = upper ? refSup + (u * gap) : refInf - (u * gap);
                    auto const satisfied
                        = [&](double inf, double sup) { return upper ? sup <= threshold : inf >= threshold; };
                    ++a.Pairs;
                    a.DirectReject += satisfied(r.Inf0, r.Sup0) ? 0 : 1;
                    auto const oracleOk = satisfied(d.InfBest, d.SupBest);
                    a.OracleReject += oracleOk ? 0 : 1;
                    for (std::size_t i = 0; i < kStrategies.size(); ++i) {
                        auto const ok = satisfied(d.Strategies[i].Inf, d.Strategies[i].Sup);
                        a.Reject[i] += ok ? 0 : 1;
                        a.FalseReject[i] += (!ok && oracleOk) ? 1 : 0;
                    }
                }
            }
        }
    }
    fmt::print(
        "\n[feasibility kind={}] reject% / false-reject% (strategy rejects, best allocation certifies) per depth\n",
        kind ? KindName(*kind) : "all");
    fmt::print("  {:<15}", "strategy");
    for (int depth = kMinOracleDepth; depth <= kMaxOracleDepth; ++depth) {
        fmt::print(" |   d{}: rej%  false%", depth);
    }
    fmt::print("\n");
    auto const pct = [](std::size_t part, std::size_t whole) {
        return whole == 0 ? 0.0 : 100.0 * static_cast<double>(part) / static_cast<double>(whole);
    };
    auto const row = [&](char const* name, auto&& rejectOf, auto&& falseOf) {
        fmt::print("  {:<15}", name);
        for (int depth = kMinOracleDepth; depth <= kMaxOracleDepth; ++depth) {
            auto const& a = byDepth[static_cast<std::size_t>(depth)];
            fmt::print(" | {:11.1f} {:6.1f}", pct(rejectOf(a), a.Pairs), pct(falseOf(a), a.Pairs));
        }
        fmt::print("\n");
    };
    row("depth0", [](Acc const& a) { return a.DirectReject; }, [](Acc const&) { return std::size_t { 0 }; });
    for (std::size_t i = 0; i < kStrategies.size(); ++i) {
        row(
            kStrategies[i].Name, [i](Acc const& a) { return a.Reject[i]; },
            [i](Acc const& a) { return a.FalseReject[i]; });
    }
    row("oracle", [](Acc const& a) { return a.OracleReject; }, [](Acc const&) { return std::size_t { 0 }; });
    fmt::print("  pairs per depth:");
    for (int depth = kMinOracleDepth; depth <= kMaxOracleDepth; ++depth) {
        fmt::print(" d{}={}", depth, byDepth[static_cast<std::size_t>(depth)].Pairs);
    }
    fmt::print("\n");
}

// Instances where even the best fixed rule (every non-probing strategy, chosen per instance) is far from the oracle.
void PrintOracleGaps(std::vector<InstanceRecord> const& records, std::size_t top)
{
    struct Gap {
        InstanceRecord const* R;
        DepthRecord const* D;
        double MinCheapRegret;
    };
    std::vector<Gap> gaps;
    for (auto const& r : records) {
        for (auto const& d : r.Depths) {
            if (!Informative(r, d)) {
                continue;
            }
            auto best = std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < kStrategies.size(); ++i) {
                if (kStrategies[i].Cheap) {
                    best = std::min(best, RegretOf(d.Strategies[i].Width, r, d));
                }
            }
            gaps.push_back({ &r, &d, best });
        }
    }
    fmt::print("\n[best-of-cheap-rules per (instance,depth), informative instances only] share with regret above X\n");
    for (auto const kind : kBoundKinds) {
        std::vector<double> values;
        for (auto const& g : gaps) {
            if (g.R->Kind == kind) {
                values.push_back(g.MinCheapRegret);
            }
        }
        auto const share = [&](double threshold) {
            return values.empty() ? 0.0 : 100.0 * static_cast<double>(std::ranges::count_if(values, [&](double v) {
                return v > threshold;
            })) / static_cast<double>(values.size());
        };
        auto const s = Summarize(values);
        fmt::print(
            "  {:<6} n={:<6} mean={:7.2f}% med={:6.2f}% p95={:8.2f}% max={:9.1f}% | >10%: {:5.1f}%  >50%: {:5.1f}%  "
            ">100%: {:5.1f}%\n",
            KindName(kind), values.size(), 100.0 * s.Mean, 100.0 * s.Median, 100.0 * s.P95, 100.0 * s.Max, share(0.1),
            share(0.5), share(1.0));
    }
    std::ranges::sort(gaps, [](auto const& a, auto const& b) { return a.MinCheapRegret > b.MinCheapRegret; });
    fmt::print("\n[largest oracle gaps vs the best cheap rule]\n");
    for (std::size_t i = 0; i < std::min(top, gaps.size()); ++i) {
        auto const& r = *gaps[i].R;
        auto const& d = *gaps[i].D;
        fmt::print("  {} {} k={} depth={} scenario={} w0={:.4g} best={:.4g} (alloc {}) worst={:.4g} "
                   "best-cheap-regret={:.1f}%\n"
                   "    f = {}\n    regret%:",
            r.Name, KindName(r.Kind), r.Axes, d.Depth, r.Scenario, r.W0, d.WBest, d.BestSplits, d.WWorst,
            100.0 * gaps[i].MinCheapRegret, r.Expression);
        for (std::size_t s = 0; s < kStrategies.size(); ++s) {
            fmt::print(" {}={:.0f}", kStrategies[s].Name, 100.0 * RegretOf(d.Strategies[s].Width, r, d));
        }
        fmt::print("\n");
    }
}

void PrintCorpusTable(std::vector<InstanceRecord> const& records)
{
    fmt::print("\n[fixed corpus] per case/kind/depth; regret% vs oracle best per strategy:\n  strategies:");
    for (auto const& s : kStrategies) {
        fmt::print(" {}", s.Name);
    }
    fmt::print("\n");
    for (auto const& r : records) {
        for (auto const& d : r.Depths) {
            fmt::print("  {:<20} {:<5} k={} d={} w0={:<10.5g} best={:<10.5g} worst={:<10.5g} alloc={} |", r.Name,
                KindName(r.Kind), r.Axes, d.Depth, r.W0, d.WBest, d.WWorst, d.BestSplits);
            for (auto const& s : d.Strategies) {
                fmt::print(" {:6.1f}", 100.0 * RegretOf(s.Width, r, d));
            }
            fmt::print("\n");
        }
    }
}

void WriteCsv(std::string const& path, std::vector<InstanceRecord> const& records)
{
    std::ofstream out(path);
    out << "name,scenario,kind,axes,tree_length,equal_widths,depth,strategy,width,w0,w_best,w_worst,leaves,splits\n";
    for (auto const& r : records) {
        for (auto const& d : r.Depths) {
            for (std::size_t i = 0; i < kStrategies.size(); ++i) {
                auto const& s = d.Strategies[i];
                out << fmt::format("{},{},{},{},{},{},{},{},{:.9g},{:.9g},{:.9g},{:.9g},{},{}\n", r.Name, r.Scenario,
                    KindName(r.Kind), r.Axes, r.TreeLength, r.EqualWidths ? 1 : 0, d.Depth, kStrategies[i].Name,
                    s.Width, r.W0, d.WBest, d.WWorst, s.Leaves, fmt::join(s.Splits, "/"));
            }
        }
    }
}

} // namespace

TEST_CASE("Bisection allocation oracle: regret of allocation strategies",
    "[performance][shape-constraints][bisection][oracle][.bisection-oracle]")
{
    auto const treeCount = EnvSize("OPERON_ORACLE_TREES", 1000);
    auto const seed = EnvSize("OPERON_ORACLE_SEED", 1234);
    auto const threads = EnvSize("OPERON_ORACLE_THREADS", std::max(1U, std::thread::hardware_concurrency()));

    std::vector<TreeSpec> specs;
    for (auto const& c : kCorpus) {
        specs.push_back(CorpusSpec(c));
    }
    for (auto const& c : kTieBreakCorpus) {
        specs.push_back(CorpusSpec(c));
    }
    auto const fixedCount = specs.size();
    for (auto& spec : RandomSpecs(treeCount, seed)) {
        specs.push_back(std::move(spec));
    }

    std::vector<SpecResult> results(specs.size());
    std::atomic<std::size_t> next { 0 };
    std::vector<std::thread> pool;
    for (std::size_t t = 0; t < threads; ++t) {
        pool.emplace_back([&] {
            for (auto i = next.fetch_add(1); i < specs.size(); i = next.fetch_add(1)) {
                results[i] = Analyse(specs[i]);
            }
        });
    }
    for (auto& thread : pool) {
        thread.join();
    }

    std::vector<InstanceRecord> fixed;
    std::vector<InstanceRecord> random;
    std::map<std::string, std::size_t> skipped;
    for (std::size_t i = 0; i < results.size(); ++i) {
        auto& target = i < fixedCount ? fixed : random;
        for (auto& record : results[i].Records) {
            target.push_back(std::move(record));
        }
        for (auto const& [reason, count] : results[i].Skipped) {
            skipped[reason] += count;
        }
    }

    fmt::print("\nrandom corpus: trees={} seed={} instances={} (instance = tree x bound kind)\n", treeCount, seed,
        random.size());
    for (auto const kind : kBoundKinds) {
        fmt::print("  {:<6} instances={}\n", KindName(kind),
            std::ranges::count_if(random, [kind](auto const& r) { return r.Kind == kind; }));
    }
    fmt::print("skipped (tree x kind) by reason, all corpora:\n");
    for (auto const& [reason, count] : skipped) {
        fmt::print("  {:<40} {}\n", reason, count);
    }
    fmt::print("strategies:");
    for (auto const& s : kStrategies) {
        fmt::print(" {}{}", s.Name, s.Cheap ? "" : "(probing)");
    }
    fmt::print("\n");

    // Sanity: no strategy beats the oracle (all strategy grids are members of the enumerated set).
    std::size_t violations = 0;
    for (auto const* records : { &fixed, &random }) {
        for (auto const& r : *records) {
            for (auto const& d : r.Depths) {
                for (auto const& s : d.Strategies) {
                    violations += s.Width < d.WBest * (1.0 - 1e-6) ? 1 : 0;
                }
            }
        }
    }
    CHECK(violations == 0);

    PrintCorpusTable(fixed);

    fmt::print("\n=============== random corpus: regret tables ===============\n");
    for (auto const kind : kBoundKinds) {
        for (int depth = kMinOracleDepth; depth <= kMaxOracleDepth; ++depth) {
            PrintRegretTable(random, { kind, depth, std::nullopt });
        }
        PrintRegretTable(random, { kind, std::nullopt, std::nullopt });
    }
    PrintRegretTable(random, {});
    PrintRegretTable(random, { std::nullopt, std::nullopt, true });
    PrintRegretTable(random, { std::nullopt, std::nullopt, false });

    fmt::print("\n=============== tie-break: production vs reversed encounter order ===============\n");
    for (auto const kind : kBoundKinds) {
        for (int depth = kMinOracleDepth; depth <= kMaxOracleDepth; ++depth) {
            for (auto const equal : { true, false }) {
                PrintTieBreakRow(random, { kind, depth, equal });
            }
        }
        for (auto const equal : { true, false }) {
            PrintTieBreakRow(random, { kind, std::nullopt, equal });
        }
    }
    for (auto const equal : { true, false }) {
        PrintTieBreakRow(random, { std::nullopt, std::nullopt, equal });
    }

    fmt::print("\n=============== feasibility ===============\n");
    for (auto const kind : kBoundKinds) {
        PrintFeasibilityMatrix(random, kind);
    }
    PrintFeasibilityMatrix(random, std::nullopt);

    PrintOracleGaps(random, 12);

    if (auto const* path = std::getenv("OPERON_ORACLE_CSV"); path != nullptr) { // NOLINT(concurrency-mt-unsafe)
        WriteCsv(path, random);
    }
}

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
