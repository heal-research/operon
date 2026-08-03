// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/operators/shape_constrained_evaluator.hpp"

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>

#include <fmt/format.h>
#include <tl/expected.hpp>

#include "operon/core/dataset.hpp"
#include "operon/core/tree_diff.hpp"
#include "operon/hash/hash.hpp"
#include "operon/interpreter/affine_evaluator.hpp"

namespace Operon {

namespace {

constexpr std::size_t NoGrad = std::numeric_limits<std::size_t>::max();

// Pure, non-mutating hash for the Feasible() memoization cache below.
// Deliberately does NOT use Tree::Hash(), which writes each node's
// CalculatedHashValue in place: Feasible() is called both from Prepare()
// (single-threaded, safe to mutate) and from Evaluate() (parallel, one
// call per offspring during generation) as well as any external caller
// (e.g. FeasibilityFirstComparison, during parallel selection/
// reinsertion) -- two threads hashing the same shared tree concurrently
// would race on that mutation. Must fold in coefficient values, not just
// structure: feasibility is checked against the tree's actual optimized
// weights, so two structurally identical trees with different
// coefficients can differ in feasibility and must not share a cache
// entry. Collision risk is accepted the same way every other hash-keyed
// cache in this codebase already does (Zobrist's own transposition
// cache, content_hash.hpp).
auto HashTreeForMemo(Tree const& tree) -> Operon::Hash
{
    Operon::Hasher const hasher;
    Operon::Hash h{};
    for (auto const& n : tree.Nodes()) {
        auto const valueHash = hasher(reinterpret_cast<uint8_t const*>(&n.Value), sizeof(n.Value)); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        auto const nodeHash = n.HashValue ^ (valueHash + 0x9e3779b97f4a7c15ULL + (n.HashValue << 6U) + (n.HashValue >> 2U));
        h ^= nodeHash + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
    }
    return h;
}

using Interval = AffineEvaluator::Interval;
using BoundResult = tl::expected<Interval, std::string>;

// Slices a VariableGradientDag root into a standalone Tree (same Ref-node
// convention as JacobianDag/HessianDag — see the tree_diff tests for the
// same idiom). std::nullopt means the derivative is identically zero, not
// an error.
auto SliceToTree(VariableGradientDag const& dag, std::size_t root) -> std::optional<Tree>
{
    if (root == NoGrad) { return std::nullopt; }
    Operon::Vector<Node> sliced(dag.Nodes.begin(), dag.Nodes.begin() + static_cast<std::ptrdiff_t>(root) + 1);
    Tree t(std::move(sliced));
    t.UpdateNodes();
    return t;
}

auto VariableIndex(VariableGradientDag const& dag, Operon::Hash variable) -> std::optional<std::size_t>
{
    auto it = std::ranges::find(dag.Variables, variable);
    if (it == dag.Variables.end()) { return std::nullopt; }
    return static_cast<std::size_t>(std::distance(dag.Variables.begin(), it));
}

// The only point in this file that crosses into AffineEvaluator, which is
// pre-existing code that throws by design (domain violations, e.g.
// dividing by a zero-containing interval; ops with no registered affine
// rule) rather than returning an expected/empty value the way
// IntervalEvaluator does. Converting AffineEvaluator itself to
// tl::expected is a separate, larger change (deferred); this is the one
// place that adapts its exceptions to this file's own expected-based
// internals, so the rest of this file never needs a try/catch.
auto TryAffineBound(Tree const& tree, AffineEvaluator::DomainMap const& domains) -> BoundResult
{
    try {
        AffineEvaluator ae(&tree, domains);
        return ae.Evaluate(tree.GetCoefficients()).to_interval();
    } catch (std::exception const& e) {
        return tl::unexpected(std::string(e.what()));
    }
}

// The bound for one constraint's Op: the tree itself for Identity, or the
// (possibly twice-)differentiated tree for First-/SecondDerivative — an
// identically-zero derivative bounds to the degenerate interval [0, 0]
// rather than requiring a special case at every call site.
//
// VariableGradientDag::Certain[k] == false means the derivative dag hit an
// op with no rule on this variable's dependency path -- Roots[k] then isn't
// a trustworthy "derivative is zero" claim (see tree_diff.hpp). Reported as
// an error result here, same as any other can't-certify case.
auto BoundFor(ShapeConstraintOp op, Tree const& tree, Operon::Hash variable,
              AffineEvaluator::DomainMap const& domains) -> BoundResult
{
    if (op == ShapeConstraintOp::Identity) { return TryAffineBound(tree, domains); }

    auto const coeff = tree.GetCoefficients();
    auto dag1 = BuildVariableGradientDag(tree, coeff);
    auto const i1 = VariableIndex(dag1, variable);
    if (!i1) { return BoundResult(Interval(Operon::Scalar{0}, Operon::Scalar{0})); }
    if (!dag1.Certain[*i1]) { return tl::unexpected("variable derivative involves an op with no differentiation rule"); }
    auto d1 = SliceToTree(dag1, dag1.Roots[*i1]);
    if (op == ShapeConstraintOp::FirstDerivative) {
        return d1 ? TryAffineBound(*d1, domains) : BoundResult(Interval(Operon::Scalar{0}, Operon::Scalar{0}));
    }

    // SecondDerivative: differentiate the materialized first-derivative
    // tree again, same variable both times — mixed partials aren't needed
    // by any constraint in this codebase's problem set.
    if (!d1) { return BoundResult(Interval(Operon::Scalar{0}, Operon::Scalar{0})); }
    auto const coeff1 = d1->GetCoefficients();
    auto dag2 = BuildVariableGradientDag(*d1, coeff1);
    auto const i2 = VariableIndex(dag2, variable);
    if (!i2) { return BoundResult(Interval(Operon::Scalar{0}, Operon::Scalar{0})); }
    if (!dag2.Certain[*i2]) { return tl::unexpected("variable derivative involves an op with no differentiation rule"); }
    auto d2 = SliceToTree(dag2, dag2.Roots[*i2]);
    return d2 ? TryAffineBound(*d2, domains) : BoundResult(Interval(Operon::Scalar{0}, Operon::Scalar{0}));
}

} // namespace

ShapeConstrainedEvaluator::ShapeConstrainedEvaluator(gsl::not_null<EvaluatorBase const*> evaluator, ShapeConstraintSet constraints)
    : EvaluatorBase(evaluator->GetProblem())
    , evaluator_(evaluator)
    , constraints_(std::move(constraints))
{
    auto const* ds = evaluator_->GetProblem()->GetDataset();

    for (auto const& [name, bound] : constraints_.Domains) {
        auto v = ds->GetVariable(name);
        if (!v) {
            throw std::invalid_argument(fmt::format(
                "ShapeConstrainedEvaluator: domain references unknown variable '{}'", name));
        }
        domainsByHash_.insert_or_assign(v->Hash, bound);
    }

    // `constraints.Domains` must cover every input variable the problem
    // actually uses, not just the ones a constraint names directly (see
    // the class-level doc comment: any variable in a scored tree needs a
    // domain, because the affine bound walks the whole original tree
    // internally). Checking this once here, loudly, beats discovering a
    // missing domain later as every individual in a run being rejected.
    for (auto const& hash : evaluator_->GetProblem()->GetInputs()) {
        if (domainsByHash_.contains(hash)) { continue; }
        auto v = ds->GetVariable(hash);
        throw std::invalid_argument(fmt::format(
            "ShapeConstrainedEvaluator: input variable '{}' has no entry in 'domains'",
            v ? v->Name : fmt::format("<hash {}>", hash)));
    }

    constraintVarHash_.reserve(constraints_.Constraints.size());
    for (auto const& c : constraints_.Constraints) {
        // ShapeConstraint::Sign/Bound is an exactly-one-of pair the JSON
        // loader (shape_constraints_config.cpp) already enforces, but the
        // struct itself doesn't -- validate here too since this
        // constructor is a public entry point a C++ caller can reach
        // directly, bypassing the loader.
        if (c.Sign.has_value() == c.Bound.has_value()) {
            throw std::invalid_argument(
                "ShapeConstrainedEvaluator: constraint must set exactly one of Sign or Bound");
        }
        if (c.Sign && *c.Sign != 1 && *c.Sign != -1) {
            throw std::invalid_argument(fmt::format(
                "ShapeConstrainedEvaluator: constraint Sign {} must be 1 or -1", *c.Sign));
        }
        if (c.Bound && c.Bound->first > c.Bound->second) {
            throw std::invalid_argument(fmt::format(
                "ShapeConstrainedEvaluator: constraint Bound [{}, {}] has lo > hi", c.Bound->first, c.Bound->second));
        }

        if (c.Op == ShapeConstraintOp::Identity) {
            constraintVarHash_.push_back(Operon::Hash{});
            continue;
        }
        auto v = ds->GetVariable(c.Variable);
        if (!v) {
            throw std::invalid_argument(fmt::format(
                "ShapeConstrainedEvaluator: constraint references unknown variable '{}'", c.Variable));
        }
        if (!domainsByHash_.contains(v->Hash)) {
            throw std::invalid_argument(fmt::format(
                "ShapeConstrainedEvaluator: constraint on '{}' has no matching entry in 'domains'", c.Variable));
        }
        constraintVarHash_.push_back(v->Hash);
    }
}

namespace {
auto ComputeFeasible(ShapeConstraintSet const& constraints, Operon::Vector<Operon::Hash> const& constraintVarHash,
                      Operon::Map<Operon::Hash, std::pair<Operon::Scalar, Operon::Scalar>> const& domainsByHash,
                      Operon::Tree const& tree) -> bool
{
    for (std::size_t i = 0; i < constraints.Constraints.size(); ++i) {
        auto const& c = constraints.Constraints[i];

        // An error result (domain violation, unmapped op, unsupported
        // derivative op -- see BoundFor/TryAffineBound) means "this
        // constraint can't be certified feasible for this tree", treated
        // the same as a proven violation. The one thing this also
        // swallows -- "no domain bound for variable hash N", a genuine
        // domains-config bug -- degrades to every individual being
        // rejected rather than a hard crash, which is still quickly
        // diagnosable (100% rejection is not a plausible real result).
        auto const bound = BoundFor(c.Op, tree, constraintVarHash[i], domainsByHash);

        bool violated = !bound.has_value();
        if (bound) {
            if (c.Sign) {
                // +1 (non-decreasing): every point in the box must have
                // derivative/value >= 0 -- the enclosure proves this only if
                // its lower bound is already >= 0. -1 is the mirror image.
                violated = (*c.Sign > 0) ? (bound->inf() < Operon::Scalar{0}) : (bound->sup() > Operon::Scalar{0});
            } else if (c.Bound) {
                violated = bound->inf() < c.Bound->first || bound->sup() > c.Bound->second;
            }
        }
        if (violated) { return false; }
    }
    return true;
}
} // namespace

auto ShapeConstrainedEvaluator::Feasible(Operon::Tree const& tree) const -> bool
{
    auto const hash = HashTreeForMemo(tree);
    bool result{};
    // LazyEmplace holds this hash's shard lock across the miss branch, so
    // a concurrent caller hashing to the same key blocks on the first
    // computation rather than duplicating it.
    feasibleCache_.LazyEmplace(hash,
        [&](auto const& e) { result = e.Value; },
        [&](auto& e) {
            result = ComputeFeasible(constraints_, constraintVarHash_, domainsByHash_, tree);
            e.Value = result;
        });
    return result;
}

auto ShapeConstrainedEvaluator::Prepare(Operon::Span<Individual const> pop) const -> void
{
    evaluator_->Prepare(pop);
    feasibleCache_.Clear();
    for (auto const& ind : pop) {
        std::ignore = Feasible(ind.Genotype); // populates the cache as a side effect
    }
}

auto ShapeConstrainedEvaluator::Evaluate(Operon::RandomGenerator& rng, Individual const& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType
{
    ++CallCount;
    if (!Feasible(ind.Genotype)) {
        ++violations_;
        return ReturnType(evaluator_->ObjectiveCount(), static_cast<Operon::Scalar>(worstValue_));
    }
    return (*evaluator_)(rng, ind, buf);
}

} // namespace Operon
