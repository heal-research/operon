// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/algorithms/enumeration.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <thread>
#include <unordered_map>

#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/taskflow.hpp>

#include "operon/optimizer/optimizer.hpp" // for FitResult/FitFailure::FinalCost

namespace Operon {

auto SymbolicComplexity(Operon::Tree const& tree) noexcept -> std::size_t
{
    auto const& nodes = tree.Nodes();
    return static_cast<std::size_t>(std::ranges::count_if(nodes, [](auto const& n) { return !n.IsConstant(); }));
}

namespace {
    // Same-budget dependencies follow this order.
    constexpr std::array ProcessingOrder {
        GrammarSymbol::SimpleTerm,
        GrammarSymbol::SimpleExpr,
        GrammarSymbol::RecurringFactor,
        GrammarSymbol::Term,
        GrammarSymbol::Expression,
    };

    auto AppendNodes(Operon::Vector<Node>& out, Operon::Vector<Node> const& in) -> void
    {
        out.insert(out.end(), in.begin(), in.end());
    }

    // Non-identity placeholders preserve unfitted weights and biases through Simplify().
    constexpr Operon::Scalar WeightPlaceholder { 2.0 };
    constexpr Operon::Scalar BiasPlaceholder { 1.0 };

    // Reduction can shrink a nominal budget by up to two nodes.
    constexpr std::size_t WorkingBudgetMargin = 2;
} // namespace

EnumerationEngine::EnumerationEngine(Operon::Grammar grammar, std::size_t maxComplexity, Operon::RandomGenerator& rng,
    DomainPruningConfig pruning)
    : grammar_(std::move(grammar))
    , maxComplexity_(maxComplexity)
    , workingCeiling_(maxComplexity_ + WorkingBudgetMargin)
    , zobrist_(rng, /*maxLength=*/1, grammar_.VariableHashes())
    , pruning_(pruning)
{
    buckets_.resize(GrammarSymbols::Count);
    seen_.resize(GrammarSymbols::Count);
    bucketMutex_.resize(GrammarSymbols::Count);
    for (auto& row : buckets_) {
        row.resize(workingCeiling_ + 1);
    }
    for (auto& row : seen_) {
        row.resize(workingCeiling_ + 1);
    }
    // Construct mutex rows at their final size: std::mutex is non-movable.
    for (auto& row : bucketMutex_) {
        row = std::vector<std::mutex>(workingCeiling_ + 1);
    }
}

auto EnumerationEngine::Bucket(GrammarSymbol nt, std::size_t budget) const -> std::span<Operon::Tree const>
{
    EXPECT(budget <= maxComplexity_);
    return buckets_[GrammarSymbols::GetIndex(nt)][budget];
}

auto EnumerationEngine::TryInsert(GrammarSymbol nt, Operon::Tree tree) -> bool
{
    tree.Reduce();
    tree.Simplify();
    if (pruning_.Enabled && pruning_.Context != nullptr
        && AnalyzeDomain(tree, *pruning_.Context, pruning_.Policy) == DomainStatus::Invalid) {
        return false;
    }
    auto complexity = SymbolicComplexity(tree);

    auto idx = GrammarSymbols::GetIndex(nt);
    auto hash = ComputeContentHash(tree, zobrist_);

    bool const novel = seen_[idx][complexity].lazy_emplace_l(
        hash, [](auto&) { /* already present, nothing to update */ }, [&](auto const& ctor) { ctor(hash); });
    if (novel) {
        std::scoped_lock lock(bucketMutex_[idx][complexity]);
        buckets_[idx][complexity].push_back(std::move(tree));
    }
    return novel;
}

void EnumerationEngine::SeedTerminals()
{
    for (auto varHash : grammar_.VariableHashes()) {
        Node n(NodeType::Variable);
        n.HashValue = varHash;
        for (auto nt : { GrammarSymbol::RecurringFactor, GrammarSymbol::SimpleTerm }) {
            Tree t = Tree({ n }).UpdateNodes();
            TryInsert(nt, std::move(t));
        }
    }
}

void EnumerationEngine::ProcessNonterminal(tf::Executor& executor, GrammarSymbol nt, std::size_t budget)
{
    tf::Taskflow taskflow;

    // Append the production root and optional fixed scale.
    auto appendOpAndScale = [](Operon::Vector<Node>& nodes, Production const& p, std::size_t childCount) {
        nodes.push_back(Node::Function(static_cast<Hash>(p.Op), static_cast<uint16_t>(childCount)));
        if (p.ResultScale != Operon::Scalar { 1 }) {
            auto scale = Node::Constant(static_cast<double>(p.ResultScale));
            scale.Optimize = false;
            nodes.push_back(std::move(scale));
            nodes.push_back(Node::Function(static_cast<Hash>(BuiltinOp::Mul), 2));
        }
    };

    for (auto const& p : grammar_.Productions(nt)) {
        if (p.IsCoercion()) {
            // Coercions preserve the operand's budget.
            auto operandIdx = GrammarSymbols::GetIndex(*p.Operands.front().Symbol);
            auto const& bucket = buckets_[operandIdx][budget];
            if (bucket.empty()) {
                continue;
            }
            taskflow.for_each_index(std::size_t { 0 }, bucket.size(), std::size_t { 1 },
                [this, nt, &bucket](std::size_t i) { TryInsert(nt, bucket[i]); });
            continue;
        }

        std::size_t const nonterminalCount
            = static_cast<std::size_t>(std::ranges::count_if(p.Operands, [](auto const& o) { return !o.IsFixed(); }));

        // Root, optional weight, and optional scale.
        std::size_t const fixedCost
            = 1UL + (p.WeightFirstOperand ? 1UL : 0UL) + (p.ResultScale != Operon::Scalar { 1 } ? 1UL : 0UL);
        if (budget <= fixedCost) {
            continue;
        }
        auto const remaining = budget - fixedCost;

        if (nonterminalCount == 1) {
            // Unary recipes may include one fixed operand (Cube/TenExp).
            std::size_t ntSlot = 0;
            for (; ntSlot < p.Operands.size(); ++ntSlot) {
                if (!p.Operands[ntSlot].IsFixed()) {
                    break;
                }
            }
            auto const operand = *p.Operands[ntSlot].Symbol;
            if (remaining < grammar_.MinComplexity(operand)) {
                continue;
            }
            auto operandIdx = GrammarSymbols::GetIndex(operand);
            auto const& bucket = buckets_[operandIdx][remaining];
            if (bucket.empty()) {
                continue;
            }
            taskflow.for_each_index(std::size_t { 0 }, bucket.size(), std::size_t { 1 },
                [this, nt, &p, &bucket, ntSlot, &appendOpAndScale](std::size_t i) {
                    auto const& t = bucket[i];
                    Operon::Vector<Node> nodes;
                    if (p.WeightFirstOperand) {
                        nodes.push_back(Node::Constant(WeightPlaceholder));
                    }
                    // Postfix stores the semantic first operand immediately before its operator.
                    for (std::size_t k = 0; k < p.Operands.size(); ++k) {
                        std::size_t const slot = p.Operands.size() - 1 - k;
                        if (slot == ntSlot) {
                            AppendNodes(nodes, t.Nodes());
                        } else {
                            nodes.push_back(p.Operands[slot].ToNode());
                        }
                        // Weights only wrap the first semantic operand.
                        if (slot == 0 && p.WeightFirstOperand) {
                            nodes.push_back(Node::Function(static_cast<Hash>(BuiltinOp::Mul), 2));
                        }
                    }
                    if (p.TrailingConstant) {
                        nodes.push_back(Node::Constant(BiasPlaceholder));
                    }
                    appendOpAndScale(nodes, p, p.Operands.size() + (p.TrailingConstant ? 1UL : 0UL));
                    TryInsert(nt, Tree(std::move(nodes)).UpdateNodes());
                });
        } else {
            // Binary recipes use two nonterminal operands.
            EXPECT(p.Operands.size() == 2);
            auto const op0 = *p.Operands[0].Symbol;
            auto const op1 = *p.Operands[1].Symbol;
            auto const idx0 = GrammarSymbols::GetIndex(op0);
            auto const idx1 = GrammarSymbols::GetIndex(op1);
            auto const min0 = grammar_.MinComplexity(op0);
            auto const min1 = grammar_.MinComplexity(op1);
            // Symmetric splits are redundant only for commutative self-combines.
            bool const selfCombineUnweighted = (op0 == op1) && !p.WeightFirstOperand && p.Commutative;

            for (std::size_t b0 = min0; b0 <= remaining; ++b0) {
                if (remaining - b0 < min1) {
                    continue;
                }
                auto const b1 = remaining - b0;
                if (selfCombineUnweighted && b0 > b1) {
                    continue;
                }

                auto const& bucket0 = buckets_[idx0][b0];
                auto const& bucket1 = buckets_[idx1][b1];
                if (bucket0.empty() || bucket1.empty()) {
                    continue;
                }

                // Flatten the Cartesian product to distribute work across either bucket.
                auto const n0 = bucket0.size();
                auto const n1 = bucket1.size();
                taskflow.for_each_index(std::size_t { 0 }, n0 * n1, std::size_t { 1 },
                    [this, nt, &p, &bucket0, &bucket1, n1, &appendOpAndScale](std::size_t k) {
                        auto const& t0 = bucket0[k / n1];
                        auto const& t1 = bucket1[k % n1];
                        Operon::Vector<Node> nodes;
                        // Store operand zero last so it is the semantic first operand in postfix.
                        AppendNodes(nodes, t1.Nodes());
                        if (p.WeightFirstOperand) {
                            nodes.push_back(Node::Constant(WeightPlaceholder));
                        }
                        AppendNodes(nodes, t0.Nodes());
                        if (p.WeightFirstOperand) {
                            nodes.push_back(Node::Function(static_cast<Hash>(BuiltinOp::Mul), 2));
                        }
                        appendOpAndScale(nodes, p, 2);
                        TryInsert(nt, Tree(std::move(nodes)).UpdateNodes());
                    });
            }
        }
    }

    executor.run(taskflow).get(); // .wait() would silently drop an exception thrown by a TryInsert task
}

void EnumerationEngine::Build(tf::Executor& executor, Operon::ReportCallback shouldStop)
{
    SeedTerminals();
    // Search the fringe required to recover candidates reduced below their nominal budget.
    for (std::size_t budget = 1; budget <= workingCeiling_; ++budget) {
        for (auto nt : ProcessingOrder) {
            ProcessNonterminal(executor, nt, budget);
        }
        if (shouldStop && shouldStop()) {
            return;
        }
    }
}

void EnumerationEngine::Build(Operon::ReportCallback shouldStop, std::size_t threads)
{
    if (threads == 0) {
        threads = std::thread::hardware_concurrency();
    }
    tf::Executor executor(threads);
    Build(executor, std::move(shouldStop));
}

auto MakeObjectiveScorer(gsl::not_null<Operon::EvaluatorBase const*> evaluator) -> EnumerationScorer
{
    // Scalar ranking cannot represent a multi-objective evaluator.
    EXPECT(evaluator->ObjectiveCount() == 1);
    return [evaluator](Operon::RandomGenerator& rng, Operon::Tree const& tree, double /*structureBits*/,
               Operon::Span<Operon::Scalar> buf) -> EnumerationScore {
        Operon::Individual ind { 1 };
        ind.Genotype = tree;
        auto fitness = (*evaluator)(rng, ind, buf);
        return EnumerationScore { .Score = fitness.front() };
    };
}

GrammarEnumerationAlgorithm::GrammarEnumerationAlgorithm(EnumerationConfig config, Operon::Grammar grammar,
    gsl::not_null<Operon::OptimizerBase const*> optimizer, EnumerationScorer scorer, Operon::RandomGenerator& rng)
    : config_(config)
    , engine_(std::move(grammar), config.MaxComplexity, rng, config.Pruning)
    , optimizer_(optimizer)
    , scorer_(std::move(scorer))
{
}

void GrammarEnumerationAlgorithm::ConsiderBest(EnumerationResult result)
{


    // TopK == 0 intentionally retains no results.
    if (config_.TopK == 0) {
        return;
    }

    // Keep results ordered by score, then canonical key.
    auto lessThan = [](EnumerationResult const& a, EnumerationResult const& b) {
        if (a.Score != b.Score) {
            return a.Score < b.Score;
        }
        return a.CanonicalKey < b.CanonicalKey;
    };
    if (best_.size() >= config_.TopK && !lessThan(result, best_.back())) {
        return;
    }

    auto pos = std::ranges::upper_bound(best_, result, lessThan);
    best_.insert(pos, std::move(result));
    if (best_.size() > config_.TopK) {
        best_.pop_back();
    }
}

void GrammarEnumerationAlgorithm::Run(
    tf::Executor& executor, Operon::RandomGenerator& rng, Operon::ReportCallback report)
{
    // Unfitted candidates have meaningless ranks.
    EXPECT(optimizer_->Iterations() > 0);

    Operon::ReportCallback shouldStop = [&]() -> bool {
        if (StopRequested()) {
            return true;
        }
        if (report && report()) {
            RequestStop();
            return true;
        }
        return false;
    };
    engine_.Build(executor, [&]() { return shouldStop(); });
    if (StopRequested()) {
        return;
    }

    // Build first, then select one deterministic representative per canonical class.
    struct ClassMember {
        Operon::Tree Candidate;
        std::size_t Complexity {};
        std::size_t BucketSize {};
    };
    auto better = [](ClassMember const& a, ClassMember const& b) {
        if (a.BucketSize != b.BucketSize) {
            return a.BucketSize < b.BucketSize;
        }
        if (a.Complexity != b.Complexity) {
            return a.Complexity < b.Complexity;
        }
        auto lessNode = [](Node const& lhs, Node const& rhs) {
            return std::tie(lhs.HashValue, lhs.CalculatedHashValue, lhs.Value, lhs.Arity, lhs.Length, lhs.Depth,
                       lhs.Level, lhs.Parent, lhs.Type, lhs.IsEnabled, lhs.Optimize, lhs.RefTo)
                < std::tie(rhs.HashValue, rhs.CalculatedHashValue, rhs.Value, rhs.Arity, rhs.Length, rhs.Depth,
                       rhs.Level, rhs.Parent, rhs.Type, rhs.IsEnabled, rhs.Optimize, rhs.RefTo);
        };
        return std::ranges::lexicographical_compare(a.Candidate.Nodes(), b.Candidate.Nodes(), lessNode);
    };

    std::unordered_map<std::string, ClassMember> representatives; // canonical Key -> current best representative
    for (std::size_t budget = 1; budget <= engine_.MaxComplexity(); ++budget) {
        auto bucket = engine_.Bucket(GrammarSymbol::Expression, budget);
        if (bucket.empty()) {
            continue;
        }
        for (auto const& tree : bucket) {
            auto canon = Operon::CanonicalizeEnumerationTree(tree);
            ClassMember candidate { .Candidate = std::move(canon.Representative), .Complexity = budget, .BucketSize = bucket.size() };
            if (auto it = representatives.find(canon.Key); it != representatives.end()) {
                if (better(candidate, it->second)) {
                    it->second = std::move(candidate);
                }
            } else {
                representatives.emplace(std::move(canon.Key), std::move(candidate));
            }
        }
    }
    if (representatives.empty()) {
        return;
    }

    // A stable key order makes worker RNG assignment reproducible per worker count.
    std::vector<std::pair<std::string, ClassMember>> repList(
        std::make_move_iterator(representatives.begin()), std::make_move_iterator(representatives.end()));
    std::ranges::sort(repList, {}, [](auto const& p) { return p.first; });

    // Each worker owns its evaluation buffer. Each representative gets a
    // seed in stable-key order, so task scheduling cannot affect its fit.
    EXPECT(config_.EvaluationBufferSize > 0);
    Operon::CoefficientOptimizer coeffOptimizer { optimizer_ };
    auto const numWorkers = executor.num_workers();
    std::vector<Operon::RandomGenerator::result_type> candidateSeeds(repList.size());
    for (auto& seed : candidateSeeds) {
        seed = rng();
    }
    std::vector<std::vector<Operon::Scalar>> workerBufs(
        numWorkers, std::vector<Operon::Scalar>(config_.EvaluationBufferSize));

    // Only TopK maintenance is shared.
    std::mutex bestMutex;

    // Report only between taskflow batches, when BestTrees() is coherent. This
    // bounds the otherwise potentially dominant fitting phase without putting
    // callback synchronization on every candidate.
    auto const batchSize = std::max<std::size_t>(numWorkers, 1) * 4;
    for (std::size_t first = 0; first < repList.size() && !StopRequested(); first += batchSize) {
        auto const last = std::min(first + batchSize, repList.size());
        tf::Taskflow taskflow;
        taskflow.for_each_index(first, last, std::size_t { 1 }, [&](std::size_t i) {
            if (StopRequested()) {
                return;
            }
            auto const worker = executor.this_worker_id();
            // Task bodies must have an assigned worker slot.
            EXPECT(worker >= 0);
            auto const slot = static_cast<std::size_t>(worker);
            Operon::RandomGenerator localRng(candidateSeeds[i]);
            auto& evalBuf = workerBufs[slot];

            auto const& [key, member] = repList[i];
            // Rank the fitted tree, not the optimizer's internal loss.
            auto tree = std::get<0>(coeffOptimizer(localRng, member.Candidate));
            tree.Reduce();
            tree.Simplify();
            // The pre-canonical bucket size is the structural code length.
            auto const structureBits = std::log2(static_cast<double>(member.BucketSize));
            auto score = scorer_(localRng, tree, structureBits, evalBuf);

            std::scoped_lock lock(bestMutex);
            ConsiderBest(EnumerationResult {
                .Score = score.Score,
                .NegativeLogLikelihood = score.NegativeLogLikelihood,
                .ParameterCodeBits = score.ParameterCodeBits,
                .StructureCodeBits = score.StructureCodeBits,
                .CanonicalKey = key,
                .Tree = std::move(tree),
            });
        });
        executor.run(taskflow).get(); // fitting failures must reach the caller
        if (shouldStop()) {
            break;
        }
    }
}

void GrammarEnumerationAlgorithm::Run(Operon::RandomGenerator& rng, Operon::ReportCallback report, std::size_t threads)
{
    if (threads == 0) {
        threads = std::thread::hardware_concurrency();
    }
    tf::Executor executor(threads);
    Run(executor, rng, std::move(report));
}

} // namespace Operon
