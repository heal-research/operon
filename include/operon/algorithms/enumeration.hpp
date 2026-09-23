// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_ALGORITHMS_ENUMERATION_HPP
#define OPERON_ALGORITHMS_ENUMERATION_HPP

#include <array>
#include <cmath>
#include <mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <gsl/pointers>
#include <gtl/phmap.hpp>

#include "operon/algorithms/enumeration_canonicalizer.hpp" // for CanonicalizeEnumerationTree
#include "operon/algorithms/stoppable.hpp" // for Operon::ReportCallback, StoppableAlgorithm
#include "operon/core/grammar.hpp"
#include "operon/core/tree.hpp"
#include "operon/hash/content_hash.hpp"
#include "operon/hash/zobrist.hpp"
#include "operon/information_criteria/minimum_description_length.hpp" // for ParameterDescriptionLength
#include "operon/operators/evaluator.hpp" // for EvaluatorBase, Individual, detail::ProfileSigma, FitLinearScaling
#include "operon/operators/local_search.hpp" // for CoefficientOptimizer, OptimizerBase/FitResult/FitFailure fwd decls
#include "operon/operon_export.hpp"
#include "operon/random/random.hpp"

// forward declaration - keeps taskflow.hpp (heavy) out of this public header; only enumeration.cpp/nsga2.cpp-style
// .cpp files that actually build a Taskflow need the full definition.
namespace tf {
class Executor;
}

namespace Operon {

// Counts all non-Constant nodes.
[[nodiscard]] OPERON_EXPORT auto SymbolicComplexity(Operon::Tree const& tree) noexcept -> std::size_t;

// Bottom-up grammar enumerator. Build deduplicates reduced trees; fitting and ranking occur separately.
class OPERON_EXPORT EnumerationEngine {
public:
    EnumerationEngine(Operon::Grammar grammar, std::size_t maxComplexity, Operon::RandomGenerator& rng);

    // Builds ordered budget levels; each level parallelizes candidate construction.
    void Build(tf::Executor& executor, Operon::ReportCallback shouldStop = {});

    // Uses `threads` workers; zero selects hardware concurrency.
    void Build(Operon::ReportCallback shouldStop = {}, std::size_t threads = 0);

    [[nodiscard]] auto Bucket(GrammarSymbol nt, std::size_t budget) const -> std::span<Operon::Tree const>;

    [[nodiscard]] auto GetGrammar() const -> Operon::Grammar const& { return grammar_; }
    [[nodiscard]] auto MaxComplexity() const -> std::size_t { return maxComplexity_; }

private:
    void SeedTerminals();
    void ProcessNonterminal(tf::Executor& executor, GrammarSymbol nt, std::size_t budget);
    auto TryInsert(GrammarSymbol nt, Operon::Tree tree) -> bool;

    Operon::Grammar grammar_;
    std::size_t maxComplexity_;
    std::size_t workingCeiling_;
    Operon::Zobrist zobrist_;
    std::vector<std::vector<std::vector<Operon::Tree>>> buckets_;
    std::vector<std::vector<gtl::parallel_flat_hash_set_m<Operon::Hash>>> seen_;
    std::vector<std::vector<std::mutex>> bucketMutex_;
};

// Candidate ranking criterion.
enum class EnumerationRanking : uint8_t {
    MinimumDescriptionLength,
    Objective,
};

struct EnumerationConfig {
    std::size_t MaxComplexity { 20 };
    std::size_t TopK { 10 }; // how many best-fitness models to retain (see GrammarEnumerationAlgorithm::BestTrees)
    EnumerationRanking Ranking { EnumerationRanking::MinimumDescriptionLength };
    // Scratch-buffer size required by the scorer.
    std::size_t EvaluationBufferSize { 0 };
};

// Lower scores are better. MDL uses bits; objective mode leaves component fields NaN.
struct EnumerationScore {
    Operon::Scalar Score {};
    double NegativeLogLikelihood { std::numeric_limits<double>::quiet_NaN() };
    double ParameterCodeBits { std::numeric_limits<double>::quiet_NaN() };
    double StructureCodeBits { std::numeric_limits<double>::quiet_NaN() };
};

// Scores one fitted tree. `structureBits` is supplied by the caller.
using EnumerationScorer
    = Operon::MoveOnlyFunction<EnumerationScore(Operon::RandomGenerator&, Tree const&, double, Span<Scalar>)>;

// Builds an MDL scorer. Empty sigma profiles Gaussian noise; otherwise sigma is fixed.
template <typename DTable, Concepts::Likelihood Lik>
    requires Concepts::HasFisherMatrix<Lik>
auto MakeMdlScorer(gsl::not_null<Operon::Problem const*> problem, gsl::not_null<DTable const*> dtable,
    std::vector<Operon::Scalar> sigma = {}) -> EnumerationScorer
{
    return [problem, dtable, sigma = std::move(sigma)](Operon::RandomGenerator& /*rng*/, Operon::Tree const& tree,
               double structureBits, Operon::Span<Operon::Scalar> buf) -> EnumerationScore {
        auto const trainingRange = problem->TrainingRange();
        auto const* dataset = problem->GetDataset();
        auto parameters = tree.GetCoefficients();

        EXPECT(buf.size() >= trainingRange.Size());
        auto yPred = buf.subspan(0, trainingRange.Size());
        Operon::Interpreter<Operon::Scalar, DTable> const interpreter { dtable.get(), dataset, &tree };
        interpreter.Evaluate(parameters, trainingRange, yPred);

        auto yTrue = problem->TargetValues(trainingRange);
        auto const weights = problem->Weights(trainingRange).value_or(Operon::Span<Operon::Scalar const> {});
        std::optional<Operon::LinearScaling> scaling {};
        if (problem->LinearScalingEnabled()) {
            scaling = Operon::FitLinearScaling(yPred, yTrue, weights, problem->LinearScalingOmitsNonFinite());
            scaling->ApplyInPlace(yPred);
        }

        Operon::Scalar profiledSigma {};
        if (sigma.empty() && Lik::UsesSigma) {
            profiledSigma = Operon::detail::ProfileSigma(yPred, yTrue);
        }
        auto const effectiveSigma = (sigma.empty() && Lik::UsesSigma)
            ? Operon::Span<Operon::Scalar const> { &profiledSigma, 1 }
            : Operon::Span<Operon::Scalar const> { sigma };

        Eigen::Matrix<Operon::Scalar, -1, -1> jac = interpreter.JacRev(parameters, trainingRange);
        if (scaling) {
            jac *= static_cast<Operon::Scalar>(scaling->Scale);
        }
        auto fisherMatrix
            = Lik::ComputeFisherMatrix(yPred, { jac.data(), static_cast<std::size_t>(jac.size()) }, effectiveSigma);
        auto fisherDiag = fisherMatrix.diagonal().array();

        auto const nllNats = static_cast<double>(Lik::ComputeLikelihood(yPred, yTrue, effectiveSigma));
        auto const paramNats = Operon::ParameterDescriptionLength(parameters, fisherDiag);

        constexpr double Ln2 = 0.6931471805599453094;
        auto const paramBits = paramNats / Ln2;
        auto score = (nllNats / Ln2) + paramBits + structureBits;
        if (!std::isfinite(score)) {
            score = static_cast<double>(EvaluatorBase::ErrMax);
        }

        return EnumerationScore {
            .Score = static_cast<Operon::Scalar>(score),
            .NegativeLogLikelihood = nllNats,
            .ParameterCodeBits = paramBits,
            .StructureCodeBits = structureBits,
        };
    };
}

// Builds a scalar objective scorer; multi-objective evaluators are rejected.
[[nodiscard]] OPERON_EXPORT auto MakeObjectiveScorer(gsl::not_null<Operon::EvaluatorBase const*> evaluator)
    -> EnumerationScorer;

// Ranked canonical representative.
struct EnumerationResult {
    Operon::Scalar Score {};
    double NegativeLogLikelihood { std::numeric_limits<double>::quiet_NaN() };
    double ParameterCodeBits { std::numeric_limits<double>::quiet_NaN() };
    double StructureCodeBits { std::numeric_limits<double>::quiet_NaN() };
    std::string CanonicalKey;
    Operon::Tree Tree;
};

// Builds, groups, fits, and ranks grammar-enumerated candidates.
class OPERON_EXPORT GrammarEnumerationAlgorithm : public StoppableAlgorithm {
public:
    GrammarEnumerationAlgorithm(EnumerationConfig config, Operon::Grammar grammar,
        gsl::not_null<Operon::OptimizerBase const*> optimizer, EnumerationScorer scorer, Operon::RandomGenerator& rng);

    void Run(tf::Executor& executor, Operon::RandomGenerator& rng, Operon::ReportCallback report = {});
    void Run(Operon::RandomGenerator& rng, Operon::ReportCallback report = {}, std::size_t threads = 0);

    // Results are sorted by score, then canonical key, and capped at TopK.
    [[nodiscard]] auto BestTrees() const -> std::span<EnumerationResult const> { return best_; }

    [[nodiscard]] auto GetEngine() const -> EnumerationEngine const& { return engine_; }

private:
    void ConsiderBest(EnumerationResult result);

    EnumerationConfig config_;
    EnumerationEngine engine_;
    gsl::not_null<Operon::OptimizerBase const*> optimizer_;
    EnumerationScorer scorer_;
    std::vector<EnumerationResult> best_;
};

} // namespace Operon

#endif
