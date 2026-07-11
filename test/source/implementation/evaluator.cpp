// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "operon/core/dataset.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/types.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/optimizer/likelihood/gaussian_likelihood.hpp"
#include "operon/optimizer/likelihood/poisson_likelihood.hpp"
#include "operon/parser/infix.hpp"
#include "operon/random/random.hpp"

namespace Operon::Test {

// ──────────────────────────────────────────────────────────────────────────────
// Shared fixture: y = X1 + X2 + X3 on 500 rows.
//   tree        — variable weights = 0.1 (imperfect fit, SSR > 0)
//   perfectTree — variable weights = 1.0 (exact fit, SSR = 0; tests epsilon clamp)
// ──────────────────────────────────────────────────────────────────────────────
struct EvaluatorFixture {
    static constexpr auto Nrow { 500 };
    static constexpr auto Ncol { 4 }; // X1, X2, X3, y

    Operon::RandomGenerator rng{0}; // NOLINT(readability-identifier-naming)
    Eigen::Array<Operon::Scalar, -1, -1> data{Nrow, Ncol}; // NOLINT(readability-identifier-naming)
    Operon::Dataset ds; // NOLINT(readability-identifier-naming)
    Operon::Tree tree; // NOLINT(readability-identifier-naming)
    Operon::Tree perfectTree; // NOLINT(readability-identifier-naming)
    using DTable = DispatchTable<Operon::Scalar>;
    DTable dtable; // NOLINT(readability-identifier-naming)
    Operon::Problem problem; // NOLINT(readability-identifier-naming)

    EvaluatorFixture()
        : ds([&]() -> Operon::Dataset {
            for (auto i = 0; i < Ncol - 1; ++i) {
                auto col = data.col(i);
                std::generate(col.begin(), col.end(), [&]() -> float { return Operon::Random::Uniform(rng, -1.0F, +1.0F); });
            }
            data.col(Ncol - 1) = data.col(0) + data.col(1) + data.col(2);
            return Operon::Dataset(gsl::not_null{data.data()}, Nrow, Ncol);
        }())
        , tree([&]() -> Tree {
            auto t = InfixParser::Parse("X1 + X2 + X3", ds);
            for (auto& node : t.Nodes()) {
                if (node.IsVariable()) { node.Value = static_cast<Operon::Scalar>(0.1); }
            }
            return t;
        }())
        , perfectTree([&]() -> Tree {
            auto t = InfixParser::Parse("X1 + X2 + X3", ds);
            for (auto& node : t.Nodes()) {
                if (node.IsVariable()) { node.Value = static_cast<Operon::Scalar>(1.0); }
            }
            return t;
        }())
        , problem(&ds)
    {
        problem.SetTrainingRange({0, Nrow});
        problem.SetTestRange({0, Nrow});
        problem.SetTarget("X4");
    }

    static auto MakeIndividual(Operon::Tree const& t) -> Operon::Individual {
        Operon::Individual ind;
        ind.Genotype = t;
        return ind;
    }
};

// ──────────────────────────────────────────────────────────────────────────────
// Poisson likelihood static methods
// ──────────────────────────────────────────────────────────────────────────────
TEST_CASE("Poisson likelihood static methods", "[likelihood]")
{
    constexpr auto n { 100 };

    SECTION("LogInput=true, log-rate=0, count=1: NLL = n") {
        using Lik = PoissonLikelihood<Operon::Scalar>;
        std::vector<Operon::Scalar> pred(n, 0.0F);   // log-rate = 0
        std::vector<Operon::Scalar> target(n, 1.0F); // count = 1
        // f(0, 1) = exp(0) - 0·1 + lgamma(2) = 1 - 0 + 0 = 1
        auto nll = Lik::ComputeLikelihood(pred, target, {});
        CHECK_THAT(static_cast<double>(nll), Catch::Matchers::WithinRel(static_cast<double>(n), 1e-5));
    }

    SECTION("LogInput=false, rate=1, count=1: NLL = n") {
        using Lik = PoissonLikelihood<Operon::Scalar, false>;
        std::vector<Operon::Scalar> pred(n, 1.0F);   // rate = 1
        std::vector<Operon::Scalar> target(n, 1.0F); // count = 1
        // f(1, 1) = 1 - 1·log(1) + lgamma(2) = 1 - 0 + 0 = 1
        auto nll = Lik::ComputeLikelihood(pred, target, {});
        CHECK_THAT(static_cast<double>(nll), Catch::Matchers::WithinRel(static_cast<double>(n), 1e-5));
    }

    SECTION("LogInput=false, scalar weight scales rate: f(1,1,2) = 2 - log(2) per obs") {
        using Lik = PoissonLikelihood<Operon::Scalar, false>;
        std::vector<Operon::Scalar> pred(n, 1.0F);
        std::vector<Operon::Scalar> target(n, 1.0F);
        std::vector<Operon::Scalar> w(1, 2.0F);
        // f(x,y,w) = f(w*x, y) = 2 - 1·log(2) + 0 = 2 - log(2)
        auto nll = Lik::ComputeLikelihood(pred, target, w);
        auto const expected = static_cast<double>(n) * (2.0 - std::numbers::ln2);
        CHECK_THAT(static_cast<double>(nll), Catch::Matchers::WithinRel(expected, 1e-5));
    }

    SECTION("PoissonLoss::ComputeLikelihood delegates to PoissonLikelihood") {
        std::vector<Operon::Scalar> pred(n, 0.0F);
        std::vector<Operon::Scalar> target(n, 1.0F);
        CHECK(PoissonLoss<Operon::Scalar>::ComputeLikelihood(pred, target, {})
           == PoissonLikelihood<Operon::Scalar>::ComputeLikelihood(pred, target, {}));
    }

    SECTION("FisherMatrix LogInput=true: pred=0, J=I => F=I") {
        // F = J^T · diag(exp(pred)) · J; exp(0)=1 => F = I
        using Lik = PoissonLikelihood<Operon::Scalar>;
        constexpr auto m { 10 };
        std::vector<Operon::Scalar> pred(m, 0.0F);
        Eigen::Matrix<Operon::Scalar, -1, -1> jac = Eigen::Matrix<Operon::Scalar, -1, -1>::Identity(m, m);
        auto fisher = Lik::ComputeFisherMatrix(pred, {jac.data(), static_cast<std::size_t>(jac.size())}, {});
        REQUIRE(fisher.rows() == m);
        REQUIRE(fisher.cols() == m);
        CHECK_THAT(static_cast<double>(fisher.diagonal().minCoeff()), Catch::Matchers::WithinRel(1.0, 1e-5));
        CHECK_THAT(static_cast<double>(fisher.diagonal().maxCoeff()), Catch::Matchers::WithinRel(1.0, 1e-5));
        auto const offDiag = (fisher - Eigen::Matrix<Operon::Scalar, -1, -1>::Identity(m, m)).norm();
        CHECK_THAT(static_cast<double>(offDiag), Catch::Matchers::WithinAbs(0.0, 1e-5));
    }

    SECTION("FisherMatrix LogInput=false: pred=1, J=I => F=I") {
        // F = J^T · diag(1/pred) · J; 1/1=1 => F = I
        using Lik = PoissonLikelihood<Operon::Scalar, false>;
        constexpr auto m { 10 };
        std::vector<Operon::Scalar> pred(m, 1.0F);
        Eigen::Matrix<Operon::Scalar, -1, -1> jac = Eigen::Matrix<Operon::Scalar, -1, -1>::Identity(m, m);
        auto fisher = Lik::ComputeFisherMatrix(pred, {jac.data(), static_cast<std::size_t>(jac.size())}, {});
        REQUIRE(fisher.rows() == m);
        REQUIRE(fisher.cols() == m);
        CHECK_THAT(static_cast<double>(fisher.diagonal().minCoeff()), Catch::Matchers::WithinRel(1.0, 1e-5));
        CHECK_THAT(static_cast<double>(fisher.diagonal().maxCoeff()), Catch::Matchers::WithinRel(1.0, 1e-5));
        auto const offDiag = (fisher - Eigen::Matrix<Operon::Scalar, -1, -1>::Identity(m, m)).norm();
        CHECK_THAT(static_cast<double>(offDiag), Catch::Matchers::WithinAbs(0.0, 1e-5));
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Gaussian per-sample sigma
// ──────────────────────────────────────────────────────────────────────────────
TEST_CASE("Gaussian per-sample sigma", "[likelihood]")
{
    using Lik = GaussianLikelihood<Operon::Scalar>;
    constexpr auto n { 100 };
    constexpr auto s { 2.0F };

    std::vector<Operon::Scalar> pred(n, 1.0F);
    std::vector<Operon::Scalar> target(n, 0.0F);
    std::vector<Operon::Scalar> scalarSigma(1, s);
    std::vector<Operon::Scalar> perSampleSigma(n, s);

    SECTION("ComputeLikelihood: uniform per-sample sigma matches scalar sigma") {
        auto const nllScalar    = Lik::ComputeLikelihood(pred, target, scalarSigma);
        auto const nllPerSample = Lik::ComputeLikelihood(pred, target, perSampleSigma);
        CHECK_THAT(static_cast<double>(nllPerSample),
                   Catch::Matchers::WithinRel(static_cast<double>(nllScalar), 1e-4));
    }

    SECTION("ComputeFisherMatrix: uniform per-sample sigma matches scalar sigma") {
        constexpr auto m { 10 };
        std::vector<Operon::Scalar> p(m, 0.0F);
        Eigen::Matrix<Operon::Scalar, -1, -1> jac = Eigen::Matrix<Operon::Scalar, -1, -1>::Identity(m, m);
        std::vector<Operon::Scalar> sig1(1, s);
        std::vector<Operon::Scalar> sigN(m, s);
        auto const f1 = Lik::ComputeFisherMatrix(p, {jac.data(), static_cast<std::size_t>(jac.size())}, sig1);
        auto const fN = Lik::ComputeFisherMatrix(p, {jac.data(), static_cast<std::size_t>(jac.size())}, sigN);
        auto const diff = (f1 - fN).norm();
        CHECK_THAT(static_cast<double>(diff), Catch::Matchers::WithinAbs(0.0, 1e-5));
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// MDL evaluator
// ──────────────────────────────────────────────────────────────────────────────
TEST_CASE("MDL evaluator", "[evaluator]")
{
    EvaluatorFixture fix;
    using DTable = EvaluatorFixture::DTable;

    SECTION("Gaussian / profiled sigma: finite positive result") {
        MinimumDescriptionLengthEvaluator<DTable, GaussianLikelihood<Operon::Scalar>> const ev{&fix.problem, &fix.dtable};
        auto ind = EvaluatorFixture::MakeIndividual(fix.tree);
        auto const result = ev(fix.rng, ind);
        REQUIRE(result.size() == 1);
        CHECK(std::isfinite(result[0]));
        CHECK(result[0] > 0);
    }

    SECTION("Gaussian / fixed sigma: finite positive result") {
        MinimumDescriptionLengthEvaluator<DTable, GaussianLikelihood<Operon::Scalar>> const ev{&fix.problem, &fix.dtable};
        ev.SetSigma({0.5F});
        auto ind = EvaluatorFixture::MakeIndividual(fix.tree);
        auto const result = ev(fix.rng, ind);
        REQUIRE(result.size() == 1);
        CHECK(std::isfinite(result[0]));
        CHECK(result[0] > 0);
    }

    SECTION("Poisson: finite result") {
        MinimumDescriptionLengthEvaluator<DTable, PoissonLikelihood<Operon::Scalar>> const ev{&fix.problem, &fix.dtable};
        auto ind = EvaluatorFixture::MakeIndividual(fix.tree);
        auto const result = ev(fix.rng, ind);
        REQUIRE(result.size() == 1);
        CHECK(std::isfinite(result[0]));
    }

    SECTION("Gaussian / profiled sigma: SSR=0 does not produce NaN (epsilon clamp)") {
        MinimumDescriptionLengthEvaluator<DTable, GaussianLikelihood<Operon::Scalar>> const ev{&fix.problem, &fix.dtable};
        auto ind = EvaluatorFixture::MakeIndividual(fix.perfectTree);
        auto const result = ev(fix.rng, ind);
        REQUIRE(result.size() == 1);
        CHECK(std::isfinite(result[0]));
    }

    // End-to-end regression test (through the real evaluator path, not just
    // detail::ProfileSigma in isolation): EvaluatorBase::Evaluate's contract
    // permits buf.size() > TrainingRange().Size(), and the operator() body
    // now slices down to exactly TrainingRange().Size() before using it
    // anywhere (interpreter output, ComputeFisherMatrix's row-count
    // inference) - so an oversized buffer must produce the same result as
    // an exactly-sized one, not crash or silently diverge.
    SECTION("Gaussian / profiled sigma: oversized buffer matches exact-size buffer") {
        MinimumDescriptionLengthEvaluator<DTable, GaussianLikelihood<Operon::Scalar>> const ev{&fix.problem, &fix.dtable};
        auto ind = EvaluatorFixture::MakeIndividual(fix.tree);

        std::vector<Operon::Scalar> exactBuf(EvaluatorFixture::Nrow);
        auto const exactResult = ev(fix.rng, ind, exactBuf);

        std::vector<Operon::Scalar> oversizedBuf(EvaluatorFixture::Nrow + 50);
        auto const oversizedResult = ev(fix.rng, ind, oversizedBuf);

        REQUIRE(exactResult.size() == 1);
        REQUIRE(oversizedResult.size() == 1);
        CHECK(std::isfinite(oversizedResult[0]));
        CHECK(oversizedResult[0] == exactResult[0]);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// FBF evaluator
// ──────────────────────────────────────────────────────────────────────────────
TEST_CASE("FBF evaluator", "[evaluator]")
{
    EvaluatorFixture fix;
    using DTable = EvaluatorFixture::DTable;

    SECTION("Gaussian / profiled sigma: finite positive result") {
        FractionalBayesFactorEvaluator<DTable, GaussianLikelihood<Operon::Scalar>> const ev{&fix.problem, &fix.dtable};
        auto ind = EvaluatorFixture::MakeIndividual(fix.tree);
        auto const result = ev(fix.rng, ind);
        REQUIRE(result.size() == 1);
        CHECK(std::isfinite(result[0]));
        CHECK(result[0] > 0);
    }

    SECTION("Gaussian / fixed sigma: finite positive result") {
        FractionalBayesFactorEvaluator<DTable, GaussianLikelihood<Operon::Scalar>> const ev{&fix.problem, &fix.dtable};
        ev.SetSigma({0.5F});
        auto ind = EvaluatorFixture::MakeIndividual(fix.tree);
        auto const result = ev(fix.rng, ind);
        REQUIRE(result.size() == 1);
        CHECK(std::isfinite(result[0]));
        CHECK(result[0] > 0);
    }

    SECTION("Poisson: finite result") {
        FractionalBayesFactorEvaluator<DTable, PoissonLikelihood<Operon::Scalar>> const ev{&fix.problem, &fix.dtable};
        auto ind = EvaluatorFixture::MakeIndividual(fix.tree);
        auto const result = ev(fix.rng, ind);
        REQUIRE(result.size() == 1);
        CHECK(std::isfinite(result[0]));
    }

    SECTION("Poisson FBF includes NLL contribution (regression: NLL was silently zero before fix)") {
        // Before the fix, the Poisson NLL was always 0 so FBF = fComplexity + cParameters.
        // After the fix, Poisson and Gaussian NLLs are computed differently, so results must differ.
        FractionalBayesFactorEvaluator<DTable, GaussianLikelihood<Operon::Scalar>> const evG{&fix.problem, &fix.dtable};
        FractionalBayesFactorEvaluator<DTable, PoissonLikelihood<Operon::Scalar>> const  evP{&fix.problem, &fix.dtable};
        auto ind = EvaluatorFixture::MakeIndividual(fix.tree);
        auto const rG = evG(fix.rng, ind);
        auto const rP = evP(fix.rng, ind);
        CHECK(rG[0] != rP[0]);
    }

    // Same regression guard as MDL's - see the comment there.
    SECTION("Gaussian / profiled sigma: oversized buffer matches exact-size buffer") {
        FractionalBayesFactorEvaluator<DTable, GaussianLikelihood<Operon::Scalar>> const ev{&fix.problem, &fix.dtable};
        auto ind = EvaluatorFixture::MakeIndividual(fix.tree);

        std::vector<Operon::Scalar> exactBuf(EvaluatorFixture::Nrow);
        auto const exactResult = ev(fix.rng, ind, exactBuf);

        std::vector<Operon::Scalar> oversizedBuf(EvaluatorFixture::Nrow + 50);
        auto const oversizedResult = ev(fix.rng, ind, oversizedBuf);

        REQUIRE(exactResult.size() == 1);
        REQUIRE(oversizedResult.size() == 1);
        CHECK(std::isfinite(oversizedResult[0]));
        CHECK(oversizedResult[0] == exactResult[0]);
    }

    SECTION("Gaussian / profiled sigma: SSR=0 does not produce NaN (epsilon clamp)") {
        FractionalBayesFactorEvaluator<DTable, GaussianLikelihood<Operon::Scalar>> const ev{&fix.problem, &fix.dtable};
        auto ind = EvaluatorFixture::MakeIndividual(fix.perfectTree);
        auto const result = ev(fix.rng, ind);
        REQUIRE(result.size() == 1);
        CHECK(std::isfinite(result[0]));
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// LikelihoodEvaluator
// ──────────────────────────────────────────────────────────────────────────────
TEST_CASE("LikelihoodEvaluator", "[evaluator]")
{
    EvaluatorFixture fix;
    using DTable = EvaluatorFixture::DTable;

    SECTION("Gaussian: finite result") {
        // LikelihoodEvaluator only overrides the 3-arg operator() (unlike
        // MDL/FBF, which also provide their own 2-arg override), so this
        // always calls the buffered form directly rather than through
        // EvaluatorBase::Evaluate.
        GaussianLikelihoodEvaluator<DTable> const ev{&fix.problem, &fix.dtable};
        auto ind = EvaluatorFixture::MakeIndividual(fix.tree);
        std::vector<Operon::Scalar> buf(EvaluatorFixture::Nrow);
        auto const result = ev(fix.rng, ind, buf);
        REQUIRE(result.size() == 1);
        CHECK(std::isfinite(result[0]));
    }

    // Same regression guard as MDL/FBF's - see MinimumDescriptionLengthEvaluator's
    // operator() for why the slice fix is needed.
    SECTION("Gaussian: oversized buffer matches exact-size buffer") {
        GaussianLikelihoodEvaluator<DTable> const ev{&fix.problem, &fix.dtable};
        auto ind = EvaluatorFixture::MakeIndividual(fix.tree);

        std::vector<Operon::Scalar> exactBuf(EvaluatorFixture::Nrow);
        auto const exactResult = ev(fix.rng, ind, exactBuf);

        std::vector<Operon::Scalar> oversizedBuf(EvaluatorFixture::Nrow + 50);
        auto const oversizedResult = ev(fix.rng, ind, oversizedBuf);

        REQUIRE(exactResult.size() == 1);
        REQUIRE(oversizedResult.size() == 1);
        CHECK(std::isfinite(oversizedResult[0]));
        CHECK(oversizedResult[0] == exactResult[0]);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// detail::ProfileSigma
// ──────────────────────────────────────────────────────────────────────────────
TEST_CASE("ProfileSigma", "[evaluator]")
{
    SECTION("estimated longer than target: bounded by the shorter span, no out-of-bounds read") {
        // Regression guard: EvaluatorBase::Evaluate's contract only requires
        // buf.size() >= TrainingRange().Size(), not equality, so a caller
        // may legitimately hand ProfileSigma an oversized `estimated` span.
        // It must not index `target` past its own (shorter) length.
        std::vector<Operon::Scalar> const target{1.0F, 2.0F, 3.0F};
        std::vector<Operon::Scalar> estimated{1.1F, 2.1F, 3.1F, 999.F, 999.F}; // 2 extra, unrelated entries
        auto const sigma = Operon::detail::ProfileSigma(estimated, target);
        CHECK(std::isfinite(sigma));
        // sqrt(SSR/n) over exactly the 3 overlapping entries (residual 0.1 each): sqrt(3*0.01/3) = 0.1
        CHECK_THAT(static_cast<double>(sigma), Catch::Matchers::WithinRel(0.1, 1e-3));
    }

    SECTION("exact-size spans: matches the straightforward SSR/n computation") {
        std::vector<Operon::Scalar> const estimated{1.0F, 2.0F, 3.0F, 4.0F};
        std::vector<Operon::Scalar> const target{0.0F, 0.0F, 0.0F, 0.0F};
        auto const sigma = Operon::detail::ProfileSigma(estimated, target);
        // SSR = 1+4+9+16 = 30, n = 4, sigma = sqrt(30/4)
        CHECK_THAT(static_cast<double>(sigma), Catch::Matchers::WithinRel(std::sqrt(30.0 / 4.0), 1e-3));
    }

    SECTION("SSR=0 clamps to epsilon rather than returning exactly zero") {
        std::vector<Operon::Scalar> const v{1.0F, 2.0F, 3.0F};
        auto const sigma = Operon::detail::ProfileSigma(v, v);
        CHECK(sigma == std::numeric_limits<Operon::Scalar>::epsilon());
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Weighted evaluator
//
// Four scenarios that justify sample weights:
//   1. No weights set       → result matches the pre-existing unweighted path.
//   2. Uniform weights      → weighted MSE == unweighted MSE (mathematical invariant).
//   3. Outlier suppression  → zero-weighting a single far-out point reduces MSE to 0
//                             when the model is otherwise perfect.
//   4. Weighted scaling     → FitLeastSquares with region-biased weights converges to
//                             the slope dominant in the high-weight region.
// ──────────────────────────────────────────────────────────────────────────────
TEST_CASE("Weighted evaluator", "[evaluator]")
{
    using DTable = DispatchTable<Operon::Scalar>;

    // ── shared data: 100 rows, Y = X1 ────────────────────────────────────────
    constexpr int N = 100;
    Eigen::Array<Operon::Scalar, -1, -1> data(N, 2); // X1, Y
    for (auto i = 0; i < N; ++i) {
        data(i, 0) = static_cast<Operon::Scalar>(i) / N; // X1 in [0, 1)
        data(i, 1) = data(i, 0);                         // Y = X1
    }
    auto makeDataset = [&]() { return Operon::Dataset(gsl::not_null{data.data()}, N, 2); };

    auto makeProblem = [&](Operon::Dataset* ds) {
        auto p = std::make_unique<Operon::Problem>(ds);
        p->SetTrainingRange({0, N});
        p->SetTestRange({0, N});
        p->SetTarget("X2");
        return p;
    };

    // single-variable expression; all variable nodes get the same coefficient (fine here, X1 only)
    auto makeInd = [&](Operon::Dataset const& ds, float varWeight) -> Operon::Individual {
        auto t = InfixParser::Parse("X1", ds);
        for (auto& node : t.Nodes()) {
            if (node.IsVariable()) { node.Value = varWeight; }
        }
        Operon::Individual ind;
        ind.Genotype = t;
        return ind;
    };

    DTable dtable;
    Operon::RandomGenerator rng{42};

    SECTION("No weights: result matches unweighted evaluator") {
        auto ds = makeDataset();
        auto problem = makeProblem(&ds);
        auto ind = makeInd(ds, 0.5F); // imperfect: predicts 0.5 * X1

        Evaluator<DTable> ev{problem.get(), &dtable, MSE{}, false};
        auto r1 = ev(rng, ind);
        REQUIRE(r1.size() == 1);
        CHECK(std::isfinite(r1[0]));
        CHECK(r1[0] > 0);

        // setting uniform weights must not change the result
        std::vector<Operon::Scalar> w(N, 1.0F);
        ds.SetWeights(w);
        auto r2 = ev(rng, ind);
        CHECK_THAT(static_cast<double>(r2[0]),
                   Catch::Matchers::WithinRel(static_cast<double>(r1[0]), 1e-5));
    }

    SECTION("Uniform weights equal unweighted result") {
        auto ds = makeDataset();
        auto problem = makeProblem(&ds);
        auto ind = makeInd(ds, 0.7F); // imperfect

        Evaluator<DTable> ev{problem.get(), &dtable, MSE{}, false};
        auto unweighted = ev(rng, ind)[0];

        std::vector<Operon::Scalar> w(N, 1.0F);
        ds.SetWeights(w);
        auto weighted = ev(rng, ind)[0];

        CHECK_THAT(static_cast<double>(weighted),
                   Catch::Matchers::WithinRel(static_cast<double>(unweighted), 1e-5));
    }

    SECTION("Outlier suppression: zero weight on outlier row gives MSE = 0 for perfect model") {
        data(0, 1) = 1000.0F; // inject outlier before dataset copies the array
        auto ds = makeDataset();
        auto problem = makeProblem(&ds);
        auto perfect = makeInd(ds, 1.0F); // predicts X1 exactly (residual != 0 at row 0)

        Evaluator<DTable> ev{problem.get(), &dtable, MSE{}, false};

        // without weights: MSE is dominated by the outlier
        auto mseUnweighted = ev(rng, perfect)[0];
        CHECK(mseUnweighted > 1.0F);

        // zero out the outlier row; perfect on all remaining rows → weighted MSE = 0
        std::vector<Operon::Scalar> w(N, 1.0F);
        w[0] = 0.0F;
        ds.SetWeights(w);
        auto mseWeighted = ev(rng, perfect)[0];
        CHECK_THAT(static_cast<double>(mseWeighted), Catch::Matchers::WithinAbs(0.0, 1e-5));
    }

    SECTION("Weighted FitLeastSquares converges to slope of high-weight region") {
        // Two datasets, each with a mixed-slope layout. In both cases the high-weight
        // region is the FIRST half so the bivariate accumulator is seeded with non-zero
        // weights before encountering any zero-weight observations. (vstat's
        // bivariate_accumulator computes f = w/(sum_w * sum_w_old); leading zero-weight
        // rows keep sum_w_old = 0, causing NaN on the first non-zero-weight row.)
        //
        // Dataset A: rows 0..49 have slope 2, rows 50..99 have slope 3.
        //   Weights = 1 on first half  → FitLeastSquares finds a ≈ 2 → weighted MSE ≈ 0.
        //
        // Dataset B: rows 0..49 have slope 3, rows 50..99 have slope 2.
        //   Weights = 1 on first half  → FitLeastSquares finds a ≈ 3 → weighted MSE ≈ 0.
        //
        // Together the two checks prove the scale adapts to whichever slope dominates
        // the weight distribution.
        auto runScalingTest = [&](float slopeHigh, float slopeLow) -> Operon::Scalar {
            Eigen::Array<Operon::Scalar, -1, -1> d(N, 2);
            for (auto i = 0; i < N / 2; ++i) {
                d(i, 0) = static_cast<Operon::Scalar>(i + 1) / N;
                d(i, 1) = slopeHigh * d(i, 0);
            }
            for (auto i = N / 2; i < N; ++i) {
                d(i, 0) = static_cast<Operon::Scalar>(i + 1) / N;
                d(i, 1) = slopeLow  * d(i, 0);
            }
            auto ds = Operon::Dataset(gsl::not_null{d.data()}, N, 2);
            auto problem = makeProblem(&ds);
            auto ind = makeInd(ds, 1.0F);
            Evaluator<DTable> ev{problem.get(), &dtable, MSE{}, /*linearScaling=*/true};
            std::vector<Operon::Scalar> w(N, 0.0F);
            std::fill(w.begin(), w.begin() + N/2, 1.0F); // weight only the first half
            ds.SetWeights(w);
            return ev(rng, ind)[0];
        };

        auto mseSlope2 = runScalingTest(2.0F, 3.0F); // first half slope 2, weighted → a ≈ 2
        auto mseSlope3 = runScalingTest(3.0F, 2.0F); // first half slope 3, weighted → a ≈ 3

        CHECK_THAT(static_cast<double>(mseSlope2), Catch::Matchers::WithinAbs(0.0, 1e-4));
        CHECK_THAT(static_cast<double>(mseSlope3), Catch::Matchers::WithinAbs(0.0, 1e-4));
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// EvaluatorBase::Evaluate (deducing-this) dispatch
//
// Regression coverage for the CRTP-to-deducing-this rewrite: the unbuffered
// 2-arg operator() overload must reach the *concrete derived* 3-arg
// override via std::invoke(self, ...) - not the base's own pure-virtual
// operator(), and not some other evaluator's override. Verified per class by
// checking the 2-arg result matches an independently-computed expectation
// (either a direct 3-arg call, or - for the one evaluator whose 3-arg body
// consumes RNG state - a separately-seeded but identical RNG sequence).
// ──────────────────────────────────────────────────────────────────────────────
TEST_CASE("EvaluatorBase::Evaluate dispatch reaches the concrete derived override", "[evaluator]")
{
    EvaluatorFixture fix;
    using DTable = EvaluatorFixture::DTable;

    SECTION("UserDefinedEvaluator") {
        int calls = 0;
        Operon::UserDefinedEvaluator ev{&fix.problem, [&](Operon::RandomGenerator&, Operon::Individual const&) -> Operon::EvaluatorBase::ReturnType {
            ++calls;
            return { Operon::Scalar{42} };
        }};
        auto ind = EvaluatorFixture::MakeIndividual(fix.tree);
        auto const result = ev(fix.rng, ind); // 2-arg
        REQUIRE(result.size() == 1);
        CHECK(result[0] == Operon::Scalar{42});
        CHECK(calls == 1); // the 3-arg override's lambda ran exactly once, not zero or twice
    }

    SECTION("MultiEvaluator") {
        Operon::Evaluator<DTable> r2{&fix.problem, &fix.dtable, Operon::R2{}};
        Operon::Evaluator<DTable> mse{&fix.problem, &fix.dtable, Operon::MSE{}};
        Operon::MultiEvaluator me{&fix.problem};
        me.Add(&r2);
        me.Add(&mse);

        auto ind = EvaluatorFixture::MakeIndividual(fix.tree);
        auto const combined = me(fix.rng, ind); // 2-arg
        auto const expectedR2  = r2(fix.rng, ind);
        auto const expectedMse = mse(fix.rng, ind);

        REQUIRE(combined.size() == 2);
        CHECK(combined[0] == expectedR2[0]);
        CHECK(combined[1] == expectedMse[0]);
    }

    SECTION("AggregateEvaluator") {
        // Aggregating a single-objective evaluator is a no-op regardless of
        // AggregateType (min/max/median/mean of one element is that
        // element), so the 2-arg result must equal the wrapped evaluator's
        // own 2-arg result exactly.
        Operon::Evaluator<DTable> inner{&fix.problem, &fix.dtable, Operon::MSE{}};
        Operon::AggregateEvaluator ae{&inner};

        auto ind = EvaluatorFixture::MakeIndividual(fix.tree);
        auto const aggregated = ae(fix.rng, ind); // 2-arg
        auto const expected   = inner(fix.rng, ind);

        REQUIRE(aggregated.size() == 1);
        CHECK(aggregated[0] == expected[0]);
    }

    SECTION("DiversityEvaluator") {
        // The 3-arg override consumes RNG state (Operon::Random::Sample per
        // sample), so two independently-seeded-but-identical RandomGenerator
        // instances must produce bit-identical results if the 2-arg path
        // reaches the same code as a direct 3-arg call.
        Operon::DiversityEvaluator dv{&fix.problem};
        std::vector<Operon::Individual> pop{
            EvaluatorFixture::MakeIndividual(fix.tree),
            EvaluatorFixture::MakeIndividual(fix.perfectTree),
        };
        dv.Prepare(pop);

        auto ind = EvaluatorFixture::MakeIndividual(fix.tree);
        Operon::RandomGenerator rngA{123};
        Operon::RandomGenerator rngB{123};

        auto const via2Arg = dv(rngA, ind); // 2-arg
        std::vector<Operon::Scalar> buf(fix.problem.TrainingRange().Size());
        auto const via3Arg = dv(rngB, ind, buf); // 3-arg, direct

        REQUIRE(via2Arg.size() == 1);
        REQUIRE(via3Arg.size() == 1);
        CHECK(std::isfinite(via2Arg[0]));
        CHECK(via2Arg[0] == via3Arg[0]);
    }
}

} // namespace Operon::Test
