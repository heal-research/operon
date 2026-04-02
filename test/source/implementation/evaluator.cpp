// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../operon_test.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/types.hpp"
#include "operon/interpreter/interpreter.hpp"
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

    Operon::RandomGenerator rng{0};
    Eigen::Array<Operon::Scalar, -1, -1> data{Nrow, Ncol};
    Operon::Dataset ds;
    Operon::Tree tree;
    Operon::Tree perfectTree;
    using DTable = DispatchTable<Operon::Scalar>;
    DTable dtable;
    Operon::Problem problem;

    EvaluatorFixture()
        : ds([&]() -> Operon::Dataset {
            for (auto i = 0; i < Ncol - 1; ++i) {
                auto col = data.col(i);
                std::generate(col.begin(), col.end(), [&]() -> float { return Operon::Random::Uniform(rng, -1.0F, +1.0F); });
            }
            data.col(Ncol - 1) = data.col(0) + data.col(1) + data.col(2);
            return Operon::Dataset(data);
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
        MinimumDescriptionLengthEvaluator<DTable, GaussianLikelihood<Operon::Scalar>> ev{&fix.problem, &fix.dtable};
        auto ind = EvaluatorFixture::MakeIndividual(fix.tree);
        auto const result = ev(fix.rng, ind);
        REQUIRE(result.size() == 1);
        CHECK(std::isfinite(result[0]));
        CHECK(result[0] > 0);
    }

    SECTION("Gaussian / fixed sigma: finite positive result") {
        MinimumDescriptionLengthEvaluator<DTable, GaussianLikelihood<Operon::Scalar>> ev{&fix.problem, &fix.dtable};
        ev.SetSigma({0.5F});
        auto ind = EvaluatorFixture::MakeIndividual(fix.tree);
        auto const result = ev(fix.rng, ind);
        REQUIRE(result.size() == 1);
        CHECK(std::isfinite(result[0]));
        CHECK(result[0] > 0);
    }

    SECTION("Poisson: finite result") {
        MinimumDescriptionLengthEvaluator<DTable, PoissonLikelihood<Operon::Scalar>> ev{&fix.problem, &fix.dtable};
        auto ind = EvaluatorFixture::MakeIndividual(fix.tree);
        auto const result = ev(fix.rng, ind);
        REQUIRE(result.size() == 1);
        CHECK(std::isfinite(result[0]));
    }

    SECTION("Gaussian / profiled sigma: SSR=0 does not produce NaN (epsilon clamp)") {
        MinimumDescriptionLengthEvaluator<DTable, GaussianLikelihood<Operon::Scalar>> ev{&fix.problem, &fix.dtable};
        auto ind = EvaluatorFixture::MakeIndividual(fix.perfectTree);
        auto const result = ev(fix.rng, ind);
        REQUIRE(result.size() == 1);
        CHECK(std::isfinite(result[0]));
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
        FractionalBayesFactorEvaluator<DTable, GaussianLikelihood<Operon::Scalar>> ev{&fix.problem, &fix.dtable};
        auto ind = EvaluatorFixture::MakeIndividual(fix.tree);
        auto const result = ev(fix.rng, ind);
        REQUIRE(result.size() == 1);
        CHECK(std::isfinite(result[0]));
        CHECK(result[0] > 0);
    }

    SECTION("Gaussian / fixed sigma: finite positive result") {
        FractionalBayesFactorEvaluator<DTable, GaussianLikelihood<Operon::Scalar>> ev{&fix.problem, &fix.dtable};
        ev.SetSigma({0.5F});
        auto ind = EvaluatorFixture::MakeIndividual(fix.tree);
        auto const result = ev(fix.rng, ind);
        REQUIRE(result.size() == 1);
        CHECK(std::isfinite(result[0]));
        CHECK(result[0] > 0);
    }

    SECTION("Poisson: finite result") {
        FractionalBayesFactorEvaluator<DTable, PoissonLikelihood<Operon::Scalar>> ev{&fix.problem, &fix.dtable};
        auto ind = EvaluatorFixture::MakeIndividual(fix.tree);
        auto const result = ev(fix.rng, ind);
        REQUIRE(result.size() == 1);
        CHECK(std::isfinite(result[0]));
    }

    SECTION("Poisson FBF includes NLL contribution (regression: NLL was silently zero before fix)") {
        // Before the fix, the Poisson NLL was always 0 so FBF = fComplexity + cParameters.
        // After the fix, Poisson and Gaussian NLLs are computed differently, so results must differ.
        FractionalBayesFactorEvaluator<DTable, GaussianLikelihood<Operon::Scalar>> evG{&fix.problem, &fix.dtable};
        FractionalBayesFactorEvaluator<DTable, PoissonLikelihood<Operon::Scalar>>  evP{&fix.problem, &fix.dtable};
        auto ind = EvaluatorFixture::MakeIndividual(fix.tree);
        auto const rG = evG(fix.rng, ind);
        auto const rP = evP(fix.rng, ind);
        CHECK(rG[0] != rP[0]);
    }

    SECTION("Gaussian / profiled sigma: SSR=0 does not produce NaN (epsilon clamp)") {
        FractionalBayesFactorEvaluator<DTable, GaussianLikelihood<Operon::Scalar>> ev{&fix.problem, &fix.dtable};
        auto ind = EvaluatorFixture::MakeIndividual(fix.perfectTree);
        auto const result = ev(fix.rng, ind);
        REQUIRE(result.size() == 1);
        CHECK(std::isfinite(result[0]));
    }
}

} // namespace Operon::Test
