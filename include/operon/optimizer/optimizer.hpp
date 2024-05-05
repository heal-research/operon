// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_OPTIMIZER_HPP
#define OPERON_OPTIMIZER_HPP

#include "operon/error_metrics/mean_squared_error.hpp"
#include "operon/error_metrics/sum_of_squared_errors.hpp"
#include "operon/interpreter/dispatch_table.hpp"
#include <functional>
#if defined(HAVE_CERES)
#include <ceres/tiny_solver.h>
#else
#include "operon/ceres/tiny_solver.h"
#endif

#include <lbfgs/solver.hpp>
#include <unsupported/Eigen/LevenbergMarquardt>

#include "dynamic_cost_function.hpp"
#include "likelihood/gaussian_likelihood.hpp"
#include "likelihood/poisson_likelihood.hpp"
#include "lm_cost_function.hpp"
#include "operon/core/comparison.hpp"
#include "operon/core/problem.hpp"
#include "solvers/sgd.hpp"

namespace Operon {

enum class OptimizerType : int { Tiny, Eigen, Ceres };

struct OptimizerSummary {
    std::vector<Operon::Scalar> InitialParameters;
    std::vector<Operon::Scalar> FinalParameters;
    Operon::Scalar InitialCost{};
    Operon::Scalar FinalCost{};
    int Iterations{};
    int FunctionEvaluations{};
    int JacobianEvaluations{};
    bool Success{};
};

class OptimizerBase {
std::reference_wrapper<Problem const> problem_;
// batch size for loss functions (default = 0 -> use entire data range)
mutable std::size_t batchSize_{0};
mutable std::size_t iterations_{100}; // NOLINT

public:
    explicit OptimizerBase(Problem const& problem)
        : problem_ { problem }
    {
    }

    OptimizerBase(const OptimizerBase&) = default;
    OptimizerBase(OptimizerBase&&) = delete;
    auto operator=(const OptimizerBase&) -> OptimizerBase& = default;
    auto operator=(OptimizerBase&&) -> OptimizerBase& = delete;

    virtual ~OptimizerBase() = default;

    [[nodiscard]] auto GetProblem() const -> Problem const& { return problem_.get(); }
    [[nodiscard]] auto BatchSize() const -> std::size_t { return batchSize_; }
    [[nodiscard]] auto Iterations() const -> std::size_t { return iterations_; }

    auto SetBatchSize(std::size_t batchSize) const { batchSize_ = batchSize; }
    auto SetIterations(std::size_t iterations) const { iterations_ = iterations; }

    [[nodiscard]] virtual auto Optimize(Operon::RandomGenerator& rng, Tree const& tree) const -> OptimizerSummary = 0;
    [[nodiscard]] virtual auto ComputeLikelihood(Operon::Span<Operon::Scalar const> x, Operon::Span<Operon::Scalar const> y, Operon::Span<Operon::Scalar const> w) const -> Operon::Scalar = 0;
    [[nodiscard]] virtual auto ComputeFisherMatrix(Operon::Span<Operon::Scalar const> pred, Operon::Span<Operon::Scalar const> jac, Operon::Span<Operon::Scalar const> sigma) const -> Eigen::Matrix<Operon::Scalar, -1, -1> = 0;
};

namespace detail {
    inline auto CheckSuccess(double initialCost, double finalCost) {
        constexpr auto CHECK_NAN{true};
        return Operon::Less<CHECK_NAN>{}(finalCost, initialCost);
    }
} // namespace detail

template <typename DTable, OptimizerType = OptimizerType::Tiny>
struct LevenbergMarquardtOptimizer : public OptimizerBase {
    explicit LevenbergMarquardtOptimizer(DTable const& dtable, Problem const& problem)
        : OptimizerBase{problem}, dtable_{dtable}
    {
    }

    [[nodiscard]] auto Optimize(Operon::RandomGenerator& /*unused*/, Operon::Tree const& tree) const -> OptimizerSummary final
    {
        auto const& dtable = this->GetDispatchTable();
        auto const& problem = this->GetProblem();
        auto const& dataset = problem.GetDataset();
        auto range  = problem.TrainingRange();
        auto target = problem.TargetValues(range);
        auto iterations = this->Iterations();

        Operon::Interpreter<Operon::Scalar, DTable> interpreter{dtable, dataset, tree};
        Operon::LMCostFunction cf{interpreter, target, range};
        ceres::TinySolver<decltype(cf)> solver;
        solver.options.max_num_iterations = static_cast<int>(iterations);

        auto x0 = tree.GetCoefficients();
        OptimizerSummary summary;
        summary.InitialParameters = x0;
        auto m0 = Eigen::Map<Eigen::Matrix<Operon::Scalar, Eigen::Dynamic, 1>>(x0.data(), x0.size());
        if (!x0.empty()) {
            typename decltype(solver)::Parameters p = m0.cast<typename decltype(cf)::Scalar>();
            solver.Solve(cf, &p);
            m0 = p.template cast<Operon::Scalar>();
        }
        summary.FinalParameters = x0;
        summary.InitialCost = solver.summary.initial_cost;
        summary.FinalCost = solver.summary.final_cost;
        summary.Iterations = solver.summary.iterations;
        summary.FunctionEvaluations = solver.summary.iterations;
        summary.Success = detail::CheckSuccess(summary.InitialCost, summary.FinalCost);
        return summary;
    }

    auto GetDispatchTable() const -> DTable const& { return dtable_.get(); }

    [[nodiscard]] auto ComputeLikelihood(Operon::Span<Operon::Scalar const> x, Operon::Span<Operon::Scalar const> y, Operon::Span<Operon::Scalar const> w) const -> Operon::Scalar final
    {
        return GaussianLikelihood<Operon::Scalar>::ComputeLikelihood(x, y, w);
    }

    [[nodiscard]] auto ComputeFisherMatrix(Operon::Span<Operon::Scalar const> pred, Operon::Span<Operon::Scalar const> jac, Operon::Span<Operon::Scalar const> sigma) const -> Eigen::Matrix<Operon::Scalar, -1, -1> final {
        return GaussianLikelihood<Operon::Scalar>::ComputeFisherMatrix(pred, jac, sigma);
    }

    private:
    std::reference_wrapper<DTable const> dtable_;
};

template <typename DTable>
struct LevenbergMarquardtOptimizer<DTable, OptimizerType::Eigen> final : public OptimizerBase {
    explicit LevenbergMarquardtOptimizer(DTable const& dtable, Problem const& problem)
        : OptimizerBase{problem}, dtable_{dtable}
    {
    }

    [[nodiscard]] auto Optimize(Operon::RandomGenerator& /*unused*/, Operon::Tree const& tree) const -> OptimizerSummary final
    {
        auto const& dtable = this->GetDispatchTable();
        auto const& problem = this->GetProblem();
        auto const& dataset = problem.GetDataset();
        auto range  = problem.TrainingRange();
        auto target = problem.TargetValues(range);
        auto iterations = this->Iterations();

        Operon::Interpreter<Operon::Scalar, DTable> interpreter{dtable, dataset, tree};
        Operon::LMCostFunction<Operon::Scalar> cf{interpreter, target, range};
        Eigen::LevenbergMarquardt<decltype(cf)> lm(cf);
        lm.setMaxfev(static_cast<int>(iterations+1));

        auto x0 = tree.GetCoefficients();
        OptimizerSummary summary;
        summary.InitialParameters = x0;
        if (!x0.empty()) {
            Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1>> m0(x0.data(), std::ssize(x0));
            Eigen::Matrix<Operon::Scalar, -1, 1> m = m0;

            // do the minimization loop manually because we want to extract the initial cost
            Eigen::LevenbergMarquardtSpace::Status status = lm.minimizeInit(m);
            summary.InitialCost = summary.FinalCost = lm.fnorm() * lm.fnorm() * 0.5; // get the initial cost after calling minimizeInit()
            if (status != Eigen::LevenbergMarquardtSpace::ImproperInputParameters) {
                do {
                    status = lm.minimizeOneStep(m);
                } while (status == Eigen::LevenbergMarquardtSpace::Running);
            }
            m0 = m;
        }
        summary.FinalParameters = x0;
        summary.FinalCost = lm.fnorm() * lm.fnorm();
        summary.Iterations = static_cast<int>(lm.iterations());
        summary.FunctionEvaluations = static_cast<int>(lm.nfev());
        summary.JacobianEvaluations = static_cast<int>(lm.njev());
        summary.Success = detail::CheckSuccess(summary.InitialCost, summary.FinalCost);
        return summary;
    }

    auto GetDispatchTable() const -> DTable const& { return dtable_.get(); }

    [[nodiscard]] auto ComputeLikelihood(Operon::Span<Operon::Scalar const> x, Operon::Span<Operon::Scalar const> y, Operon::Span<Operon::Scalar const> w) const -> Operon::Scalar final
    {
        return GaussianLikelihood<Operon::Scalar>::ComputeLikelihood(x, y, w);
    }

    [[nodiscard]] auto ComputeFisherMatrix(Operon::Span<Operon::Scalar const> pred, Operon::Span<Operon::Scalar const> jac, Operon::Span<Operon::Scalar const> sigma) const -> Eigen::Matrix<Operon::Scalar, -1, -1> final {
        return GaussianLikelihood<Operon::Scalar>::ComputeFisherMatrix(pred, jac, sigma);
    }

    private:
    std::reference_wrapper<DTable const> dtable_;
};

#if defined(HAVE_CERES)
template <typename T = Operon::Scalar>
struct NonlinearLeastSquaresOptimizer<T, OptimizerType::Ceres> : public OptimizerBase<T> {
    explicit NonlinearLeastSquaresOptimizer(InterpreterBase<T>& interpreter)
        : OptimizerBase<T>{interpreter}
    {
    }

    auto Optimize(Operon::Span<Operon::Scalar const> target, Range range, size_t iterations, OptimizerSummary& summary) -> std::vector<Operon::Scalar> final
    {
        auto const& tree = this->GetTree();
        auto const& ds = this->GetDataset();
        auto const& dt = this->GetDispatchTable();

        auto x0 = tree.GetCoefficients();

        Operon::CostFunction<DTable, Eigen::RowMajor> cf(tree, ds, target, range, dt);
        auto costFunction = new Operon::DynamicCostFunction(cf); // NOLINT

        ceres::Solver::Summary s;
        if (!x0.empty()) {
            Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1>> m0(x0.data(), std::ssize(x0));
            auto sz = static_cast<Eigen::Index>(x0.size());
            Eigen::VectorXd params = Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1>>(x0.data(), sz).template cast<double>();
            ceres::Problem problem;
            problem.AddResidualBlock(costFunction, nullptr, params.data());
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.logging_type = ceres::LoggingType::SILENT;
            options.max_num_iterations = static_cast<int>(iterations - 1); // workaround since for some reason ceres sometimes does 1 more iteration
            options.minimizer_progress_to_stdout = false;
            options.num_threads = 1;
            options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
            options.use_inner_iterations = false;
            Solve(options, &problem, &s);
            m0 = params.cast<Operon::Scalar>();
        }
        summary.InitialCost = s.initial_cost;
        summary.FinalCost = s.final_cost;
        summary.Iterations = static_cast<int>(s.iterations.size());
        summary.FunctionEvaluations = s.num_residual_evaluations;
        summary.JacobianEvaluations = s.num_jacobian_evaluations;
        summary.Success = detail::CheckSuccess(summary.InitialCost, summary.FinalCost);
        return x0;
    }
};
#endif

template<typename DTable, Concepts::Likelihood LossFunction = GaussianLikelihood<Operon::Scalar>>
struct LBFGSOptimizer final : public OptimizerBase {
    LBFGSOptimizer(DTable const& dtable, Problem const& problem)
        : OptimizerBase{problem}, dtable_{dtable}
    {
    }

    [[nodiscard]] auto Optimize(Operon::RandomGenerator& rng, Operon::Tree const& tree) const -> OptimizerSummary final
    {
        auto const& dtable = this->GetDispatchTable();
        auto const& problem = this->GetProblem();
        auto const& dataset = problem.GetDataset();
        auto range  = problem.TrainingRange();
        auto target = problem.TargetValues(range);
        auto iterations = this->Iterations();
        auto batchSize = this->BatchSize();
        if (batchSize == 0) { batchSize = range.Size(); }

        Operon::Interpreter<Operon::Scalar, DTable> interpreter{dtable, dataset, tree};
        LossFunction loss{rng, interpreter, target, range, batchSize};

        auto cost = [&](auto const& coeff) {
            auto pred = interpreter.Evaluate(coeff, range);
            return 0.5 * Operon::SumOfSquaredErrors(pred.begin(), pred.end(), target.begin());
        };

        auto coeff = tree.GetCoefficients();
        Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1> const> x0(coeff.data(), std::ssize(coeff));

        lbfgs::solver solver{loss};
        solver.max_iterations = iterations;
        solver.max_line_search_iterations = iterations;
        auto const f0 = cost(coeff);
        OptimizerSummary summary;
        summary.InitialParameters = coeff;
        summary.InitialCost = f0;

        if (auto res = solver.optimize(x0)) {
            auto xf = res.value();
            std::copy(xf.begin(), xf.end(), coeff.begin());
        }

        summary.FinalParameters = coeff;
        auto const f1 = cost(coeff);
        summary.FinalCost = f1;
        summary.Success = detail::CheckSuccess(f0, f1);
        auto const funEvals = loss.FunctionEvaluations();
        auto const jacEvals = loss.JacobianEvaluations();
        auto const rangeSize = range.Size();
        summary.FunctionEvaluations = static_cast<std::size_t>(static_cast<double>(funEvals + jacEvals) * batchSize / rangeSize);
        summary.JacobianEvaluations = summary.FunctionEvaluations;
        return summary;
    }

    auto GetDispatchTable() const -> DTable const& { return dtable_.get(); }

    [[nodiscard]] virtual auto ComputeLikelihood(Operon::Span<Operon::Scalar const> x, Operon::Span<Operon::Scalar const> y, Operon::Span<Operon::Scalar const> w) const -> Operon::Scalar
    {
        return LossFunction::ComputeLikelihood(x, y, w);
    }

    [[nodiscard]] virtual auto ComputeFisherMatrix(Operon::Span<Operon::Scalar const> pred, Operon::Span<Operon::Scalar const> jac, Operon::Span<Operon::Scalar const> sigma) const -> Eigen::Matrix<Operon::Scalar, -1, -1> final {
        return LossFunction::ComputeFisherMatrix(pred, jac, sigma);
    }

    private:
    std::reference_wrapper<DTable const> dtable_;
};

template<typename DTable, Concepts::Likelihood LossFunction = GaussianLikelihood<Operon::Scalar>>
struct SGDOptimizer final : public OptimizerBase {
    SGDOptimizer(DTable const& dtable, Problem const& problem)
        : OptimizerBase{problem}
        , dtable_{dtable}
        , update_{std::make_unique<UpdateRule::Constant<Operon::Scalar>>(Operon::Scalar{0.01})}
    { }

    SGDOptimizer(DTable const& dtable, Problem const& problem, UpdateRule::LearningRateUpdateRule const& update)
        : OptimizerBase{problem}
        , dtable_{dtable}
        , update_{update.Clone(0)}
    { }

    auto GetDispatchTable() const -> DTable const& { return dtable_.get(); }

    [[nodiscard]] auto Optimize(Operon::RandomGenerator& rng, Operon::Tree const& tree) const -> OptimizerSummary final
    {
        auto const& dtable = this->GetDispatchTable();
        auto const& problem = this->GetProblem();
        auto const& dataset = problem.GetDataset();
        auto range  = problem.TrainingRange();
        auto target = problem.TargetValues(range);
        auto iterations = this->Iterations();
        auto batchSize = this->BatchSize();
        if (batchSize == 0) { batchSize = range.Size(); }

        Operon::Interpreter<Operon::Scalar, DTable> interpreter{dtable, dataset, tree};
        LossFunction loss{rng, interpreter, target, range, batchSize};

        auto cost = [&](auto const& coeff) {
            auto pred = interpreter.Evaluate(coeff, range);
            return 0.5 * Operon::SumOfSquaredErrors(pred.begin(), pred.end(), target.begin());
        };

        auto coeff = tree.GetCoefficients();
        auto const f0 = cost(coeff);
        OptimizerSummary summary;
        summary.InitialParameters = coeff;
        summary.InitialCost = f0;
        auto rule = update_->Clone(coeff.size());
        SGDSolver<LossFunction> solver(loss, *rule);

        Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> x0(coeff.data(), std::ssize(coeff));
        auto x = solver.Optimize(x0, iterations);
        std::copy(x.begin(), x.end(), coeff.begin());
        auto const f1 = cost(coeff);

        summary.FinalParameters = coeff;
        summary.FinalCost = f1;
        summary.Success = detail::CheckSuccess(f0, f1);
        summary.Iterations = solver.Epochs();
        auto const funEvals = loss.FunctionEvaluations();
        auto const jacEvals = loss.JacobianEvaluations();
        auto const rangeSize = range.Size();
        summary.FunctionEvaluations = static_cast<std::size_t>(static_cast<double>(funEvals + jacEvals) * batchSize / rangeSize);
        summary.JacobianEvaluations = summary.FunctionEvaluations;
        return summary;
    }

    [[nodiscard]] virtual auto ComputeLikelihood(Operon::Span<Operon::Scalar const> x, Operon::Span<Operon::Scalar const> y, Operon::Span<Operon::Scalar const> w) const -> Operon::Scalar
    {
        return LossFunction::ComputeLikelihood(x, y, w);
    }

    [[nodiscard]] virtual auto ComputeFisherMatrix(Operon::Span<Operon::Scalar const> pred, Operon::Span<Operon::Scalar const> jac, Operon::Span<Operon::Scalar const> sigma) const -> Eigen::Matrix<Operon::Scalar, -1, -1> final {
        return LossFunction::ComputeFisherMatrix(pred, jac, sigma);
    }

    auto SetUpdateRule(std::unique_ptr<UpdateRule::LearningRateUpdateRule const> update) {
        update_ = std::move(update);
    }

    auto UpdateRule() const { return update_.get(); }

    private:
    std::reference_wrapper<DTable const> dtable_;
    std::unique_ptr<UpdateRule::LearningRateUpdateRule const> update_{nullptr};
};
} // namespace Operon
#endif
