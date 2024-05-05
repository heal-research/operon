// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research
//
#include "../operon_test.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/types.hpp"
#include "operon/error_metrics/mean_squared_error.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/optimizer/likelihood/gaussian_likelihood.hpp"
#include "operon/optimizer/likelihood/poisson_likelihood.hpp"
#include "operon/optimizer/optimizer.hpp"
#include "operon/optimizer/solvers/sgd.hpp"
#include "operon/parser/infix.hpp"
#include <doctest/doctest.h>
#include <utility>

namespace Operon::Test {

TEST_CASE("Evaluation correctness")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto range = Range { 0, ds.Rows<std::size_t>() };

    using DTable = DispatchTable<Operon::Scalar>;
    auto const& X = ds.Values(); // NOLINT

    Operon::Map<std::string, Operon::Hash> vars;
    for (auto const& v : ds.GetVariables()) {
        fmt::print("{} : {} {}\n", v.Name, v.Hash, v.Index);
        vars[v.Name] = v.Hash;
    }

    std::vector<size_t> indices(range.Size());
    std::iota(indices.begin(), indices.end(), 0);

    DTable dtable;

    SUBCASE("Basic operations")
    {
        const auto eps = 1e-3;

        auto tree = InfixParser::Parse("X1 + X2 + X3", vars);
        auto coeff = tree.GetCoefficients();
        auto estimatedValues = Interpreter<Operon::Scalar, DTable>(dtable, ds, tree).Evaluate(coeff, range);
        Eigen::Array<Operon::Scalar, -1, 1> res1 = X.col(0) + X.col(1) + X.col(2);

        fmt::print("estimated: {}\n", std::span{estimatedValues.data(), 5UL});
        fmt::print("actual: {}\n", std::span{res1.data(), 5UL});
        CHECK(std::all_of(indices.begin(), indices.end(), [&](auto i) { return std::abs(estimatedValues[i] - res1(i)) < eps; }));

        tree = InfixParser::Parse("X1 - X2 + X3", vars);
        estimatedValues = Interpreter<Operon::Scalar, DTable>(dtable, ds, tree).Evaluate(tree.GetCoefficients(), range);
        auto res2 = X.col(0) - X.col(1) + X.col(2);
        CHECK(std::all_of(indices.begin(), indices.end(), [&](auto i) { return std::abs(estimatedValues[i] - res2(i)) < eps; }));

        fmt::print("tree: {}\n", InfixFormatter::Format(tree, ds));
        estimatedValues = Interpreter<Operon::Scalar, DTable>(dtable, ds, tree).Evaluate(tree.GetCoefficients(), range);
        auto res3 = X.col(0) - X.col(1) + X.col(2);
        CHECK(std::all_of(indices.begin(), indices.end(), [&](auto i) { return std::abs(estimatedValues[i] - res3(i)) < eps; }));

        tree = InfixParser::Parse("log(abs(X1))", vars);
        fmt::print("tree: {}\n", InfixFormatter::Format(tree, ds));
        estimatedValues = Interpreter<Operon::Scalar, DTable>(dtable, ds, tree).Evaluate(tree.GetCoefficients(), range);
        Eigen::Array<Operon::Scalar, -1, 1> res4 = X.col(0).abs().log();
        fmt::print("estimated: {}\n", std::span{estimatedValues.data(), 5UL});
        fmt::print("actual: {}\n", std::span{res4.data(), 5UL});
        CHECK(std::all_of(indices.begin(), indices.end(), [&](auto i) { return std::abs(estimatedValues[i] - res4(i)) < eps; }));

        tree = InfixParser::Parse("log(0.12485691905021667)", vars);
        fmt::print("tree: {}\n", InfixFormatter::Format(tree, ds));
        estimatedValues = Interpreter<Operon::Scalar, DTable>(dtable, ds, tree).Evaluate(tree.GetCoefficients(), range);
        Eigen::Array<Operon::Scalar, 1, 1> res5; res5 << std::log(0.12485691905021667);
        fmt::print("estimated: {}\n", std::span{estimatedValues.data(), 1});
        fmt::print("actual: {}\n", std::span{res5.data(), 1});
        CHECK(std::all_of(indices.begin(), indices.end(), [&](auto i) { return std::abs(estimatedValues[i] - res5(i)) < eps; }));

        Operon::Node node(Operon::NodeType::Fmax);
        node.Arity = 3;
        auto a = Operon::Node::Constant(2);
        auto b = Operon::Node::Constant(3);
        auto c = Operon::Node::Constant(4);
        tree = Operon::Tree({ a, b, c, node });
        fmt::print("tree: {}\n", InfixFormatter::Format(tree, ds));
        estimatedValues = Interpreter<Operon::Scalar, DTable>(dtable, ds, tree).Evaluate(tree.GetCoefficients(), range);
        CHECK(estimatedValues[0] == 4);

        node = Operon::Node(Operon::NodeType::Sub);
        node.Arity = 1;
        tree = Operon::Tree({ Operon::Node::Constant(2), node });
        fmt::print("tree: {}\n", InfixFormatter::Format(tree, ds));
        estimatedValues = Interpreter<Operon::Scalar, DTable>(dtable, ds, tree).Evaluate(tree.GetCoefficients(), range);
        CHECK(estimatedValues[0] == -2);
    }
}

#if defined(OPERON_MATH_VDT) || defined(OPERON_MATH_FAST_V1) || defined(OPERON_MATH_FAST_V2) || defined(OPERON_MATH_FAST_V3)
TEST_CASE("relative accuracy")
{
    auto constexpr N{10'000};
    Operon::RandomGenerator rng{1234};

    using UnaryFunction = std::add_pointer_t<Operon::Scalar(Operon::Scalar)>;
    using BinaryFunction = std::add_pointer_t<Operon::Scalar(Operon::Scalar, Operon::Scalar)>;

    auto testRange = [&]<typename F>(char const* name, F&& f, F&& g, std::pair<Operon::Scalar, Operon::Scalar> range) {
        std::uniform_real_distribution<Operon::Scalar> dist(range.first, range.second);

        vstat::univariate_accumulator<double> acc;

        for (auto i = 0; i < N; ++i) {
            Operon::Scalar x1{};
            Operon::Scalar x2{};
            Operon::Scalar y1{};
            Operon::Scalar y2{};
            if constexpr (std::is_same_v<F, UnaryFunction>) {
                x1 = dist(rng);
                y1 = f(x1);
                y2 = g(x1);
            } else {
                x1 = dist(rng);
                x2 = dist(rng);
                y1 = f(x1, x2);
                y2 = g(x1, x2);
            }
            if (!(std::isfinite(y1) && std::isfinite(y2))) { continue; }
            auto d = std::abs(y1-y2);
            auto r = d / std::abs(y1);
            fmt::print("{},{:.25f},{:.25f},{:.25f},{:.25f},{}\n", name, x1, x2, y1, y2, r);
            acc(r);
        }

        auto const m = vstat::univariate_statistics(acc).mean;
        fmt::print("{},{:4f}%\n", name, 100.0 * m);
    };

    auto div = [](auto a, auto b) { return a / b; };
    auto aq = [](auto a, auto b) { return a / std::sqrt(1 + b*b); };
    auto inv = [](auto x) { return 1/x; };
    auto isqrt = [](auto x) { return 1/std::sqrt(x); };

    auto constexpr nan = std::numeric_limits<Operon::Scalar>::quiet_NaN();

    // auto constexpr lim = Operon::Scalar{std::numeric_limits<Operon::Scalar>::max()};
    auto constexpr lim = Operon::Scalar{10};

    #if defined(OPERON_MATH_FAST_V1) || defined(OPERON_MATH_FAST_V2) || defined(OPERON_MATH_FAST_V3)
    namespace backend = Backend::detail::fast_approx;

    fmt::print("precision level: {}\n", OPERON_MATH_FAST_APPROX_PRECISION);

    SUBCASE("mean accuracy") {
        // reciprocal
        testRange("inv", UnaryFunction{inv}, UnaryFunction{backend::Inv}, {0.001, lim});
        testRange("isqrt", UnaryFunction{isqrt}, UnaryFunction{backend::ISqrt}, {0.001, lim});
        testRange("log", UnaryFunction{std::log}, UnaryFunction{backend::Log}, {0, lim});
        testRange("exp", UnaryFunction{std::exp}, UnaryFunction{backend::Exp}, {0.001, lim});
        testRange("sin", UnaryFunction{std::sin}, UnaryFunction{backend::Sin}, {-lim, +lim});
        testRange("cos", UnaryFunction{std::cos}, UnaryFunction{backend::Cos}, {-lim, +lim});
        testRange("sinh", UnaryFunction{std::sinh}, UnaryFunction{backend::Sinh}, {-lim, +lim});
        testRange("cosh", UnaryFunction{std::cosh}, UnaryFunction{backend::Cosh}, {-lim, +lim});
        testRange("tanh", UnaryFunction{std::tanh}, UnaryFunction{backend::Tanh}, {-lim, +lim});
        testRange("sqrt", UnaryFunction{std::sqrt}, UnaryFunction{backend::Sqrt}, {0, lim});
        testRange("div", BinaryFunction{div}, BinaryFunction{backend::Div}, {-lim, lim});
        testRange("aq",  BinaryFunction{aq}, BinaryFunction{backend::Aq}, {-lim, lim});
        testRange("pow", BinaryFunction{std::pow}, BinaryFunction{backend::Pow}, {0.001, lim});
    }

    SUBCASE("edge cases") {
        fmt::print("log(nan): {} {}\n", backend::Log(nan), std::log(nan));
        fmt::print("exp(nan): {} {}\n", backend::Exp(nan), std::exp(nan));
        fmt::print("sin(nan): {} {}\n", backend::Sin(nan), std::sin(nan));
        fmt::print("cos(nan): {} {}\n", backend::Cos(nan), std::cos(nan));
        fmt::print("tanh(nan): {} {}\n", backend::Tanh(nan), std::tanh(nan));
        fmt::print("sqrt(nan): {} {}\n", backend::Sqrt(nan), std::sqrt(nan));
        fmt::print("div(nan, x): {} {}\n", backend::Div(nan, 2), nan / 2);
        fmt::print("aq(nan, x): {} {}\n", backend::Aq(nan, 2), nan / std::sqrt(5));
    }
    #endif

    #if defined(OPERON_MATH_VDT)
    namespace backend = Backend::detail::vdt;
    SUBCASE("[-1, +1]") {
        testRange("inv", UnaryFunction{inv}, UnaryFunction{backend::Inv}, {0.001, lim});
        testRange("isqrt", UnaryFunction{isqrt}, UnaryFunction{backend::ISqrt}, {0.001, lim});
        testRange("log", UnaryFunction{std::log}, UnaryFunction{backend::Log}, {0, lim});
        testRange("exp", UnaryFunction{std::exp}, UnaryFunction{backend::Exp}, {-lim, lim});
        testRange("sin", UnaryFunction{std::sin}, UnaryFunction{backend::Sin}, {-lim, +lim});
        testRange("cos", UnaryFunction{std::cos}, UnaryFunction{backend::Cos}, {-lim, +lim});
        testRange("sinh", UnaryFunction{std::sinh}, UnaryFunction{backend::Sinh}, {-lim, +lim});
        testRange("cosh", UnaryFunction{std::cosh}, UnaryFunction{backend::Cosh}, {-lim, +lim});
        testRange("tanh", UnaryFunction{std::tanh}, UnaryFunction{backend::Tanh}, {-lim, +lim});
        testRange("sqrt", UnaryFunction{std::sqrt}, UnaryFunction{backend::Sqrt}, {0, lim});
        testRange("div", BinaryFunction{div}, BinaryFunction{backend::Div}, {-lim, lim});
        testRange("aq",  BinaryFunction{aq}, BinaryFunction{backend::Aq}, {-lim, lim});
        testRange("pow", BinaryFunction{std::pow}, BinaryFunction{backend::Pow}, {-lim, lim});
    }

    SUBCASE("edge cases") {
        fmt::print("log(nan): {} {}\n", backend::Log(nan), std::log(nan));
        fmt::print("exp(nan): {} {}\n", backend::Exp(nan), std::exp(nan));
        fmt::print("sin(nan): {} {}\n", backend::Sin(nan), std::sin(nan));
        fmt::print("cos(nan): {} {}\n", backend::Cos(nan), std::cos(nan));
        fmt::print("tanh(nan): {} {}\n", backend::Tanh(nan), std::tanh(nan));
        fmt::print("sqrt(nan): {} {}\n", backend::Sqrt(nan), std::sqrt(nan));
        fmt::print("div(nan, x): {} {}\n", backend::Div(nan, 2), nan / 2);
        fmt::print("aq(nan, x): {} {}\n", backend::Aq(nan, 2), nan / std::sqrt(5));
    }
    #endif
}
#endif

TEST_CASE("Batch evaluation")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto range = Range { 0, ds.Rows<std::size_t>() };

    Operon::Problem problem{ds, range, range};
    Operon::PrimitiveSet pset{PrimitiveSet::Arithmetic};
    Operon::BalancedTreeCreator creator{pset, ds.VariableHashes()};

    Operon::RandomGenerator rng{0};
    auto constexpr n{10};

    std::vector<Operon::Tree> trees;
    std::vector<Operon::Scalar> result(range.Size() * n);    for (auto i = 0; i < n; ++i) {
        trees.push_back(creator(rng, 20, 10, 20));
    }

    Operon::EvaluateTrees(trees, ds, range, {result.data(), result.size()});
    Operon::EvaluateTrees(trees, ds, range);
}

TEST_CASE("parameter optimization")
{
    Operon::RandomGenerator rng{0};
    constexpr auto nrow{500};
    constexpr auto ncol{7};
    auto range = Range { 0, nrow };

    Eigen::Array<Operon::Scalar, -1, -1> data(nrow, ncol);
    for (auto i = 0; i < ncol; ++i) {
        auto col = data.col(i);
        std::generate(col.begin(), col.end(), [&](){ return Operon::Random::Uniform(rng, -1.0F, +1.0F); });
    }

    // input variables
    auto x1 = data.col(0);
    auto x2 = data.col(1);
    auto x3 = data.col(2);
    auto x4 = data.col(3);
    auto x5 = data.col(4);
    auto x6 = data.col(5);

    // target variable
    data.col(ncol-1) = x1 * x2 + x3 * x4 + x5  * x6;

    Operon::Dataset ds(data);
    Operon::Map<std::string, Operon::Hash> vars;
    for (auto v : ds.GetVariables()) {
        // fmt::print("{} : {}\n", v.Name, v.Hash);
        vars[v.Name] = v.Hash;
    }

    Eigen::Array<Operon::Scalar, -1, 1> res = x1 * x2 + x3 * x4 + x5 * x6;
    Operon::Span<Operon::Scalar> target(res.data(), res.size());
    auto tree = InfixParser::Parse("X1 * X2 + X3 * X4 + X5 * X6", vars);
    for (auto& node : tree.Nodes()) {
        if (node.IsVariable()) {
            node.Value = static_cast<Operon::Scalar>(0.01);
        } // NOLINT
    }

    using DTable = DispatchTable<Operon::Scalar>;
    DTable dtable;

    Operon::Interpreter<Operon::Scalar, DTable> interpreter { dtable, ds, tree };

    Operon::Problem problem{ds, range, range};

    auto constexpr batchSize { 32 };

#if defined(HAVE_CERES)
    SUBCASE("ceres autodiff")
    {
        LevenbergMarquardtOptimizer<DerivativeCalculator, OptimizerType::Ceres> optimizer(dc, tree, ds);
        OptimizerSummary summary {};
        auto coeff = optimizer.Optimize(target, range, iterations, summary);
        fmt::print("iterations: {}, initial cost: {}, final cost: {}\n", summary.Iterations, summary.InitialCost, summary.FinalCost);
        tree.SetCoefficients(coeff);
        fmt::print("final tree: {}\n", InfixFormatter::Format(tree, ds));
    }
#endif
    auto const dim { tree.CoefficientsCount() };

    std::vector<std::unique_ptr<UpdateRule::LearningRateUpdateRule const>> rules;
    rules.emplace_back(new UpdateRule::Constant<Operon::Scalar>(dim, 1e-3)); // NOLINT
    rules.emplace_back(new UpdateRule::Momentum<Operon::Scalar>(dim));
    rules.emplace_back(new UpdateRule::RmsProp<Operon::Scalar>(dim));
    rules.emplace_back(new UpdateRule::AdaDelta<Operon::Scalar>(dim));
    rules.emplace_back(new UpdateRule::AdaMax<Operon::Scalar>(dim));
    rules.emplace_back(new UpdateRule::Adam<Operon::Scalar>(dim));
    rules.emplace_back(new UpdateRule::YamAdam<Operon::Scalar>(dim));
    rules.emplace_back(new UpdateRule::AmsGrad<Operon::Scalar>(dim));
    rules.emplace_back(new UpdateRule::Yogi<Operon::Scalar>(dim));

    auto testOptimizer = [&](OptimizerBase& optimizer, std::string const& name) {
        fmt::print(fmt::fg(fmt::color::orange), "=== {} Solver ===\n", name);
        auto summary = optimizer.Optimize(rng, tree);
        fmt::print("batch size: {}\n", batchSize);
        fmt::print("expression: {}\n", InfixFormatter::Format(tree, ds));
        fmt::print("iterations: {}, initial cost: {}, final cost: {}\n", summary.Iterations, summary.InitialCost, summary.FinalCost);
        fmt::print("final parameters: {}\n\n", summary.FinalParameters);
    };

    SUBCASE("tiny")
    {
        LevenbergMarquardtOptimizer<DTable, OptimizerType::Tiny> optimizer{dtable, problem};
        testOptimizer(optimizer, "tiny solver");
    }

    SUBCASE("eigen")
    {
        LevenbergMarquardtOptimizer<DTable, OptimizerType::Eigen> optimizer { dtable, problem };
        testOptimizer(optimizer, "eigen solver");
    }

    SUBCASE("ceres")
    {
        LevenbergMarquardtOptimizer<DTable, OptimizerType::Ceres> optimizer { dtable, problem };
        testOptimizer(optimizer, "ceres solver");
    }

    SUBCASE("lbfgs / gaussian")
    {
        LBFGSOptimizer<DTable, GaussianLikelihood<Operon::Scalar>> optimizer { dtable, problem };
        testOptimizer(optimizer, "l-bfgs / gaussian");
    }

    SUBCASE("lbfgs / poisson")
    {
        LBFGSOptimizer<DTable, PoissonLikelihood<Operon::Scalar>> optimizer { dtable, problem };
        testOptimizer(optimizer, "l-bfgs / poisson");
    }

    SUBCASE("sgd / gaussian")
    {
        for (auto const& rule : rules) {
            SGDOptimizer<DTable, GaussianLikelihood<Operon::Scalar>> optimizer { dtable, problem, *rule };
            testOptimizer(optimizer, fmt::format("sgd / gaussian / {}", rule->Name()));
        }
    }

    SUBCASE("sgd / poisson")
    {
        for (auto const& rule : rules) {
            SGDOptimizer<DTable, PoissonLikelihood<Operon::Scalar>> optimizer { dtable, problem, *rule };
            testOptimizer(optimizer, fmt::format("sgd / poisson / {}", rule->Name()));
        }
    }
}
} // namespace Operon::Test
