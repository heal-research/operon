// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include <doctest/doctest.h>
#include "../operon_test.hpp"

#include "operon/autodiff/autodiff.hpp"
#include "operon/autodiff/forward/forward.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/operators/creator.hpp"
#include "operon/optimizer/optimizer.hpp"
#include "operon/parser/infix.hpp"

namespace dt = doctest;

namespace Operon::Test {

TEST_CASE("autodiff performance" * dt::test_suite("[performance]")) {
    constexpr auto nrow{1000};
    constexpr auto ncol{10};

    Operon::RandomGenerator rng(0);
    auto ds = Operon::Test::Util::RandomDataset(rng, nrow, ncol);
    //auto ds = Dataset("./data/Friedman-I.csv", /*hasHeader=*/true);
    ankerl::nanobench::Bench b;
    //b.output(nullptr);

    Operon::PrimitiveSet pset;
    Operon::PrimitiveSetConfig psetcfg = Operon::PrimitiveSet::Arithmetic;
    //Operon::PrimitiveSetConfig psetcfg = Operon::PrimitiveSet::Arithmetic | Operon::NodeType::Exp | Operon::NodeType::Log | Operon::NodeType::Sin | Operon::NodeType::Cos | Operon::NodeType::Sqrt;
    pset.SetConfig(psetcfg);
    Operon::BalancedTreeCreator creator(pset, ds.Variables());

    auto benchmark = [&]<typename DerivativeCalculator>(ankerl::nanobench::Bench& bench, DerivativeCalculator& dc, std::string const& prefix, std::size_t n, std::size_t s) {
        std::vector<Operon::Tree> trees(n);
        double a{0};
        auto z{20}; // max size
        Operon::Range range{0, ds.Rows()};
        std::vector<Operon::Scalar> y(ds.Rows());
        Operon::Span<Operon::Scalar> target{y.data(), y.size()};

        do {
            std::uniform_int_distribution<size_t> dist(1, z);
            std::generate(trees.begin(), trees.end(), [&](){ return creator(rng, dist(rng), 1, 1000); }); // NOLINT
            a = std::transform_reduce(trees.begin(), trees.end(), double{0}, std::plus{}, [](auto const& t) { return t.CoefficientsCount(); }) / static_cast<double>(n);
            auto b = std::transform_reduce(trees.begin(), trees.end(), 0UL, std::plus{}, [](auto const& t) { return t.Length(); });
            bench.batch(range.Size() * b).run(fmt::format("{};{};{}", prefix, a, static_cast<double>(b)/static_cast<double>(n)), [&]() {
                std::size_t sz{0};
                Operon::Scalar r2{0};
                for (auto const& tree : trees) {
                    auto coeff = tree.GetCoefficients();
                    //std::vector<Operon::Scalar> residual(range.Size());
                    //dc.GetInterpreter().template operator()<Operon::Scalar>(tree, ds, range, residual, coeff);
                    //sz += static_cast<std::size_t>(y.front());
                    //r2 += Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1>>(y.data(), std::ssize(y)).sum();
                    Eigen::Matrix<Operon::Scalar, -1, -1> jac = dc(tree, ds, coeff, range);
                    //sz += jac.size();
                }
                return sz;
            });
            z += 10;
        } while (a < s);
    };

    constexpr auto n{1000};
    constexpr auto m{10};
    Operon::GenericInterpreter<Operon::Scalar, Operon::Dual> interpreter;

    SUBCASE("forward") {
        Operon::Autodiff::Forward::DerivativeCalculator dc{ interpreter };
        benchmark(b, dc, "forward", n, m);
    }

    SUBCASE("reverse") {
        Operon::Autodiff::Reverse::DerivativeCalculator dc{ interpreter };
        benchmark(b, dc, "reverse", n, m);
    }

    b.render(ankerl::nanobench::templates::csv(), std::cout);
}

TEST_CASE("optimizer performance" * dt::test_suite("[performance]")) {
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto range = Range { 0, ds.Rows() };

    Interpreter interpreter;
    auto const& X = ds.Values(); // NOLINT

    Operon::Map<std::string, Operon::Hash> vars;
    for (auto const& v : ds.Variables()) {
        fmt::print("{} : {} {}\n", v.Name, v.Hash, v.Index);
        vars[v.Name] = v.Hash;
    }

    std::vector<size_t> indices(range.Size());
    std::iota(indices.begin(), indices.end(), 0);

    Operon::RandomGenerator rng(0);

    auto benchmark = [&]<typename DerivativeCalculator>(ankerl::nanobench::Bench& bench, DerivativeCalculator& dc, Operon::BalancedTreeCreator const& creator, std::string const& prefix, std::size_t n, std::size_t s, std::size_t r) {
        std::vector<Operon::Tree> trees(n);
        double a{0};
        auto z{20}; // max size
        Operon::Range range{0, ds.Rows()};
        std::vector<Operon::Scalar> y(ds.Rows());
        Operon::Span<Operon::Scalar> target{y.data(), y.size()};

        do {
            std::uniform_int_distribution<size_t> dist(1, z);
            std::generate(trees.begin(), trees.end(), [&](){ return creator(rng, dist(rng), 1, 1000); }); // NOLINT
            a = std::transform_reduce(trees.begin(), trees.end(), double{0}, std::plus{}, [](auto const& t) { return t.CoefficientsCount(); }) / static_cast<double>(n);
            auto b = std::transform_reduce(trees.begin(), trees.end(), 0UL, std::plus{}, [](auto const& t) { return t.Length(); });

            bench.batch(range.Size() * b).run(fmt::format("{};{};{};{}", prefix, a, static_cast<double>(b)/static_cast<double>(n), r), [&]() {
                std::size_t sz{0};
                for (auto const& tree : trees) {
                    Operon::NonlinearLeastSquaresOptimizer<DerivativeCalculator, OptimizerType::Eigen> optimizer(dc, tree, ds);
                    Operon::OptimizerSummary summary{};
                    auto x = optimizer.Optimize(target, range, r, summary);
                    sz += x.size();
                }
                return sz;
            });
            z += 10;
        } while (a < s);
    };

    constexpr auto n{1000};
    constexpr auto m{50};
    constexpr auto r{20};

    ankerl::nanobench::Bench b;
    Operon::PrimitiveSet pset;
    Operon::PrimitiveSetConfig psetcfg = Operon::PrimitiveSet::Arithmetic | Operon::NodeType::Exp | Operon::NodeType::Log | Operon::NodeType::Sin | Operon::NodeType::Cos;
    pset.SetConfig(psetcfg);
        Operon::BalancedTreeCreator creator(pset, ds.Variables());

    SUBCASE("forward") {
        Operon::Autodiff::Forward::DerivativeCalculator calcForward{ interpreter };
        benchmark(b, calcForward, creator, "forward", n, m, r);
    }

    SUBCASE("reverse") {
        Operon::Autodiff::Reverse::DerivativeCalculator calcReverse{ interpreter };
        benchmark(b, calcReverse, creator, "reverse", n, m, r);
    }

    b.render(ankerl::nanobench::templates::csv(), std::cout);
}

} // namespace Operon::Test
