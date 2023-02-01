// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include <doctest/doctest.h>
#include "../operon_test.hpp"

#include "operon/autodiff/autodiff.hpp"
#include "operon/autodiff/forward/forward.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/operators/creator.hpp"
#include "operon/optimizer/optimizer.hpp"
#include "operon/parser/infix.hpp"

namespace dt = doctest;
namespace nb = ankerl::nanobench;

namespace Operon::Test {

TEST_CASE("comparison benchmark" * dt::test_suite("[performance]")) {
    Operon::RandomGenerator rng(0);

    Operon::GenericInterpreter<Operon::Scalar, Operon::Dual> interpreter;
    //Operon::Autodiff::Forward::DerivativeCalculator dc{ interpreter };
    Operon::Autodiff::Reverse::DerivativeCalculator dc{ interpreter };

    constexpr auto nrow{50000};
    constexpr auto ncol{10};
    auto ds = Operon::Test::Util::RandomDataset(rng, nrow, ncol);
    std::string f1{ "10 * sin(3.141592654 * X1 * X2) + 20 * (X3 - 0.5) ^ 2 + 10 * X4 + 5 * X5" };
    std::string f2{"X1 * X2 + X3 * X4 + X5 * X6 + X1 * X7 * X9 + X3 * X6 * X10"};

    Operon::Map<std::string, Operon::Hash> variables;
    for (auto&& v : ds.Variables()) {
        variables.insert({v.Name, v.Hash});
    }
    auto tree = Operon::InfixParser::Parse(f1, variables);
    auto& nodes = tree.Nodes();
    for (auto i = 0; i < std::ssize(nodes); ++i) {
        auto& n = nodes[i];
        n.Optimize = n.IsVariable();

        // replace the raise to the power of 2 with a square primitive (more efficient)
        if (n.IsPow()) {
            auto j = i-1;
            auto k = j - (nodes[j].Length + 1);
            if (nodes[k].IsConstant() && nodes[k].Value == 2) {
                nodes[k].IsEnabled = false;
                nodes[i].Type = NodeType::Square;
                nodes[i].Arity -= 1;
            }
        }
    }
    std::erase_if(nodes, [](auto const& n) { return !n.IsEnabled; });

    auto benchmark = [&](auto const& tree) {
        nb::Bench b;
        b.timeUnit(std::chrono::microseconds(1), "us");
        //b.output(nullptr);

        auto coeff = tree.GetCoefficients();
        for (auto rows = 1000UL; rows <= 50000; rows += 1000) {
            b.run(fmt::format("rows = {}", rows), [&]() {
                Operon::Range range{0, rows};
                Eigen::Array<Operon::Scalar, -1, -1> jacobian = dc(tree, ds, range, coeff);
            });
        }

        b.render(nb::templates::csv(), std::cout);
    };

    fmt::print("F1: {}\n", Operon::InfixFormatter::Format(tree, ds));
    fmt::print("coefficients to optimize: {}\n", std::count_if(nodes.begin(), nodes.end(), [](auto const& node) { return node.Optimize; }));
    benchmark(tree);


    tree = Operon::InfixParser::Parse(f2, variables);
    for (auto& n : tree.Nodes()) {
        n.Optimize = n.IsVariable();
    }
    fmt::print("F2: {}\n", Operon::InfixFormatter::Format(tree, ds));
    fmt::print("coefficients to optimize: {}\n", std::count_if(tree.Nodes().begin(), tree.Nodes().end(), [](auto const& node) { return node.Optimize; }));
    benchmark(tree);

}

TEST_CASE("autodiff performance" * dt::test_suite("[performance]")) {
    constexpr auto nrow{1000};
    constexpr auto ncol{10};

    Operon::RandomGenerator rng(0);
    auto ds = Operon::Test::Util::RandomDataset(rng, nrow, ncol);
    nb::Bench b;
    //b.output(nullptr);
    b.timeUnit(std::chrono::milliseconds(1), "ms");

    Operon::PrimitiveSet pset{ Operon::PrimitiveSet::Arithmetic | Operon::NodeType::Exp | Operon::NodeType::Log | Operon::NodeType::Sin | Operon::NodeType::Cos | Operon::NodeType::Sqrt };
    Operon::BalancedTreeCreator creator(pset, ds.Variables());

    auto constexpr initialSize{20};
    auto benchmark = [&](ankerl::nanobench::Bench& bench, auto const& dc, auto&& f, std::string const& prefix, std::size_t n, std::size_t s) {
        std::vector<Operon::Tree> trees(n);
        double a{0};
        auto z{initialSize}; // max size
        Operon::Range range{0, ds.Rows()};
        std::vector<Operon::Scalar> y(ds.Rows());
        Operon::Span<Operon::Scalar> target{y.data(), y.size()};

        do {
            std::uniform_int_distribution<size_t> dist(1, z);
            std::generate(trees.begin(), trees.end(), [&](){ return creator(rng, dist(rng), 1, 1000); }); // NOLINT
            a = std::transform_reduce(trees.begin(), trees.end(), double{0}, std::plus{}, [](auto const& t) { return t.CoefficientsCount(); }) / static_cast<double>(n);
            auto b = std::transform_reduce(trees.begin(), trees.end(), 0UL, std::plus{}, [](auto const& t) { return t.Length(); });
            bench.batch(trees.size()).run(fmt::format("{};{};{}", prefix, a, static_cast<double>(b)/static_cast<double>(n)), [&]() {
                for (auto const& tree : trees) { f(dc, ds, tree, range); }
            });
            z += 10; // NOLINT
        } while (a < s); // NOLINT
    };

    constexpr auto n{1000};
    constexpr auto m{50};
    Operon::GenericInterpreter<Operon::Scalar, Operon::Dual> interpreter;

    auto residual = [](auto const& dc, auto const& ds, auto const& tree, auto range) {
        auto coeff = tree.GetCoefficients();
        return dc.GetInterpreter().template operator()<Operon::Scalar>(tree, ds, range, coeff);
    };

    auto jacobian = [](auto const& dc, auto const& ds, auto const& tree, auto range) {
        auto coeff = tree.GetCoefficients();
        return dc(tree, ds, range, coeff);
    };

    SUBCASE("residual") {
        Operon::Autodiff::Forward::DerivativeCalculator dc{ interpreter };
        benchmark(b, dc, residual, "residual;", n, m);
    }

    SUBCASE("forward") {
        Operon::Autodiff::Forward::DerivativeCalculator dc{ interpreter };
        benchmark(b, dc, jacobian, "forward;jacobian", n, m);
    }

    SUBCASE("reverse") {
        Operon::Autodiff::Reverse::DerivativeCalculator dc{ interpreter };
        benchmark(b, dc, jacobian, "reverse;jacobian", n, m);
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

    nb::Bench b;
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

    b.render(nb::templates::csv(), std::cout);
}

} // namespace Operon::Test
