// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include <doctest/doctest.h>
#include "../operon_test.hpp"

#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/interpreter/derivative_calculator.hpp"
#include "operon/interpreter/dispatch_table.hpp"
#include "operon/operators/creator.hpp"
#include "operon/optimizer/optimizer.hpp"
#include "operon/parser/infix.hpp"

namespace dt = doctest;

namespace Operon::Test {

namespace detail {
    using T = Operon::detail::Array<Operon::Scalar>;

} // namespace detail

TEST_CASE("autodiff performance" * dt::test_suite("[performance]")) {
    constexpr auto nrow{10000};
    constexpr auto ncol{10};

    Operon::RandomGenerator rng(0);
    auto ds = Operon::Test::Util::RandomDataset(rng, nrow, ncol);
    ankerl::nanobench::Bench b;
    //b.output(nullptr);

    Operon::PrimitiveSet pset;
    Operon::PrimitiveSetConfig psetcfg = Operon::PrimitiveSet::Arithmetic | Operon::NodeType::Exp | Operon::NodeType::Cos | Operon::NodeType::Exp | Operon::NodeType::Log;
    pset.SetConfig(psetcfg);
    Operon::BalancedTreeCreator creator(pset, ds.Variables());

    auto benchForward = [&](ankerl::nanobench::Bench& bench, std::size_t n, std::size_t s) {
        std::vector<Operon::Tree> trees(n);
        double a{0};
        auto z{20}; // max size
                    //
        Operon::Range range{0, ds.Rows()};
        std::vector<Operon::Scalar> y(ds.Rows());
        Operon::Span<Operon::Scalar> target{y.data(), y.size()};
        Operon::Interpreter interpreter;

        do {
            std::uniform_int_distribution<size_t> dist(1, z);
            std::generate(trees.begin(), trees.end(), [&](){ return creator(rng, dist(rng), 1, 1000); }); // NOLINT
            a = std::transform_reduce(trees.begin(), trees.end(), double{0}, std::plus{}, [](auto const& t) { return t.CoefficientsCount(); }) / static_cast<double>(n);
            auto b = std::transform_reduce(trees.begin(), trees.end(), 0UL, std::plus{}, [](auto const& t) { return t.Length(); });
            bench.batch(range.Size() * b).run(fmt::format("forward;{};{}", a, static_cast<double>(b)/static_cast<double>(n)), [&]() {
                std::size_t sz{0};
                for (auto const& tree : trees) {
                    Operon::ResidualEvaluator re(interpreter, tree, ds, target, range);
                    auto parameters = tree.GetCoefficients();
                    Eigen::Matrix<Operon::Scalar, -1, -1> jacobian(range.Size(), parameters.size());
                    auto autodiff = Operon::detail::Autodiff<decltype(re), Operon::Dual, Operon::Scalar, Eigen::ColMajor>;
                    autodiff(re, parameters.data(), nullptr, jacobian.data());
                    sz += jacobian.size();
                }
                return sz;
            });
            z += 10;
        } while (a < s);
    };

    auto benchReverse = [&](ankerl::nanobench::Bench& bench, std::size_t n, std::size_t s) {
        std::vector<Operon::Tree> trees(n);

        double a{0};
        auto z{20};

        do {
            std::uniform_int_distribution<size_t> dist(1, z);
            std::generate(trees.begin(), trees.end(), [&](){ return creator(rng, dist(rng), 1, 1000); }); // NOLINT
            Operon::Range range{0, ds.Rows()};
            a = std::transform_reduce(trees.begin(), trees.end(), double{0}, std::plus{}, [](auto const& t) { return t.CoefficientsCount(); }) / static_cast<double>(n);
            auto b = std::transform_reduce(trees.begin(), trees.end(), 0UL, std::plus{}, [](auto const& t) { return t.Length(); });
            bench.batch(range.Size() * b).run(fmt::format("reverse;{};{}", a, static_cast<double>(b)/static_cast<double>(n)), [&]() {
                std::size_t sz{0};
                Operon::DispatchTable<Operon::Scalar> dtable;
                for (auto const& tree : trees) {
                    Operon::Interpreteur interpreteur{ tree, ds, range, dtable};
                    auto parameters = tree.GetCoefficients();
                    Operon::DerivativeCalculator dt(interpreteur);
                    dt(parameters);
                    sz += dt.Jacobian().size();
                }
                return sz;
            });
            z += 10;
        } while (a < s);
    };

    constexpr auto n{5000};
    constexpr auto m{20};

    SUBCASE("forward mode") {
        benchForward(b, n, m);
    }

    SUBCASE("reverse mode") {
        benchReverse(b, n, m);
    }

    b.render(ankerl::nanobench::templates::csv(), std::cout);
}

} // namespace Operon::Test
