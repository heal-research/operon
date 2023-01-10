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
    ankerl::nanobench::Bench b;
    //b.output(nullptr);

    Operon::PrimitiveSet pset;
    Operon::PrimitiveSetConfig psetcfg = Operon::PrimitiveSet::Arithmetic | Operon::NodeType::Exp | Operon::NodeType::Log | Operon::NodeType::Sin | Operon::NodeType::Cos;
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
                for (auto const& tree : trees) {
                    auto coeff = tree.GetCoefficients();
                    auto jac = dc(tree, ds, coeff, range);
                    sz += jac.size();
                }
                return sz;
            });
            z += 10;
        } while (a < s);
    };

    constexpr auto n{1000};
    constexpr auto m{50};

    SUBCASE("forward mode") {
        Operon::GenericInterpreter<Operon::Scalar, Operon::Dual> interpreter;
        Operon::Autodiff::Forward::DerivativeCalculator dc{ interpreter };
        benchmark(b, dc, "forward", n, m);
    }

    SUBCASE("reverse mode") {
        Operon::GenericInterpreter<Operon::Scalar> interpreter;
        Operon::Autodiff::Reverse::DerivativeCalculator dc{ interpreter };
        benchmark(b, dc, "reverse", n, m);
    }

    b.render(ankerl::nanobench::templates::csv(), std::cout);
}

} // namespace Operon::Test
