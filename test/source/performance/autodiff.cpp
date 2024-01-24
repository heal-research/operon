// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include <doctest/doctest.h>
#include "../operon_test.hpp"

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

TEST_CASE("autodiff performance" * dt::test_suite("[performance]")) {
    constexpr auto nrow{1000};
    constexpr auto ncol{10};

    Operon::RandomGenerator rng(0);
    auto ds = Operon::Test::Util::RandomDataset(rng, nrow, ncol);
    nb::Bench b;
    //b.output(nullptr);
    b.timeUnit(std::chrono::milliseconds(1), "ms");

    Operon::PrimitiveSet pset{ Operon::PrimitiveSet::Arithmetic | Operon::NodeType::Exp | Operon::NodeType::Log | Operon::NodeType::Sin | Operon::NodeType::Cos | Operon::NodeType::Sqrt | Operon::NodeType::Pow | Operon::NodeType::Tanh };
    Operon::BalancedTreeCreator creator(pset, ds.VariableHashes());

    auto constexpr initialSize{20};
    auto benchmark = [&](ankerl::nanobench::Bench& bench, auto&& f, std::string const& prefix, std::size_t n, std::size_t s) {
        std::vector<Operon::Tree> trees(n);
        double a{0};
        auto z{initialSize}; // max size
        Operon::Range range{0, ds.Rows<std::size_t>()};
        std::vector<Operon::Scalar> y(ds.Rows());
        Operon::Span<Operon::Scalar> target{y.data(), y.size()};

        do {
            std::uniform_int_distribution<size_t> dist(1, z);
            std::generate(trees.begin(), trees.end(), [&](){ return creator(rng, dist(rng), 1, 1000); }); // NOLINT
            a = std::transform_reduce(trees.begin(), trees.end(), double{0}, std::plus{}, [](auto const& t) { return t.CoefficientsCount(); }) / static_cast<double>(n);
            auto b = std::transform_reduce(trees.begin(), trees.end(), 0UL, std::plus{}, [](auto const& t) { return t.Length(); });
            bench.batch(trees.size()).run(fmt::format("{};{};{}", prefix, a, static_cast<double>(b)/static_cast<double>(n)), [&]() {
                for (auto const& tree : trees) { f(ds, tree, range); }
            });
            z += 10; // NOLINT
        } while (a < s); // NOLINT
    };

    constexpr auto n{1000};
    constexpr auto m{50};

    using DTable = DispatchTable<Operon::Scalar>;
    DTable dtable;
    using INT = Interpreter<Operon::Scalar, DTable>;

    auto residual = [](auto const& ds, auto const& tree, auto range) {
        auto coeff = tree.GetCoefficients();
        return INT::Evaluate(tree, ds, range);
    };

    auto jacrev = [&](auto const& ds, auto const& tree, auto range) {
        auto coeff = tree.GetCoefficients();
        return INT{dtable, ds, tree}.JacRev(coeff, range);
    };

    auto jacfwd = [&](auto const& ds, auto const& tree, auto range) {
        auto coeff = tree.GetCoefficients();
        return INT{dtable, ds, tree}.JacFwd(coeff, range);
    };

    SUBCASE("residual") {
        benchmark(b, residual, "residual", n, m);
    }

    SUBCASE("forward") {
        benchmark(b, jacfwd, "forward;jacobian", n, m);
    }

    SUBCASE("reverse") {
        benchmark(b, jacrev, "reverse;jacobian", n, m);
    }

    b.render(ankerl::nanobench::templates::csv(), std::cout);
}

TEST_CASE("reverse mode" * dt::test_suite("[performance]")) {
    constexpr auto nrow{1000};
    constexpr auto ncol{10};

    Operon::RandomGenerator rng(0);
    auto ds = Operon::Test::Util::RandomDataset(rng, nrow, ncol);
    nb::Bench b;
    //b.output(nullptr);
    b.timeUnit(std::chrono::milliseconds(1), "ms");

    Operon::PrimitiveSet pset{ Operon::PrimitiveSet::Arithmetic | Operon::NodeType::Exp | Operon::NodeType::Log | Operon::NodeType::Sin | Operon::NodeType::Cos | Operon::NodeType::Sqrt };
    Operon::BalancedTreeCreator creator(pset, ds.VariableHashes());

    using DTable = DispatchTable<Operon::Scalar>;
    DTable dtable;
    using INT = Interpreter<Operon::Scalar, DTable>;

    constexpr auto maxlength{ 100 };
    constexpr auto maxdepth { 10 };
    constexpr auto numtrees { 10000 };

    std::vector<Operon::Tree> trees;
    trees.reserve(numtrees);

    std::uniform_int_distribution<size_t> dist(1, maxlength);
    for (auto i = 0; i < numtrees; ++i) {
        trees.push_back(creator(rng, dist(rng), 1, maxdepth));
    }
    auto totalnodes = std::transform_reduce(trees.begin(), trees.end(), 0.0, std::plus{}, [](auto const& tree) { return tree.Length(); });
    fmt::print("average nodes: {}\n", totalnodes / numtrees);
    Operon::Range range(0, ds.Rows());
    b.batch(numtrees).run("rev", [&]() {
        for (auto const& tree : trees) {
            auto coeff { tree.GetCoefficients() };
            INT{dtable, ds, tree}.JacRev(coeff, range);
        }
    });
}

TEST_CASE("primitive performance" * dt::test_suite("[performance]")) {
    auto constexpr N = 100;

    Operon::RandomGenerator rng(0);
    std::uniform_real_distribution<Operon::Scalar> dist;

    auto ds = Util::RandomDataset(rng, Backend::BatchSize<Operon::Scalar> * 100, 1);
    Range rg(0, ds.Rows());

    auto generate = [&](Node n) {
        std::vector<Node> nodes{n};
        for (auto k = 0; k < n.Arity; ++k) {
            nodes.push_back(Node::Constant(dist(rng)));
            nodes.back().Optimize = true;
        }
        std::ranges::reverse(nodes);
        return Tree{nodes}.UpdateNodes();
    };

    nb::Bench b;
    b.output(nullptr);

    Operon::DefaultDispatch dt;

    std::string name = "eigen";

    for (auto i = 0UL; i < NodeTypes::Count; ++i) {
        auto t = static_cast<NodeType>(1UL << i);
        Node n{t}; n.Value = dist(rng);
        if (n.IsLeaf()) { continue; }
        if (t == NodeType::Dynamic) { continue; }
        std::vector<Tree> trees;
        trees.reserve(N);
        for (auto j = 0; j < N; ++j) {
            trees.push_back(generate(n));
        }

        std::vector<Operon::Scalar> out(ds.Rows());

        b.title(name).batch(N * (n.Arity + 1) * ds.Rows()).run(fmt::format("{}\";\"res", n.Name()), [&]() {
            auto sum = 0.;
            for (auto const& tree : trees) {
                auto coeff = tree.GetCoefficients();
                Operon::Interpreter<Operon::Scalar, Operon::DefaultDispatch>(dt, ds, tree).Evaluate(coeff, rg, out);
                sum += std::reduce(out.begin(), out.end());
            }
            return sum;
        });
    }

    for (auto i = 0UL; i < NodeTypes::Count; ++i) {
        auto t = static_cast<NodeType>(1UL << i);
        Node n{t}; n.Value = dist(rng);
        if (n.IsLeaf()) { continue; }
        if (t == NodeType::Dynamic) { continue; }
        std::vector<Tree> trees;
        trees.reserve(N);
        for (auto j = 0; j < N; ++j) {
            trees.push_back(generate(n));
        }

        std::vector<Operon::Scalar> jac(ds.Rows() * (n.Arity + 1));

        b.title(name).batch(N * (n.Arity + 1) * ds.Rows()).run(fmt::format("{}\";\"jac", n.Name()), [&]() {
            auto sum = 0.;
            for (auto const& tree : trees) {
                auto coeff = tree.GetCoefficients();
                Operon::Interpreter<Operon::Scalar, Operon::DefaultDispatch>(dt, ds, tree).JacRev(coeff, rg, jac);
                sum += std::reduce(jac.begin(), jac.end());
            }
            return sum;
        });
    }

    b.render(nb::templates::csv(), std::cout);
}

TEST_CASE("optimizer performance" * dt::test_suite("[performance]")) {
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto range = Range { 0, ds.Rows<std::size_t>() };

    auto const& X = ds.Values(); // NOLINT

    Operon::Map<std::string, Operon::Hash> vars;
    for (auto const& v : ds.GetVariables()) {
        fmt::print("{} : {} {}\n", v.Name, v.Hash, v.Index);
        vars[v.Name] = v.Hash;
    }

    Operon::Problem problem{ds, range, range};

    std::vector<size_t> indices(range.Size());
    std::iota(indices.begin(), indices.end(), 0);

    Operon::RandomGenerator rng(0);

    using DTable = DispatchTable<Operon::Scalar>;
    DTable dtable;
    using INT = Interpreter<Operon::Scalar, DTable>;

    auto benchmark = [&](ankerl::nanobench::Bench& bench, DTable const& dt, Operon::BalancedTreeCreator const& creator, std::string const& prefix, std::size_t n, std::size_t s, std::size_t r) {
        std::vector<Operon::Tree> trees(n);
        double a{0};
        auto z{20}; // max size
        Operon::Range range{0, ds.Rows<std::size_t>()};
        std::vector<Operon::Scalar> y(ds.Rows());
        Operon::Span<Operon::Scalar> target{y.data(), y.size()};

        //constexpr auto iterations{50};

        do {
            std::uniform_int_distribution<size_t> dist(1, z);
            std::generate(trees.begin(), trees.end(), [&](){ return creator(rng, dist(rng), 1, 1000); }); // NOLINT
            a = std::transform_reduce(trees.begin(), trees.end(), double{0}, std::plus{}, [](auto const& t) { return t.CoefficientsCount(); }) / static_cast<double>(n);
            auto b = std::transform_reduce(trees.begin(), trees.end(), 0UL, std::plus{}, [](auto const& t) { return t.Length(); });

            bench.batch(range.Size() * b).run(fmt::format("{};{};{};{}", prefix, a, static_cast<double>(b)/static_cast<double>(n), r), [&]() {
                std::size_t sz{0};
                for (auto const& tree : trees) {
                    INT interpreter{dtable, ds, tree};
                    Operon::LevenbergMarquardtOptimizer<DTable, OptimizerType::Eigen> optimizer{dt, problem};
                    auto summary = optimizer.Optimize(rng, tree);
                    sz += summary.FinalParameters.size();
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
    Operon::BalancedTreeCreator creator(pset, ds.VariableHashes());

    SUBCASE("forward") {
        benchmark(b, dtable, creator, "forward", n, m, r);
    }

    SUBCASE("reverse") {
        benchmark(b, dtable, creator, "reverse", n, m, r);
    }

    b.render(nb::templates::csv(), std::cout);
}
} // namespace Operon::Test
