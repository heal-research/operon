// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include <doctest/doctest.h>
#include <random>

#include "../operon_test.hpp"

#include "operon/core/pset.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/core/problem.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/interpreter/dual.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/parser/infix.hpp"


#include <iomanip>
#include <iostream>
#include <fstream>
#include <vstat/vstat.hpp>

namespace dt = doctest;

namespace Operon::Test {
    auto RelativeError(auto yTrue, auto yEst) {
        auto constexpr inf = std::numeric_limits<float>::infinity();
        if (std::isnan(yTrue) && std::isnan(yEst)) { return 0.F; }
        if (!std::isnan(yTrue) && std::isnan(yEst)) { return inf; }
        if (std::isnan(yTrue) && !std::isnan(yEst)) { return inf; }
        if (yTrue == yEst && std::isinf(yTrue)) { return 0.F; }
        if (yTrue != yEst && std::isinf(yTrue) && std::isinf(yEst)) { return inf; }
        if (std::isinf(yTrue) && std::isfinite(yEst)) { return inf; }
        if (yTrue == 0 && yEst == 0) { return 0.F; }
        if (yTrue == 0 && yEst != 0) { return inf; }
        return std::abs(yTrue - yEst) / std::abs(yTrue);
    }

    auto AbsoluteError(auto yTrue, auto yEst) {
        auto constexpr inf = std::numeric_limits<float>::infinity();
        if (std::isnan(yTrue) && std::isnan(yEst)) { return 0.F; }
        if (!std::isnan(yTrue) && std::isnan(yEst)) { return inf; }
        if (std::isnan(yTrue) && !std::isnan(yEst)) { return inf; }
        if (yTrue == yEst && std::isinf(yTrue)) { return 0.F; }
        return std::abs(yTrue - yEst);
    }

    auto Median(auto& v) {
        using T = typename std::remove_cvref_t<decltype(v)>::value_type;
        if (v.empty()) {
            return T{0};
        }
        auto n = v.size() / 2;
        std::nth_element(v.begin(), v.begin() + n, v.end());
        auto med = v[n];
        if (v.size() % 2 != 0) { // If the set size is even
            auto m = max_element(v.begin(), v.begin() + n);
            med = (*m + med) / 2;
        }
        return med;
    }

#if defined (OPERON_MATH_MAD_TRANSCENDENTAL_FAST) || defined (OPERON_MATH_MAD_TRANSCENDENTAL_FASTER) || defined (OPERON_MATH_MAD_TRANSCENDENTAL_FASTEST)

    TEST_CASE("test sin") {
        std::vector<Operon::Scalar> inputs{ 31875756.1F, 3.4028235E+38F, 1E+38F };

        for (auto v : inputs) {
            fmt::print("{:.20f} ", std::sin(v));
        }
        fmt::print("\n");

        for (auto v : inputs) {
            fmt::print("{:.20f} ", Operon::Backend::detail::mad::SinImpl<0>(v));
        }
        fmt::print("\n");

        for (auto v : inputs) {
            fmt::print("{:.20f} ", Operon::Backend::detail::mad::SinImpl<1>(v));
        }
        fmt::print("\n");

    }

    TEST_CASE("test tanh") {
        auto constexpr d{ 10.F };
        auto constexpr n{ 1'000'000 };

        std::vector<float> xs(n);
        std::vector<float> ys(n);
        std::vector<float> zs(n);

        Operon::RandomGenerator rng{1234};

        auto f = std::fopen("tanh.csv", "w");
        std::uniform_real_distribution<float> dist{-d, +d};

        for (auto i = 0; i < n; ++i) {
            auto x = dist(rng);
            auto y = std::tanh(x);
            auto z = Operon::Backend::detail::mad::Tanh(x);

            fmt::print(f, "{:.20f},{:.20f},{:.20f}\n", x, y, z);
        }
    }
#endif

    TEST_CASE("function accuracy" * dt::test_suite("[implementation]")) {
        // std::vector<std::pair<float, float>> domains {{
        //     {0, +10.F},
        //     {0, +10000.F},
        //     {0, +31875756.0F},
        //     {0, +3.4028235e+38F}
        //     // { std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max() }
        // }};

        std::vector<float> domains{ 10.F, 10000.F, 31875756.0F, 3.4028235e+38F};

        auto constexpr n{ 1'000'000 };

        using UnaryPtr  = std::add_pointer_t<float(float)>;
        using BinaryPtr = std::add_pointer_t<float(float, float)>;

#if defined (OPERON_MATH_MAD_TRANSCENDENTAL_FAST) || defined (OPERON_MATH_MAD_TRANSCENDENTAL_FASTER) || defined (OPERON_MATH_MAD_TRANSCENDENTAL_FASTEST)
        std::vector<std::tuple<std::string, UnaryPtr, UnaryPtr>> unaryFunctions {
            { "exp",  static_cast<UnaryPtr>(&std::exp),  &Operon::Backend::detail::mad::Exp },
            { "log",  static_cast<UnaryPtr>(&std::log),  &Operon::Backend::detail::mad::Log },
            { "sin",  static_cast<UnaryPtr>(&std::sin),  &Operon::Backend::detail::mad::Sin },
            { "cos",  static_cast<UnaryPtr>(&std::cos),  &Operon::Backend::detail::mad::Cos },
            { "sqrt", static_cast<UnaryPtr>(&std::sqrt), &Operon::Backend::detail::mad::Sqrt },
            { "tanh", static_cast<UnaryPtr>(&std::tanh), &Operon::Backend::detail::mad::Tanh }
        };

        std::vector<std::tuple<std::string, BinaryPtr, BinaryPtr>> binaryFunctions {
            { "div", [](auto a, auto b){ return a / b; }, &Operon::Backend::detail::mad::Div }
        };
#elif defined (OPERON_MATH_VDT)
        std::vector<std::tuple<std::string, UnaryPtr, UnaryPtr>> unaryFunctions {
            { "exp",  static_cast<UnaryPtr>(&std::exp),  &Operon::Backend::detail::vdt::Exp },
            { "log",  static_cast<UnaryPtr>(&std::log),  &Operon::Backend::detail::vdt::Log },
            { "sin",  static_cast<UnaryPtr>(&std::sin),  &Operon::Backend::detail::vdt::Sin },
            { "cos",  static_cast<UnaryPtr>(&std::cos),  &Operon::Backend::detail::vdt::Cos },
            { "sqrt", static_cast<UnaryPtr>(&std::sqrt), &Operon::Backend::detail::vdt::Sqrt },
            { "tanh", static_cast<UnaryPtr>(&std::tanh), &Operon::Backend::detail::vdt::Tanh }
        };

        std::vector<std::tuple<std::string, BinaryPtr, BinaryPtr>> binaryFunctions {
            { "div", [](auto a, auto b){ return a / b; }, &Operon::Backend::detail::vdt::Div }
        };
#else
        std::vector<std::tuple<std::string, UnaryPtr, UnaryPtr>> unaryFunctions {};
        std::vector<std::tuple<std::string, BinaryPtr, BinaryPtr>> binaryFunctions {};
#endif

        Operon::RandomGenerator rng{1234};

        std::vector<float> absErr(n);
        std::vector<float> relErr(n);

        std::uniform_real_distribution<float> dist{-1, +1};

        auto f1 = std::fopen("unary.csv", "w");
        auto f2 = std::fopen("binary.csv", "w");

        fmt::print(f1, "name,domain,x,y_true,y_est\n");
        fmt::print(f2, "name,domain,x1,x2,y_true,y_est\n");

        for (auto d : domains) {
            auto dom = fmt::format("[{}, {}]", -d, +d);

            for (auto const& [name, f, g] : unaryFunctions) {
                for (auto i = 0; i < n; ++i) {
                    auto x = dist(rng) * d;
                    auto y = f(x);
                    auto z = g(x);

                    fmt::print(f1, "{},\"{}\",{},{},{}\n", name, dom, x, y, z);

                    absErr[i] = AbsoluteError(y, z);
                    relErr[i] = RelativeError(y, z);
                }

                auto absErrMed = Median(absErr);
                auto relErrMed = Median(relErr);

                fmt::print("{},[{},{}],{:.12e},{:.12e}\n", name, -d, +d, absErrMed, 100 * relErrMed);
            }

            for (auto const& [name, f, g] : binaryFunctions) {
                absErr.clear(); absErr.resize(n);
                relErr.clear(); relErr.resize(n);

                for (auto i = 0; i < n; ++i) {
                    auto x1 = dist(rng) * d;
                    auto x2 = dist(rng) * d;
                    auto y = f(x1, x2);
                    auto z = g(x1, x2);

                    fmt::print(f2, "{},\"{}\",{},{},{},{}\n", name, dom, x1, x2, y, z);

                    absErr[i] = AbsoluteError(y, z);
                    relErr[i] = RelativeError(y, z);
                }

                auto absErrMed = Median(absErr);
                auto relErrMed = Median(relErr);

                fmt::print("{},[{},{}],{:.12g},{:.12g}\n", name, -d, +d, absErrMed, 100 * relErrMed);
            }
            fmt::print("\n");
        }
    }

    TEST_CASE("derivative accuracy" * dt::test_suite("[implementation]")) {
        const size_t n         = 10;
        const size_t maxDepth  = 2;

        constexpr size_t nrow = 100000;
        constexpr size_t ncol = 2;

        Operon::RandomGenerator rd(1234);
        // auto ds = Util::RandomDataset(rd, nrow, ncol);
        Eigen::Array<Operon::Scalar, -1, -1> d = decltype(d)::Ones(nrow, ncol);
        Operon::Dataset ds(d);

        auto variables = ds.GetVariables();
        auto target = variables.back().Name;
        auto inputs = ds.VariableHashes();
        std::erase(inputs, ds.GetVariable(target)->Hash);
        Range range = { 0, ds.Rows<std::size_t>() };
        using DTable = DefaultDispatch;
        DTable dtable;

        std::array primitives{ NodeType::Div, NodeType::Sin, NodeType::Cos, NodeType::Exp, NodeType::Log, NodeType::Sqrt, NodeType::Tanh };

        std::uniform_real_distribution<Operon::Scalar> dist{-10.F, +10.F};

        using Derivative = std::function<std::vector<Operon::Scalar>(std::vector<Operon::Scalar> const&)>;

        auto dsin = [](auto const& x) -> std::vector<Operon::Scalar> {
            return { std::cos(x[0]) };
        };

        auto dcos = [](auto const& x) -> std::vector<Operon::Scalar> {
            return { -std::sin(x[0]) };
        };

        auto dexp = [](auto const& x) -> std::vector<Operon::Scalar> {
            return { std::exp(x[0]) };
        };

        auto dlog = [](auto const& x) -> std::vector<Operon::Scalar> {
            return { 1 / x[0] };
        };

        auto dsqrt = [](auto const& x) -> std::vector<Operon::Scalar> {
            return { 0.5F * std::sqrt(x[0]) };
        };

        auto dtanh = [](auto const& x) -> std::vector<Operon::Scalar> {
            auto t = std::tanh(x[0]);
            return { 1.F - t*t };
        };

        auto ddiv = [](auto const& x) -> std::vector<Operon::Scalar> {
            return { -x[1]/(x[0]*x[0]), 1/x[0] };
        };

        Operon::Map<Operon::NodeType, Derivative> dmap {
            { Operon::NodeType::Sin, dsin },
            { Operon::NodeType::Cos, dcos },
            { Operon::NodeType::Exp, dexp },
            { Operon::NodeType::Log, dlog },
            { Operon::NodeType::Sqrt, dsqrt },
            { Operon::NodeType::Tanh, dtanh },
            { Operon::NodeType::Div, ddiv }
        };

        for (auto p : primitives) {
            Operon::PrimitiveSet pset{p | NodeType::Variable};
            auto creator = BalancedTreeCreator { pset, inputs };
            std::vector<Tree> trees(n);
            auto maxLength = p == NodeType::Div ? 3 : 2;

            auto tree = creator(rd, maxLength, 0, maxDepth);


            auto str = Operon::InfixFormatter::Format(tree, ds);

            for (auto i = 0UL; i < ds.Rows<std::size_t>(); ++i) {
                std::vector<Operon::Scalar> values;

                // compute derivative using our derivatives defined above
                for (auto& n : tree.Nodes()) {
                    if (n.Arity > 0) {
                        continue;
                    }
                    n.Optimize = true;
                    n.Value = dist(rd);
                    values.push_back(n.Value);
                }

                // compute jacobian
                auto jac = Operon::Interpreter<Operon::Scalar, DTable>(dtable, ds, tree).JacRev(tree.GetCoefficients(), {i, i+1});

                // report results
                std::string line;
                if (auto it = dmap.find(p); it != dmap.end()) {
                    ENSURE(it->first == p);
                    auto res = it->second(values);

                    fmt::format_to(std::back_inserter(line), "{},", Operon::Node(p).Name());
                    for (auto v : values) {
                        fmt::format_to(std::back_inserter(line), "{:.20f},", v);
                    }
                    for (auto v : jac.reshaped()) {
                        fmt::format_to(std::back_inserter(line), "{:.20f},", v);
                    }
                    for (auto v : res) {
                        fmt::format_to(std::back_inserter(line), "{:.20f},", v);
                    }
                }
                line.pop_back();
                fmt::print("{}\n", line);
            }
        }
    }
} // namespace Operon::Test
