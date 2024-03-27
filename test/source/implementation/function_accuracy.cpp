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

    TEST_CASE("function accuracy" * dt::test_suite("[implementation]")) {
        std::vector<std::pair<float, float>> domains {{
            {-10.F, +10.F},
            {-10000.F, +10000.F},
            {-31875756.0F, +31875756.0F},
            {-3.4028235e+38, +3.4028235e+38}
        }};

        auto constexpr n{ 1'000'000 };

        using unaryPtr  = std::add_pointer_t<float(float)>;
        using binaryPtr = std::add_pointer_t<float(float, float)>;

        std::vector<std::tuple<std::string, unaryPtr, unaryPtr>> unaryFunctions {
            { "exp",  static_cast<unaryPtr>(&std::exp),  &Operon::Backend::detail::mad::Exp },
            { "log",  static_cast<unaryPtr>(&std::log),  &Operon::Backend::detail::mad::Log },
            { "sin",  static_cast<unaryPtr>(&std::sin),  &Operon::Backend::detail::mad::Sin },
            { "cos",  static_cast<unaryPtr>(&std::cos),  &Operon::Backend::detail::mad::Cos },
            { "sqrt", static_cast<unaryPtr>(&std::sqrt), &Operon::Backend::detail::mad::Sqrt },
            { "tanh", static_cast<unaryPtr>(&std::tanh), &Operon::Backend::detail::mad::Tanh }
        };

        std::vector<std::tuple<std::string, binaryPtr, binaryPtr>> binaryFunctions {
            { "div", [](auto a, auto b){ return a / b; }, &Operon::Backend::detail::mad::Div }
        };

        Operon::RandomGenerator rng{1234};

        auto relativeError = [](auto yTrue, auto yEst) {
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
        };

        auto absoluteError = [](auto yTrue, auto yEst) {
            auto constexpr inf = std::numeric_limits<float>::infinity();
            if (std::isnan(yTrue) && std::isnan(yEst)) { return 0.F; }
            if (!std::isnan(yTrue) && std::isnan(yEst)) { return inf; }
            if (std::isnan(yTrue) && !std::isnan(yEst)) { return inf; }
            if (yTrue == yEst && std::isinf(yTrue)) { return 0.F; }
            return std::abs(yTrue - yEst);
        };

        SUBCASE("unary") {
            std::vector<float> absErr(n);
            std::vector<float> relErr(n);

            for (auto [a, b] : domains) {
                std::uniform_real_distribution<float> dist{a, b};

                for (auto const& [name, f, g] : unaryFunctions) {
                    for (auto i = 0; i < n; ++i) {
                        auto x = dist(rng);
                        auto y = f(x);
                        auto z = g(x);

                        if (y > 1e+38F) { y = std::numeric_limits<float>::infinity(); }
                        if (z > 1e+38F) { z = std::numeric_limits<float>::infinity(); }

                        absErr[i] = absoluteError(y, z);
                        relErr[i] = relativeError(y, z);

                        if ((std::isfinite(y) && !std::isfinite(z)) || (!std::isfinite(y) && std::isfinite(z))) {
                            fmt::print("true: {}({}) = {}\n", name, x, y);
                            fmt::print("pred: {}({}) = {}\n", name, x, z);
                            fmt::print("\n");
                        }
                    }

                    std::ranges::sort(absErr);
                    std::ranges::sort(relErr);
                    auto absErrMed = absErr[n/2];
                    auto relErrMed = relErr[n/2];
                    fmt::print("{},[{},{}],{:.12g},{:.12g}\n", name, a, b, absErrMed, 100 * relErrMed);
                }
                fmt::print("\n");
            }
        }

        SUBCASE("binary") {
            std::vector<float> absErr(n);
            std::vector<float> relErr(n);

            for (auto [a, b] : domains) {
                std::uniform_real_distribution<float> dist{a, b};

                for (auto const& [name, f, g] : binaryFunctions) {
                    for (auto i = 0; i < n; ++i) {
                        auto x1 = dist(rng);
                        auto x2 = dist(rng);
                        auto y = f(x1, x2);
                        auto z = g(x1, x2);

                        if (y > 1e+38F) { y = std::numeric_limits<float>::infinity(); }
                        if (z > 1e+38F) { z = std::numeric_limits<float>::infinity(); }

                        absErr[i] = absoluteError(y, z);
                        relErr[i] = relativeError(y, z);

                        if ((std::isfinite(y) && !std::isfinite(z)) || (!std::isfinite(y) && std::isfinite(z))) {
                            fmt::print("true: {}({}, {}) = {}\n", name, x1, x2, y);
                            fmt::print("pred: {}({}, {}) = {}\n", name, x1, x2, z);
                            fmt::print("\n");
                        }
                    }

                    std::ranges::sort(absErr);
                    std::ranges::sort(relErr);
                    auto absErrMed = absErr[n/2];
                    auto relErrMed = relErr[n/2];
                    fmt::print("{},[{},{}],{:.12g},{:.12g}\n", name, a, b, absErrMed, 100 * relErrMed);
                }
                fmt::print("\n");
            }
        }
    }
} // namespace Operon::Test