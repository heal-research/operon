// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include <doctest/doctest.h>
#include "../operon_test.hpp"

#include <ranges>

#include "operon/error_metrics/correlation_coefficient.hpp"
#include "operon/error_metrics/error_metrics.hpp"
#include "operon/error_metrics/mean_squared_error.hpp"

namespace dt = doctest;

namespace Operon::Test {
    TEST_CASE("mean squared error" * dt::test_suite("[error metrics]"))
    {
        auto const n{100UL};

        std::vector<double> x(n);
        std::vector<double> y(n);
        std::vector<double> z(n);

        Operon::RandomGenerator rng{1234}; // NOLINT
        std::uniform_real_distribution<double> ureal(0, 1);

        for (auto i = 0; i < n; ++i) {
            x[i] = ureal(rng);
            y[i] = ureal(rng);
            z[i] = ureal(rng);
        }

        auto constexpr eps{1e-6};

        auto resSqr = [](auto a, auto b){ auto e = a-b; return e*e; };
        auto resAbs = [](auto a, auto b){ return std::abs(a-b); };

        auto naiveMean = [](auto const& x) {
            return std::reduce(x.begin(), x.end(), 0.0, std::plus{}) / std::ssize(x);
        };

        auto naiveMeanWeighted = [](auto const& x, auto const& z) {
            return std::transform_reduce(x.begin(), x.end(), z.begin(), 0.0, std::plus{}, std::multiplies{}) /
                std::reduce(z.begin(), z.end(), 0.0, std::plus{});
        };

        auto naiveErr = [](auto const& x, auto const& y, auto&& f) {
            ENSURE(x.size() == y.size());
            auto mse{0.0};
            auto w{0.0};
            for (auto i = 0; i < std::ssize(x); ++i) {
                auto const v = f(x[i], y[i]);
                mse += v;
                w += 1;
            }
            return mse / w; 
        };

        auto naiveErrWeighted = [](auto const& x, auto const& y, auto const& z, auto&& f) {
            ENSURE(x.size() == y.size());
            ENSURE(x.size() == z.size());
            auto mse{0.0};
            auto w{0.0};
            for (auto i = 0; i < std::ssize(x); ++i) {
                auto const v = f(x[i], y[i]);
                mse += z[i] * v;
                w += z[i];
            }
            return mse / w;
        };

        auto naiveCorr = [&](auto const& x, auto const& y) {
            ENSURE(x.size() == y.size());
            auto corr{0.0};

            auto mx = naiveMean(x);
            auto my = naiveMean(y);

            auto sx{0.0};
            auto sy{0.0};
            auto sxy{0.0};
            for (auto i = 0; i < std::ssize(x); ++i) {
                auto dx = x[i] - mx;
                sx += dx * dx;

                auto dy = y[i] - my;
                sy += dy * dy;

                sxy += (x[i] - mx) * (y[i] - my);
            }
            sx /= std::ssize(x);
            sy /= std::ssize(x);
            sxy /= std::ssize(x);

            return sxy / std::sqrt(sx * sy);
        };

        auto naiveCorrWeighted = [&](auto const& x, auto const& y, auto const& z) {
            ENSURE(x.size() == y.size());
            auto corr{0.0};

            auto mx = naiveMeanWeighted(x, z);
            auto my = naiveMeanWeighted(y, z);
            auto sz = std::reduce(z.begin(), z.end(), 0.0, std::plus{});

            auto sx{0.0};
            auto sy{0.0};
            auto sxy{0.0};
            for (auto i = 0; i < std::ssize(x); ++i) {
                auto dx = x[i] - mx;
                sx += z[i] * dx * dx;

                auto dy = y[i] - my;
                sy += z[i] * dy * dy;

                sxy += z[i] * (x[i] - mx) * (y[i] - my);
            }
            sx  /= sz;
            sy  /= sz;
            sxy /= sz;

            return sxy / std::sqrt(sx * sy);
        };

        SUBCASE("mean") {
            auto m1 = naiveMean(x);
            auto m2 = vstat::univariate::accumulate<double>(x.begin(), x.end()).mean;
            CHECK(std::abs(m1-m2) < eps);
        }

        SUBCASE("weighted mean") {
            auto m1 = naiveMeanWeighted(x, z);
            auto m2 = vstat::univariate::accumulate<double>(x.begin(), x.end(), z.begin()).mean;
            CHECK(std::abs(m1-m2) < eps);
        }

        SUBCASE("mse") {
            auto mse1 = naiveErr(x, y, resSqr);
            auto mse2 = MeanSquaredError(x.begin(), x.end(), y.begin());
            CHECK(std::abs(mse1-mse2) < eps);
        }

        SUBCASE("weighted mse") {
            auto mse1 = naiveErrWeighted(x, y, z, resSqr);
            auto mse2 = MeanSquaredError(x.begin(), x.end(), y.begin(), z.begin());
            CHECK(std::abs(mse1-mse2) < eps);
        }

        SUBCASE("mae") {
            auto mae1 = naiveErr(x, y, resAbs);
            auto mae2 = MeanAbsoluteError(x.begin(), x.end(), y.begin());
            CHECK(std::abs(mae1-mae2) < eps);
        }
        
        SUBCASE("weighted mae") {
            auto mae1 = naiveErrWeighted(x, y, z, resAbs);
            auto mae2 = MeanAbsoluteError(x.begin(), x.end(), y.begin(), z.begin());
            CHECK(std::abs(mae1-mae2) < eps);
        }

        SUBCASE("correlation") {
            auto c1 = naiveCorr(x, y);
            auto c2 = CorrelationCoefficient(x.begin(), x.end(), y.begin());
            CHECK(std::abs(c1-c2) < eps);
        }

        SUBCASE("weighted correlation") {
            auto c1 = naiveCorrWeighted(x, y, z);
            auto c2 = CorrelationCoefficient(x.begin(), x.end(), y.begin(), z.begin());
            CHECK(std::abs(c1-c2) < eps);
        }
    }

} // namespace Operon::Test
