// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include <doctest/doctest.h>
#include "../operon_test.hpp"

#include <fmt/ranges.h>

#include "operon/error_metrics/error_metrics.hpp"

namespace dt = doctest;


namespace Operon::Test {
    namespace detail {
    };


    TEST_CASE("error metrics" * dt::test_suite("[implementation]"))
    {
        auto const n{100UL};

        std::vector<double> x(n);
        std::vector<double> y(n);
        std::vector<double> z(n);

        Operon::RandomGenerator rng{1234}; // NOLINT
        std::uniform_real_distribution<double> ureal(0, 1);

        for (auto i = 0UL; i < n; ++i) {
            x[i] = ureal(rng);
            y[i] = ureal(rng);
            z[i] = ureal(rng);
        }

        auto constexpr eps{1e-6};

        using std::cbegin;
        using std::cend;

        SUBCASE("mean") {
            auto m1 = Elki::MeanVariance::PopulationStats(x).Mean;
            auto m2 = vstat::univariate::accumulate<double>(cbegin(x), cend(x)).mean;
            CHECK(std::abs(m1-m2) < eps);
        }

        SUBCASE("weighted mean") {
            auto m1 = Elki::MeanVariance::PopulationStats(x, z).Mean;
            auto m2 = vstat::univariate::accumulate<double>(cbegin(x), cend(x), cbegin(z)).mean;
            CHECK(std::abs(m1-m2) < eps);
        }

        SUBCASE("variance") {
            auto v1 = Elki::MeanVariance::PopulationStats(x).Variance;
            auto v2 = vstat::univariate::accumulate<double>(cbegin(x), cend(x)).variance;
            CHECK(std::abs(v1-v2) < eps);
        }

        SUBCASE("weighted variance") {
            auto v1 = Elki::MeanVariance::PopulationStats(x, z).Variance;
            auto v2 = vstat::univariate::accumulate<double>(cbegin(x), cend(x), cbegin(z)).variance;
            fmt::print("v1: {}\nv2: {}\n", v1, v2);
            CHECK(std::abs(v1-v2) < eps);
        }

        SUBCASE("mse") {
            auto mse1 = Elki::MeanVariance::MSE(x, y);
            auto mse2 = MeanSquaredError(cbegin(x), cend(x), cbegin(y));
            CHECK(std::abs(mse1-mse2) < eps);
        }

        SUBCASE("weighted mse") {
            auto mse1 = Elki::MeanVariance::MSE(x, y, z);
            auto mse2 = MeanSquaredError(cbegin(x), cend(x), cbegin(y), cbegin(z));
            CHECK(std::abs(mse1-mse2) < eps);
        }

        SUBCASE("mae") {
            auto mae1 = Elki::MeanVariance::MAE(x, y);
            auto mae2 = MeanAbsoluteError(cbegin(x), cend(x), cbegin(y));
            CHECK(std::abs(mae1-mae2) < eps);
        }

        SUBCASE("weighted mae") {
            auto mae1 = Elki::MeanVariance::MAE(x, y, z);
            auto mae2 = MeanAbsoluteError(cbegin(x), cend(x), cbegin(y), cbegin(z));
            CHECK(std::abs(mae1-mae2) < eps);
        }

        SUBCASE("nmse") {
            auto nmse1 = Elki::MeanVariance::NMSE(x, y);
            auto nmse2 = NormalizedMeanSquaredError(cbegin(x), cend(x), cbegin(y));
            CHECK(std::abs(nmse1-nmse2) < eps);
        }

        SUBCASE("weighted nmse") {
            auto nmse1 = Elki::MeanVariance::NMSE(x, y, z);
            auto nmse2 = NormalizedMeanSquaredError(cbegin(x), cend(x), cbegin(y), cbegin(z));
            CHECK(std::abs(nmse1-nmse2) < eps);
        }

        SUBCASE("correlation") {
            auto c1 = Elki::MeanVariance::Corr(x, y);
            auto c2 = CorrelationCoefficient(cbegin(x), cend(x), cbegin(y));
            CHECK(std::abs(c1-c2) < eps);
        }

        SUBCASE("weighted correlation") {
            auto c1 = Elki::MeanVariance::Corr(x, y, z);
            auto c2 = CorrelationCoefficient(cbegin(x), cend(x), cbegin(y), cbegin(z));
            CHECK(std::abs(c1-c2) < eps);
        }

        SUBCASE("r2 score") {
            auto r1 = Elki::MeanVariance::R2(x, y);
            auto r2 = R2Score(cbegin(x), cend(x), cbegin(y));
            CHECK(std::abs(r1-r2) < eps);
        }

        SUBCASE("weighted r2 score") {
            auto r1 = Elki::MeanVariance::R2(x, y, z);
            auto r2 = R2Score(cbegin(x), cend(x), cbegin(y), cbegin(z));
            CHECK(std::abs(r1-r2) < eps);
        }
    }

} // namespace Operon::Test
