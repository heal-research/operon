// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include "../operon_test.hpp"
#include "operon/error_metrics/error_metrics.hpp"

namespace Operon::Test {

TEST_CASE("Error metrics", "[metrics]") // NOLINT(readability-function-cognitive-complexity)
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

    SECTION("mean") {
        auto m1 = Elki::MeanVariance::PopulationStats(x).Mean;
        auto m2 = vstat::univariate::accumulate<double>(cbegin(x), cend(x)).mean;
        CHECK(std::abs(m1 - m2) < eps);
    }

    SECTION("weighted mean") {
        auto m1 = Elki::MeanVariance::PopulationStats(x, z).Mean;
        auto m2 = vstat::univariate::accumulate<double>(cbegin(x), cend(x), cbegin(z)).mean;
        CHECK(std::abs(m1 - m2) < eps);
    }

    SECTION("variance") {
        auto v1 = Elki::MeanVariance::PopulationStats(x).Variance;
        auto v2 = vstat::univariate::accumulate<double>(cbegin(x), cend(x)).variance;
        CHECK(std::abs(v1 - v2) < eps);
    }

    SECTION("weighted variance") {
        auto v1 = Elki::MeanVariance::PopulationStats(x, z).Variance;
        auto v2 = vstat::univariate::accumulate<double>(cbegin(x), cend(x), cbegin(z)).variance;
        CHECK(std::abs(v1 - v2) < eps);
    }

    SECTION("mse") {
        auto mse1 = Elki::MeanVariance::MSE(x, y);
        auto mse2 = MeanSquaredError(cbegin(x), cend(x), cbegin(y));
        CHECK(std::abs(mse1 - mse2) < eps);
    }

    SECTION("weighted mse") {
        auto mse1 = Elki::MeanVariance::MSE(x, y, z);
        auto mse2 = MeanSquaredError(cbegin(x), cend(x), cbegin(y), cbegin(z));
        CHECK(std::abs(mse1 - mse2) < eps);
    }

    SECTION("mae") {
        auto mae1 = Elki::MeanVariance::MAE(x, y);
        auto mae2 = MeanAbsoluteError(cbegin(x), cend(x), cbegin(y));
        CHECK(std::abs(mae1 - mae2) < eps);
    }

    SECTION("weighted mae") {
        auto mae1 = Elki::MeanVariance::MAE(x, y, z);
        auto mae2 = MeanAbsoluteError(cbegin(x), cend(x), cbegin(y), cbegin(z));
        CHECK(std::abs(mae1 - mae2) < eps);
    }

    SECTION("nmse") {
        auto nmse1 = Elki::MeanVariance::NMSE(x, y);
        auto nmse2 = NormalizedMeanSquaredError(cbegin(x), cend(x), cbegin(y));
        CHECK(std::abs(nmse1 - nmse2) < eps);
    }

    SECTION("weighted nmse") {
        auto nmse1 = Elki::MeanVariance::NMSE(x, y, z);
        auto nmse2 = NormalizedMeanSquaredError(cbegin(x), cend(x), cbegin(y), cbegin(z));
        CHECK(std::abs(nmse1 - nmse2) < eps);
    }

    SECTION("correlation") {
        auto c1 = Elki::MeanVariance::Corr(x, y);
        auto c2 = CorrelationCoefficient(cbegin(x), cend(x), cbegin(y));
        CHECK(std::abs(c1 - c2) < eps);
    }

    SECTION("weighted correlation") {
        auto c1 = Elki::MeanVariance::Corr(x, y, z);
        auto c2 = CorrelationCoefficient(cbegin(x), cend(x), cbegin(y), cbegin(z));
        CHECK(std::abs(c1 - c2) < eps);
    }

    SECTION("r2 score") {
        auto r1 = Elki::MeanVariance::R2(x, y);
        auto r2 = R2Score(cbegin(x), cend(x), cbegin(y));
        CHECK(std::abs(r1 - r2) < eps);
    }

    SECTION("weighted r2 score") {
        auto r1 = Elki::MeanVariance::R2(x, y, z);
        auto r2 = R2Score(cbegin(x), cend(x), cbegin(y), cbegin(z));
        CHECK(std::abs(r1 - r2) < eps);
    }
}

TEST_CASE("Error metrics edge cases", "[metrics]")
{
    auto constexpr eps{1e-6};
    auto const n{50UL};

    using std::cbegin;
    using std::cend;

    SECTION("Perfect prediction gives MSE = 0") {
        std::vector<double> x(n);
        Operon::RandomGenerator rng{42};
        std::uniform_real_distribution<double> dist(0, 1);
        for (auto& v : x) { v = dist(rng); }

        auto mse = MeanSquaredError(cbegin(x), cend(x), cbegin(x));
        CHECK(mse < eps);
    }

    SECTION("Perfect prediction gives R2 = 1") {
        std::vector<double> x(n);
        Operon::RandomGenerator rng{42};
        std::uniform_real_distribution<double> dist(0, 1);
        for (auto& v : x) { v = dist(rng); }

        auto r2 = R2Score(cbegin(x), cend(x), cbegin(x));
        CHECK(std::abs(r2 - 1.0) < eps);
    }
}

TEST_CASE("Error metrics finite subset", "[metrics]")
{
    auto constexpr eps{1e-6};
    auto const n{100UL};

    using std::cbegin;
    using std::cend;

    Operon::RandomGenerator rng{7}; // NOLINT
    std::uniform_real_distribution<double> ureal(0, 1);

    std::vector<double> x(n);
    std::vector<double> y(n);
    std::vector<double> w(n);
    for (auto i = 0UL; i < n; ++i) {
        x[i] = ureal(rng);
        y[i] = ureal(rng);
        w[i] = 0.1 + ureal(rng);
    }
    // inject a few non-finite predictions
    x[0] = std::numeric_limits<double>::quiet_NaN();
    x[n / 2] = std::numeric_limits<double>::infinity();
    x[n - 1] = -std::numeric_limits<double>::infinity();

    std::vector<double> xf, yf, wf;
    for (auto i = 0UL; i < n; ++i) {
        if (std::isfinite(x[i])) {
            xf.push_back(x[i]);
            yf.push_back(y[i]);
            wf.push_back(w[i]);
        }
    }

    SECTION("mse skips non-finite rows") {
        auto [mse, skipped] = MeanSquaredErrorFinite(cbegin(x), cend(x), cbegin(y));
        auto ref = MeanSquaredError(cbegin(xf), cend(xf), cbegin(yf));
        CHECK(skipped == 3UL);
        CHECK(std::abs(mse - ref) < eps);
    }

    SECTION("weighted mse skips non-finite rows") {
        auto [mse, skipped] = MeanSquaredErrorFinite(cbegin(x), cend(x), cbegin(y), cbegin(w));
        auto ref = MeanSquaredError(cbegin(xf), cend(xf), cbegin(yf), cbegin(wf));
        CHECK(skipped == 3UL);
        CHECK(std::abs(mse - ref) < eps);
    }

    SECTION("nmse skips non-finite rows") {
        auto [nmse, skipped] = NormalizedMeanSquaredErrorFinite(cbegin(x), cend(x), cbegin(y));
        auto ref = NormalizedMeanSquaredError(cbegin(xf), cend(xf), cbegin(yf));
        CHECK(skipped == 3UL);
        CHECK(std::abs(nmse - ref) < eps);
    }

    SECTION("weighted nmse skips non-finite rows") {
        auto [nmse, skipped] = NormalizedMeanSquaredErrorFinite(cbegin(x), cend(x), cbegin(y), cbegin(w));
        auto ref = NormalizedMeanSquaredError(cbegin(xf), cend(xf), cbegin(yf), cbegin(wf));
        CHECK(skipped == 3UL);
        CHECK(std::abs(nmse - ref) < eps);
    }

    SECTION("sse skips non-finite rows") {
        auto [sse, skipped] = SumOfSquaredErrorsFinite(cbegin(x), cend(x), cbegin(y));
        auto ref = SumOfSquaredErrors(cbegin(xf), cend(xf), cbegin(yf));
        CHECK(skipped == 3UL);
        CHECK(std::abs(sse - ref) < eps);
    }

    SECTION("weighted sse skips non-finite rows") {
        auto [sse, skipped] = SumOfSquaredErrorsFinite(cbegin(x), cend(x), cbegin(y), cbegin(w));
        auto ref = SumOfSquaredErrors(cbegin(xf), cend(xf), cbegin(yf), cbegin(wf));
        CHECK(skipped == 3UL);
        CHECK(std::abs(sse - ref) < eps);
    }

    SECTION("rmse skips non-finite rows") {
        auto [rmse, skipped] = RootMeanSquaredErrorFinite(cbegin(x), cend(x), cbegin(y));
        auto ref = RootMeanSquaredError(cbegin(xf), cend(xf), cbegin(yf));
        CHECK(skipped == 3UL);
        CHECK(std::abs(rmse - ref) < eps);
    }

    SECTION("weighted rmse skips non-finite rows") {
        auto [rmse, skipped] = RootMeanSquaredErrorFinite(cbegin(x), cend(x), cbegin(y), cbegin(w));
        auto ref = RootMeanSquaredError(cbegin(xf), cend(xf), cbegin(yf), cbegin(wf));
        CHECK(skipped == 3UL);
        CHECK(std::abs(rmse - ref) < eps);
    }

    SECTION("mae skips non-finite rows") {
        auto [mae, skipped] = MeanAbsoluteErrorFinite(cbegin(x), cend(x), cbegin(y));
        auto ref = MeanAbsoluteError(cbegin(xf), cend(xf), cbegin(yf));
        CHECK(skipped == 3UL);
        CHECK(std::abs(mae - ref) < eps);
    }

    SECTION("weighted mae skips non-finite rows") {
        auto [mae, skipped] = MeanAbsoluteErrorFinite(cbegin(x), cend(x), cbegin(y), cbegin(w));
        auto ref = MeanAbsoluteError(cbegin(xf), cend(xf), cbegin(yf), cbegin(wf));
        CHECK(skipped == 3UL);
        CHECK(std::abs(mae - ref) < eps);
    }

    SECTION("all-finite input matches the plain metric exactly") {
        auto [mse, skipped] = MeanSquaredErrorFinite(cbegin(xf), cend(xf), cbegin(yf));
        auto ref = MeanSquaredError(cbegin(xf), cend(xf), cbegin(yf));
        CHECK(skipped == 0UL);
        CHECK(std::abs(mse - ref) < eps);
    }
}

} // namespace Operon::Test
