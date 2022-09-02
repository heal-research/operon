// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include <doctest/doctest.h>
#include <vstat/vstat.hpp>
#include "operon/core/dataset.hpp"

namespace Operon::Test {

TEST_CASE("correlation")
{
    const auto *target = "Y";
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);

    auto names = ds.VariableNames();
    Eigen::Matrix<double, -1, -1> corr(names.size(), names.size());

    std::sort(names.begin(), names.end());

    for (auto i = 0UL; i < names.size() - 1; ++i) {
        corr(i, i) = 1;
        for (auto j = i+1; j < names.size(); ++j) {
            auto v1 = ds.GetValues(names[i]);
            auto v2 = ds.GetValues(names[j]);
            corr(i, j) = corr(j, i) = vstat::bivariate::accumulate<Operon::Scalar>(
                    v1.data(), v2.data(), v1.size()
                    ).correlation;
        }
    }

    for (auto const& n : names) {
        fmt::print("{} ", n);
    }
    fmt::print("\n");
    std::cout << corr << "\n";
}
} // namespace Operon::Test
