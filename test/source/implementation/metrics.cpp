// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "operon/core/metrics.hpp"
#include <doctest/doctest.h>

#include <vector>

using std::vector;

namespace Operon::Test {

TEST_CASE("coefficient of determination")
{
    vector<double> x{2.5, 0.0, 2, 8};
    vector<double> y{3, -0.5, 2, 7};

    auto r2 = Operon::CoefficientOfDetermination<double>(x, y); 
    fmt::print("r2: {}\n", r2);
}

} // namespace Operon::Test
