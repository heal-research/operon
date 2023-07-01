// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_TEST_HPP
#define OPERON_TEST_HPP

#include "thirdparty/elki_stats.hpp"
#include "thirdparty/nanobench.h"

#include "operon/core/dataset.hpp"

namespace Operon::Test::Util {
    inline auto RandomDataset(Operon::RandomGenerator& rng, int rows, int cols) -> Operon::Dataset {
        std::uniform_real_distribution<Operon::Scalar> dist(-1.f, +1.f);
        Eigen::Matrix<decltype(dist)::result_type, -1, -1> data(rows, cols);
        for (auto& v : data.reshaped()) { v = dist(rng); }
        Operon::Dataset ds(data);
        return ds;
    }
} // namespace Operon::Test::Util

#endif
