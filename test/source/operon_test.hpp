// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_TEST_HPP
#define OPERON_TEST_HPP

#include "thirdparty/nanobench.h"

#include "operon/algorithms/config.hpp"
#include "operon/algorithms/nsga2.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/pset.hpp"
#include "operon/interpreter/dispatch_table.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/generator.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/operators/mutation.hpp"
#include "operon/operators/non_dominated_sorter.hpp"
#include "operon/operators/reinserter.hpp"

namespace Operon::Test {
namespace Util {
    auto RandomDataset(Operon::RandomGenerator& rng, int rows, int cols) -> Operon::Dataset {
        std::uniform_real_distribution<Operon::Scalar> dist(-1.f, +1.f);
        Eigen::Matrix<decltype(dist)::result_type, -1, -1> data(rows, cols);
        for (auto& v : data.reshaped()) { v = dist(rng); }
        Operon::Dataset ds(data);
        return ds;
    }
} // namespace Util
} // namespace Operon::Test

#endif
