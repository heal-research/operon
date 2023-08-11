// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>   // for taskflow.for_each_index
#include "operon/interpreter/interpreter.hpp"

namespace Operon {
    auto EvaluateTrees(std::vector<Operon::Tree> const& trees, Operon::Dataset const& dataset, Operon::Range range, size_t nthread) -> std::vector<std::vector<Operon::Scalar>> {
        if (nthread == 0) { nthread = std::thread::hardware_concurrency(); }
        tf::Executor executor(nthread);
        tf::Taskflow taskflow;
        std::vector<std::vector<Operon::Scalar>> result(trees.size());
        Operon::DefaultDispatch dtable;
        using INT = Operon::Interpreter<Operon::Scalar, Operon::DefaultDispatch>;

        taskflow.for_each_index(size_t{0}, size_t{trees.size()}, size_t{1}, [&](size_t i) {
            result[i].resize(range.Size());
            Operon::Span<Operon::Scalar> s{result[i].data(), result[i].size()};
            INT{dtable, dataset, trees[i]}.Evaluate({}, range, s);
        });
        executor.run(taskflow);
        executor.wait_for_all();
        return result;
    }

    auto EvaluateTrees(std::vector<Operon::Tree> const& trees, Operon::Dataset const& dataset, Operon::Range range, std::span<Operon::Scalar> result, size_t nthread) -> void {
        if (nthread == 0) { nthread = std::thread::hardware_concurrency(); }
        tf::Executor executor(nthread);
        tf::Taskflow taskflow;
        Operon::DefaultDispatch dtable;
        using INT = Operon::Interpreter<Operon::Scalar, Operon::DefaultDispatch>;

        taskflow.for_each_index(size_t{0}, size_t{trees.size()}, size_t{1}, [&](size_t i) {
            auto res = result.subspan(i * range.Size(), range.Size());
            INT{dtable, dataset, trees[i]}.Evaluate({}, range, res);
        });
        executor.run(taskflow);
        executor.wait_for_all();
    }
} // namespace Operon
