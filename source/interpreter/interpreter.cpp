// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <taskflow/algorithm/for_each.hpp>   // for taskflow.for_each_index
#include "operon/interpreter/interpreter.hpp"

namespace Operon {
    auto TryEvaluateTrees(Operon::Vector<Operon::Tree> const& trees, Operon::Dataset const* dataset, Operon::Range range, size_t nthread)
        -> tl::expected<Operon::Vector<Operon::Vector<Operon::Scalar>>, TreeEvaluationError>
    {
        if (nthread == 0) { nthread = std::thread::hardware_concurrency(); }
        tf::Executor executor(nthread);
        tf::Taskflow taskflow;
        Operon::Vector<Operon::Vector<Operon::Scalar>> result(trees.size());
        Operon::Vector<std::optional<InterpreterError>> errors(trees.size());
        Operon::ScalarDispatch dtable;
        using INT = Operon::Interpreter<Operon::Scalar, Operon::ScalarDispatch>;

        taskflow.for_each_index(size_t{0}, size_t{trees.size()}, size_t{1}, [&](size_t i) -> void {
            auto evaluated = INT{&dtable, dataset, &trees[i]}.TryEvaluate({}, range);
            if (evaluated) {
                result[i] = std::move(*evaluated);
            } else {
                errors[i] = std::move(evaluated.error());
            }
        });
        executor.run(taskflow).get();
        for (size_t i = 0; i < errors.size(); ++i) {
            if (errors[i]) { return tl::unexpected(TreeEvaluationError{i, std::move(*errors[i])}); }
        }
        return result;
    }

    auto TryEvaluateTrees(Operon::Vector<Operon::Tree> const& trees, Operon::Dataset const* dataset, Operon::Range range, std::span<Operon::Scalar> result, size_t nthread)
        -> tl::expected<void, TreeEvaluationError>
    {
        auto const expectedSize = trees.size() * range.Size();
        if (result.size() != expectedSize) {
            return tl::unexpected(TreeEvaluationError {
                trees.size(),
                InterpreterError { InterpreterError::Code::InvalidOutputSize, {}, expectedSize, result.size() },
            });
        }
        if (nthread == 0) { nthread = std::thread::hardware_concurrency(); }
        tf::Executor executor(nthread);
        tf::Taskflow taskflow;
        Operon::Vector<std::optional<InterpreterError>> errors(trees.size());
        Operon::ScalarDispatch dtable;
        using INT = Operon::Interpreter<Operon::Scalar, Operon::ScalarDispatch>;

        taskflow.for_each_index(size_t{0}, size_t{trees.size()}, size_t{1}, [&](size_t i) -> void {
            auto output = result.subspan(i * range.Size(), range.Size());
            auto evaluated = INT{&dtable, dataset, &trees[i]}.TryEvaluate({}, range, output);
            if (!evaluated) { errors[i] = std::move(evaluated.error()); }
        });
        executor.run(taskflow).get();
        for (size_t i = 0; i < errors.size(); ++i) {
            if (errors[i]) { return tl::unexpected(TreeEvaluationError{i, std::move(*errors[i])}); }
        }
        return {};
    }

    auto EvaluateTrees(Operon::Vector<Operon::Tree> const& trees, Operon::Dataset const* dataset, Operon::Range range, size_t nthread) -> Operon::Vector<Operon::Vector<Operon::Scalar>>
    {
        auto result = TryEvaluateTrees(trees, dataset, range, nthread);
        if (!result) { throw std::runtime_error(FormatInterpreterError(result.error().Error)); }
        return std::move(*result);
    }

    auto EvaluateTrees(Operon::Vector<Operon::Tree> const& trees, Operon::Dataset const* dataset, Operon::Range range, std::span<Operon::Scalar> result, size_t nthread) -> void
    {
        auto evaluated = TryEvaluateTrees(trees, dataset, range, result, nthread);
        if (!evaluated) { throw std::runtime_error(FormatInterpreterError(evaluated.error().Error)); }
    }
} // namespace Operon
