// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_NNLS_RESIDUAL_EVALUATOR_HPP
#define OPERON_NNLS_RESIDUAL_EVALUATOR_HPP

#include <Eigen/Core>
#include "operon/interpreter/interpreter.hpp"

namespace Operon {
// simple functor that wraps everything together and provides residuals
struct ResidualEvaluator {
    ResidualEvaluator(Interpreter const& interpreter, Tree const& tree, Dataset const& dataset, const Operon::Span<const Operon::Scalar> targetValues, Range const range)
        : interpreter_(interpreter)
        , tree_(tree)
        , dataset_(dataset)
        , range_(range)
        , target_(targetValues)
        , numParameters_(tree_.get().GetCoefficients().size())
    {
    }

    template <typename T>
    auto operator()(T const* const* parameters, T* residuals) const -> bool
    {
        Operon::Span<T> result(residuals, target_.size());
        GetInterpreter().Evaluate<T>(tree_.get(), dataset_.get(), range_, result, parameters[0]);
        Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1, Eigen::ColMajor>> resMap(residuals, target_.size());
        Eigen::Map<const Eigen::Array<Operon::Scalar, Eigen::Dynamic, 1, Eigen::ColMajor>> targetMap(target_.data(), static_cast<Eigen::Index>(target_.size()));
        resMap -= targetMap.cast<T>();
        return true;
    }

    [[nodiscard]] auto NumParameters() const -> size_t { return numParameters_; }
    [[nodiscard]] auto NumResiduals() const -> size_t { return target_.size(); }

    [[nodiscard]] auto GetInterpreter() const -> Interpreter const& { return interpreter_.get(); }

private:
    std::reference_wrapper<Interpreter const> interpreter_;
    std::reference_wrapper<Tree const> tree_;
    std::reference_wrapper<Dataset const> dataset_;
    Range range_;
    Operon::Span<const Operon::Scalar> target_;
    size_t numParameters_; // cache the number of parameters in the tree
};
} // namespace Operon

#endif
