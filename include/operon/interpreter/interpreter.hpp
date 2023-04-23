// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_INTERPRETER_HPP
#define OPERON_INTERPRETER_HPP

#include <algorithm>
#include <optional>
#include <utility>

#include "operon/autodiff/dual.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "dispatch_table.hpp"

namespace Operon {

template<typename... Ts>
struct GenericInterpreter {
    using DTable = DispatchTable<Ts...>;

    explicit GenericInterpreter(DTable dt)
        : dtable_(std::move(dt))
    { }

    GenericInterpreter() : GenericInterpreter(DTable{}) { }

    // evaluate a tree and return a vector of values
    template <typename T = Operon::Scalar, typename F = Dispatch::Noop>
    requires std::invocable<F, Dispatch::Matrix<T> const&, int>
    auto operator()(Tree const& tree, Dataset const& dataset, Range const range, Operon::Span<T const> coeff = {}, F&& callback = F{}) const noexcept -> Operon::Vector<T>
    {
        Operon::Vector<T> result(range.Size());
        this->operator()<T>(tree, dataset, range, result, coeff, callback);
        return result;
    }

    template <typename T = Operon::Scalar, typename F = Dispatch::Noop>
    requires std::invocable<F, Dispatch::Matrix<T> const&, int>
    void operator()(Tree const& tree, Dataset const& dataset, Range const range, Operon::Span<T> result, Operon::Span<T const> coeff = {}, F&& callback = F{}) const noexcept
    {
        using Callable = Dispatch::Callable<T>;
        const auto& nodes = tree.Nodes();
        EXPECT(!nodes.empty());

        auto constexpr S{ static_cast<int>(Dispatch::BatchSize<T>) };
        auto const nn{ std::ssize(nodes) };
        Dispatch::Matrix<T> m = decltype(m)::Zero(S, nodes.size());

        using NodeMeta = std::tuple<T, Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const>, std::optional<Callable const>>;
        Operon::Vector<NodeMeta> meta; meta.reserve(nodes.size());

        size_t idx = 0;
        int len = static_cast<int>(range.Size());
        for (auto i = 0; i < nn; ++i) {
            auto const& n = nodes[i];

            auto const* ptr = n.IsVariable() ? dataset.GetValues(n.HashValue).subspan(range.Start(), len).data() : nullptr;
            auto const coefficient = (!coeff.empty() && n.Optimize) ? coeff[idx++] : T{n.Value};

            meta.push_back({
                coefficient,
                std::tuple_element_t<1, NodeMeta>(ptr, len),
                dtable_.template TryGet<T>(n.HashValue)
            });
            if (n.IsConstant()) { m.col(i).setConstant(coefficient); }
        }

        for (int row = 0; row < len; row += S) {
            auto remainingRows = std::min(S, len - row);
            Operon::Range rg(range.Start() + row, range.Start() + row + remainingRows);

            for (auto i = 0; i < nn; ++i) {
                auto const& [ param, values, func ] = meta[i];
                if (nodes[i].IsVariable()) {
                    m.col(i).head(remainingRows) = param * values.segment(row, remainingRows).template cast<T>();
                } else if (func) {
                    std::invoke(*func, m, nodes, i, rg);
                    m.col(i).head(remainingRows) *= param;
                }
            }
            std::invoke(callback, m, row);

            // the final result is found in the last section of the buffer corresponding to the root node
            // sometimes the interpreter is used to fill in the primal trace for reverce-mode autodiff to compute the Jacobian
            // in this case, the residual is not needed and memory for it is not allocated, so we need the check below
            if (result.size() == range.Size()) {
                Eigen::Map<Eigen::Array<T, -1, 1>>(result.data(), result.size()).segment(row, remainingRows) = m.col(m.cols()-1).head(remainingRows);
            }
        }
    }

    auto GetDispatchTable() -> DTable& { return dtable_; }
    [[nodiscard]] auto GetDispatchTable() const -> DTable const& { return dtable_; }

private:
    DTable dtable_;
};

using Interpreter = GenericInterpreter<Operon::Scalar, Operon::Dual>;

// convenience method to interpret many trees in parallel (mostly useful from the python wrapper)
auto OPERON_EXPORT EvaluateTrees(std::vector<Operon::Tree> const& trees, Operon::Dataset const& dataset, Operon::Range range, size_t nthread = 0) -> std::vector<std::vector<Operon::Scalar>> ;
auto OPERON_EXPORT EvaluateTrees(std::vector<Operon::Tree> const& trees, Operon::Dataset const& dataset, Operon::Range range, std::span<Operon::Scalar> result, size_t nthread) -> void;
} // namespace Operon
#endif
