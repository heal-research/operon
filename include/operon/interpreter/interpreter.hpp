// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_INTERPRETER_HPP
#define OPERON_INTERPRETER_HPP

#include <algorithm>
#include <optional>
#include <utility>

#include "operon/autodiff/forward/dual.hpp"
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
    requires std::invocable<F, Dispatch::Matrix<T>&, int>
    auto operator()(Tree const& tree, Dataset const& dataset, Range const range, Operon::Span<T const> coeff = {}, F&& callback = F{}) const noexcept -> Operon::Vector<T>
    {
        Operon::Vector<T> result(range.Size());
        this->operator()<T>(tree, dataset, range, result, coeff, callback);
        return result;
    }

    template <typename T = Operon::Scalar, typename F = Dispatch::Noop>
    requires std::invocable<F, Dispatch::Matrix<T>&, int>
    void operator()(Tree const& tree, Dataset const& dataset, Range const range, Operon::Span<T> result, Operon::Span<T const> coeff = {}, F&& callback = F{}) const noexcept
    {
        using Callable = Dispatch::Callable<T>;
        const auto& nodes = tree.Nodes();
        EXPECT(!nodes.empty());

        auto constexpr S{ static_cast<int>(Dispatch::BatchSize<T>) };
        Dispatch::Matrix<T> m = decltype(m)::Zero(S, nodes.size());

        using NodeMeta = std::tuple<T, Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const>, std::optional<Callable const>>;
        Operon::Vector<NodeMeta> meta; meta.reserve(nodes.size());

        size_t idx = 0;
        int numRows = static_cast<int>(range.Size());
        for (size_t i = 0; i < nodes.size(); ++i) {
            auto const& n = nodes[i];

            auto const* ptr = n.IsVariable() ? dataset.GetValues(n.HashValue).subspan(range.Start(), numRows).data() : nullptr;
            auto const param = (!coeff.empty() && n.Optimize) ? coeff[idx++] : T{n.Value};

            meta.push_back({
                param,
                std::tuple_element_t<1, NodeMeta>(ptr, numRows),
                dtable_.template TryGet<T>(n.HashValue)
            });
            if (n.IsConstant()) { m.col(i).setConstant(param); }
        }

        for (int row = 0; row < numRows; row += S) {
            auto remainingRows = std::min(S, numRows - row);
            Operon::Range rg(range.Start() + row, range.Start() + row + remainingRows);

            for (size_t i = 0; i < nodes.size(); ++i) {
                auto const& [ param, values, func ] = meta[i];
                if (nodes[i].IsVariable()) {
                    m.col(i).segment(0, remainingRows) = param * values.segment(row, remainingRows).template cast<T>();
                } else if (func) {
                    std::invoke(*func, m, nodes, i, rg);
                }
            }
            // the final result is found in the last section of the buffer corresponding to the root node
            if (result.size() == range.Size()) {
                Eigen::Map<Eigen::Array<T, -1, 1>>(result.data(), result.size()).segment(row, remainingRows) = m.col(m.cols()-1).segment(0, remainingRows);
            }
            callback(m, row);
        }
    }

    auto GetDispatchTable() -> DTable& { return dtable_; }
    [[nodiscard]] auto GetDispatchTable() const -> DTable const& { return dtable_; }

private:
    DTable dtable_;
};

using Interpreter = GenericInterpreter<Operon::Scalar, Operon::Dual>;
//using Interpreter = GenericInterpreter<Operon::Scalar>;

// convenience method to interpret many trees in parallel (mostly useful from the python wrapper)
auto OPERON_EXPORT EvaluateTrees(std::vector<Operon::Tree> const& trees, Operon::Dataset const& dataset, Operon::Range range, size_t nthread = 0) -> std::vector<std::vector<Operon::Scalar>> ;
auto OPERON_EXPORT EvaluateTrees(std::vector<Operon::Tree> const& trees, Operon::Dataset const& dataset, Operon::Range range, std::span<Operon::Scalar> result, size_t nthread) -> void;

} // namespace Operon


#endif
