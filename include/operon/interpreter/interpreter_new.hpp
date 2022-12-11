// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_INTERPRETER_NEW_HPP
#define OPERON_INTERPRETER_NEW_HPP

#include <type_traits>
#include <utility>

#include "operon/core/dataset.hpp"
#include "operon/core/dual.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "dispatch_table.hpp"

namespace Operon {

namespace detail {
    auto const NOP = [](auto&&... args) { /* no op */ };
} // namespace detail

template<typename... Ts>
struct Interpreteur {
    Operon::Tree const& Tree;
    Operon::Dataset const& Dataset;
    Operon::Range const Range; // NOLINT
    Operon::DispatchTable<Ts...> const& Table;

    Interpreteur(Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Range const range, Operon::DispatchTable<Ts...> const& dtable = Operon::DispatchTable<Ts...>{})
        : Tree(tree), Dataset(dataset), Range(range), Table(dtable)
    { }

    template<typename T, typename F>
    auto operator()(Operon::Span<T const> parameters, Operon::Span<T> result, F&& callback = detail::NOP) const {
        using Callable = typename Operon::DispatchTable<Ts...>::template Callable<T>;
        const auto& nodes = Tree.Nodes();
        EXPECT(!nodes.empty());

        constexpr int S = static_cast<Eigen::Index>(detail::BatchSize<T>::Value);
        Operon::Vector<detail::Array<T>> m(nodes.size());
        Eigen::Map<Eigen::Array<T, -1, 1>> res(result.data(), result.size(), 1);

        using NodeMeta = std::tuple<T, Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const>, std::optional<Callable const>>;
        Operon::Vector<NodeMeta> meta; meta.reserve(nodes.size());

        size_t idx = 0;
        for (auto i = 0; i < std::ssize(nodes); ++i) {
            auto const& n = nodes[i];
            const auto *ptr = n.IsVariable() ? Dataset.GetValues(n.HashValue).subspan(Range.Start(), Range.Size()).data() : nullptr;
            const auto sz = static_cast<int64_t>(ptr ? Range.Size() : 0UL);
            auto const param = (!parameters.empty() && n.Optimize) ? parameters[idx++] : T{n.Value};
            meta.emplace_back(
                param,
                std::tuple_element_t<1, NodeMeta>(ptr, sz),
                Table.template TryGet<T>(n.HashValue)
            );
            if (n.IsConstant()) { m[i].setConstant(param); }
        }

        int numRows = static_cast<int>(Range.Size());
        for (int row = 0; row < numRows; row += S) {
            auto remainingRows = std::min(S, numRows - row);
            Operon::Range rg(Range.Start() + row, Range.Start() + row + remainingRows);

            for (size_t i = 0; i < nodes.size(); ++i) {
                auto const& [ param, values, func ] = meta[i];
                if (func) {
                    std::invoke(func.value(), m, nodes, i, rg);
                } else if (nodes[i].IsVariable()) {
                    m[i].segment(0, remainingRows) = param * values.segment(row, remainingRows).template cast<T>();
                }
            }
            // the final result is found in the last section of the buffer corresponding to the root node
            res.segment(row, remainingRows) = m.back().segment(0, remainingRows);
            callback(m);
        }
    }

    template<typename T, typename F = decltype(detail::NOP)>
    auto operator()(Operon::Span<T const> parameters, F&& callback = F{}) const -> std::vector<T> {
        std::vector<T> result(Range.Size());
        (*this)(parameters, std::span{result.data(), result.size()}, callback);
        return result;
    }

    template<typename T, typename F = decltype(detail::NOP)>
    auto operator()(F&& callback = F{}) const -> std::vector<T> {
        auto parameters = Tree.GetCoefficients();
        return (*this)(std::span{parameters.data(), parameters.size()}, callback);
    }
};

} // namespace Operon

#endif
