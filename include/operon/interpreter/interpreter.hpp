// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_INTERPRETER_HPP
#define OPERON_INTERPRETER_HPP

#include <algorithm>
#include <optional>
#include <utility>

#include "operon/core/dataset.hpp"
#include "operon/core/dual.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "dispatch_table.hpp"

namespace Operon {

template<typename... Ts>
struct GenericInterpreter {
    using DTable = DispatchTable<Ts...>;

    explicit GenericInterpreter(DTable ft)
        : ftable_(std::move(ft))
    {
    }

    GenericInterpreter() : GenericInterpreter(DTable{}) { }

    // evaluate a tree and return a vector of values
    template <typename T>
    auto Evaluate(Tree const& tree, Dataset const& dataset, Range const range, T const* const parameters = nullptr) const noexcept -> Operon::Vector<T>
    {
        Operon::Vector<T> result(range.Size());
        Evaluate<T>(tree, dataset, range, Operon::Span<T>(result), parameters);
        return result;
    }

    template <typename T>
    auto Evaluate(Tree const& tree, Dataset const& dataset, Range const range, size_t const batchSize, T const* const parameters = nullptr) const noexcept -> Operon::Vector<T>
    {
        Operon::Vector<T> result(range.Size());
        Operon::Span<T> view(result);

        size_t n = range.Size() / batchSize;
        size_t m = range.Size() % batchSize;
        std::vector<size_t> indices(n + (m != 0));
        std::iota(indices.begin(), indices.end(), 0UL);
        std::for_each(indices.begin(), indices.end(), [&](auto idx) {
            auto start = range.Start() + idx * batchSize;
            auto end = std::min(start + batchSize, range.End());
            auto subview = view.subspan(idx * batchSize, end - start);
            Evaluate(tree, dataset, Range { start, end }, subview, parameters);
        });
        return result;
    }

    template <typename T>
    void Evaluate(Tree const& tree, Dataset const& dataset, Range const range, Operon::Span<T> result, T const* const parameters = nullptr) const noexcept
    {
        using Callable = typename DTable::template Callable<T>;
        const auto& nodes = tree.Nodes();
        EXPECT(!nodes.empty());

        constexpr int S = static_cast<Eigen::Index>(detail::BatchSize<T>::Value);
        Operon::Vector<detail::Array<T>> m(nodes.size());
        Eigen::Map<Eigen::Array<T, -1, 1>> res(result.data(), result.size(), 1);

        struct NodeMeta {
            T Param;
            Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> Values;
            std::optional<Callable const> Func;

            NodeMeta(T param, decltype(Values) values, decltype(Func) func)
                : Param(param), Values(values), Func(func)
            {
            }
        };

        Operon::Vector<NodeMeta> meta; meta.reserve(nodes.size());

        size_t idx = 0;
        for (size_t i = 0; i < nodes.size(); ++i) {
            auto const& n = nodes[i];

            const auto *ptr = n.IsVariable() ? dataset.GetValues(n.HashValue).subspan(range.Start(), range.Size()).data() : nullptr;
            const auto sz = ptr ? range.Size() : 0UL;
            meta.emplace_back(
                (parameters && n.Optimize) ? parameters[idx++] : T{n.Value},
                decltype(NodeMeta::Values)(ptr, static_cast<int64_t>(sz)),
                ftable_.template TryGet<T>(n.HashValue)
            );
            if (n.IsConstant()) { m[i].setConstant(meta.back().Param); }
        }

        int numRows = static_cast<int>(range.Size());
        for (int row = 0; row < numRows; row += S) {
            auto remainingRows = std::min(S, numRows - row);
            Operon::Range rg(range.Start() + row, range.Start() + row + remainingRows);

            for (size_t i = 0; i < nodes.size(); ++i) {
                auto const& s = nodes[i];
                auto const& [ param, values, func ] = meta[i];
                if (func) {
                    std::invoke(func.value(), m, nodes, i, rg);
                } else if (s.IsVariable()) {
                    m[i].segment(0, remainingRows) = meta[i].Param * values.segment(row, remainingRows).template cast<T>();
                }
            }
            // the final result is found in the last section of the buffer corresponding to the root node
            res.segment(row, remainingRows) = m.back().segment(0, remainingRows);
        }
    }

    auto GetDispatchTable() -> DTable& { return ftable_; }
    [[nodiscard]] auto GetDispatchTable() const -> DTable const& { return ftable_; }

private:
    DTable ftable_;
};

using Interpreter = GenericInterpreter<Operon::Scalar, Operon::Dual>;
} // namespace Operon


#endif
