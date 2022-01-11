// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

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

    template <typename T, typename C = typename DTable::template Callable<T>>
    void Evaluate(Tree const& tree, Dataset const& dataset, Range const range, Operon::Span<T> result, T const* const parameters = nullptr) const noexcept
    {
        const auto& nodes = tree.Nodes();
        EXPECT(!nodes.empty());

        constexpr int S = static_cast<int>(detail::BatchSize<T>::Value);

        using M = Eigen::Array<T, S, -1>;
        M m = M::Zero(S, nodes.size());

        Eigen::Map<Eigen::Array<T, -1, 1>> res(result.data(), result.size(), 1);

        struct NodeMeta {
            T Param;
            Operon::Span<Operon::Scalar const> Values;
            std::optional<C const> Func;
        };

        Operon::Vector<NodeMeta> meta; meta.reserve(nodes.size());

        size_t idx = 0;
        for (size_t i = 0; i < nodes.size(); ++i) {
            auto const& n = nodes[i];

            if (n.IsLeaf()) {
                auto v = parameters ? parameters[idx++] : T{n.Value};
                Operon::Span<Operon::Scalar const> vals{};
                if (n.IsConstant()) { m.col(i).setConstant(v); }
                if (n.IsVariable()) { vals = dataset.GetValues(n.HashValue).subspan(range.Start(), range.Size()); }
                auto call = n.IsDynamic() ? std::make_optional(ftable_.template Get<T>(n.HashValue)) : std::nullopt;
                meta.push_back({ v, vals, call });
            } else {
                meta.push_back({ T{}, {}, std::make_optional(ftable_.template Get<T>(n.HashValue)) });
            }
        }

        auto lastCol = m.col(nodes.size() - 1);

        int numRows = static_cast<int>(range.Size());
        for (int row = 0; row < numRows; row += S) {
            auto remainingRows = std::min(S, numRows - row);

            for (size_t i = 0; i < nodes.size(); ++i) {
                auto const& s = nodes[i];
                auto const& [ param, values, func ] = meta[i];
                if (func) {
                    func.value()(m, nodes, i, range.Start() + row);
                } else if (s.IsVariable()) {
                    Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> seg(values.data() + row, remainingRows);
                    m.col(i).segment(0, remainingRows) = meta[i].Param * seg.cast<T>();
                }
            }
            // the final result is found in the last section of the buffer corresponding to the root node
            res.segment(row, remainingRows) = lastCol.segment(0, remainingRows);
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
