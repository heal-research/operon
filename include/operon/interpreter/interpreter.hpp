// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_INTERPRETER_HPP
#define OPERON_INTERPRETER_HPP

#include <algorithm>

#include "core/dataset.hpp"
#include "core/tree.hpp"
#include "core/types.hpp"
#include "interpreter/dispatch_table.hpp"

namespace Operon {

template<typename T>
using Span = nonstd::span<T>;

struct Interpreter {
    Interpreter(DispatchTable const& ft)
        : ftable(ft)
    {
    }

    Interpreter(DispatchTable&& ft) : ftable(std::move(ft)) { }

    Interpreter() : Interpreter(DispatchTable{}) { }

    // evaluate a tree and return a vector of values
    template <typename T>
    Operon::Vector<T> Evaluate(Tree const& tree, Dataset const& dataset, Range const range, T const* const parameters = nullptr) const noexcept
    {
        Operon::Vector<T> result(range.Size());
        Evaluate<T>(tree, dataset, range, Operon::Span<T>(result), parameters);
        return result;
    }

    template <typename T>
    Operon::Vector<T> Evaluate(Tree const& tree, Dataset const& dataset, Range const range, size_t const batchSize, T const* const parameters = nullptr) const noexcept
    {
        Operon::Vector<T> result(range.Size());
        Operon::Span<T> view(result);

        size_t n = range.Size() / batchSize;
        size_t m = range.Size() % batchSize;
        std::vector<size_t> indices(n + (m != 0));
        std::iota(indices.begin(), indices.end(), 0ul);
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
        const auto& nodes = tree.Nodes();
        EXPECT(nodes.size() > 0);

        constexpr int S = static_cast<int>(detail::batch_size<T>::value);

        using M = Eigen::Array<T, S, Eigen::Dynamic, Eigen::ColMajor>;
        M m = M::Zero(S, nodes.size());

        Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1, Eigen::ColMajor>> res(result.data(), result.size(), 1);

        Operon::Vector<std::reference_wrapper<DispatchTable::Callable<T> const>> funcs;
        Operon::Vector<T> params(nodes.size());
        Operon::Vector<Operon::Span<const Operon::Scalar>> vals(nodes.size());
        size_t idx = 0;

        for (size_t i = 0; i < nodes.size(); ++i) {
            auto const& n = nodes[i];
            if (n.IsLeaf()) {
                auto v = parameters ? parameters[idx++] : T{n.Value};
                if (n.IsConstant()) {
                    m.col(i).setConstant(v);
                } else if (n.IsVariable()) {
                    params[i] = v;
                    vals[i] = dataset.GetValues(n.HashValue);
                }
            } else {
                funcs.push_back(std::ref(ftable.Get<T>(nodes[i].HashValue)));
            }
        }

        auto lastCol = m.col(nodes.size() - 1);

        int numRows = static_cast<int>(range.Size());
        for (int row = 0; row < numRows; row += S) {
            auto remainingRows = std::min(S, numRows - row);

            idx = 0;
            for (size_t i = 0; i < nodes.size(); ++i) {
                auto const& s = nodes[i];
                if (s.IsConstant()) {
                    continue; // constants already filled above
                } else if (s.IsVariable()) {
                    Eigen::Map<const Eigen::Array<Operon::Scalar, Eigen::Dynamic, 1, Eigen::ColMajor>> seg(vals[i].data() + range.Start() + row, remainingRows);
                    m.col(i).segment(0, remainingRows) = params[i] * seg.cast<T>();
                } else {
                    funcs[idx++](m, nodes, i, range.Start() + row);
                }
            }
            // the final result is found in the last section of the buffer corresponding to the root node
            res.segment(row, remainingRows) = lastCol.segment(0, remainingRows);
        }
    }

    template<typename T>
    static void Evaluate(DispatchTable& ft, Tree const& tree, Dataset const& dataset, Range const range, Operon::Span<T> result, T const* const parameters = nullptr) noexcept {
        Interpreter interpreter(ft);
        interpreter.Evaluate(tree, dataset, range, result, parameters);
    }

    template<typename T>
    static Operon::Vector<T> Evaluate(DispatchTable& ft, Tree const& tree, Dataset const& dataset, Range const range, T const* const parameters = nullptr) {
        Interpreter interpreter(ft);
        return interpreter.Evaluate(tree, dataset, range, parameters);  
    }

    DispatchTable& GetDispatchTable() { return ftable; }
    DispatchTable const& GetDispatchTable() const { return ftable; }

private:
    DispatchTable ftable;
};
};

#endif
