/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC>
 * Copyright (C) 2020 Bogdan Burlacu
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE.
 */

#ifndef OPERON_EVAL
#define OPERON_EVAL

#include "dataset.hpp"
#include "eval_detail.hpp"
#include "gsl/gsl"
#include "tree.hpp"
#include "pset.hpp"
#include "range.hpp"

namespace Operon {
// evaluate a tree and return a vector of values
template <typename T>
Operon::Vector<T> Evaluate(Tree const& tree, Dataset const& dataset, Range const range, T const* const parameters = nullptr)
{
    Operon::Vector<T> result(range.Size());
    Evaluate(tree, dataset, range, gsl::span<T>(result), parameters);
    return result;
}

template <typename T>
Operon::Vector<T> Evaluate(Tree const& tree, Dataset const& dataset, Range const range, size_t const batchSize, T const* const parameters = nullptr)
{
    Operon::Vector<Operon::Scalar> result(range.Size());
    gsl::span<Operon::Scalar> view(result);

    size_t n = range.Size() / batchSize;
    size_t m = range.Size() % batchSize;
    std::vector<size_t> indices(n + (m != 0));
    std::iota(indices.begin(), indices.end(), 0ul);
    std::for_each(indices.begin(), indices.end(), [&](auto idx)
    {
        auto start = range.Start() + idx * batchSize;
        auto end = std::min(start + batchSize, range.End());
        auto subview = view.subspan(idx * batchSize, end-start);
        Evaluate<Operon::Scalar>(tree, dataset, Range{ start, end }, subview, parameters);
    });
    return result;
}

template <typename T, size_t S, NodeType N>
constexpr auto dispatch_op = detail::dispatch_op<T, S, N>;

template <typename T, size_t S = 512 / sizeof(T)>
void Evaluate(Tree const& tree, Dataset const& dataset, Range const range, gsl::span<T> result, T const* const parameters = nullptr) noexcept
{
    const auto& nodes = tree.Nodes();
    EXPECT(nodes.size() > 0);
    Eigen::Array<T, S, Eigen::Dynamic, Eigen::ColMajor> m(S, nodes.size());
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1, Eigen::ColMajor>> res(result.data(), result.size(), 1);

    Operon::Vector<T> params(nodes.size());
    Operon::Vector<gsl::span<const Operon::Scalar>> vals(nodes.size());
    size_t idx = 0;

    bool treeContainsNonlinearSymbols = false;
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i].IsConstant()) {
            auto v = parameters ? parameters[idx] : T(nodes[i].Value);
            m.col(i).setConstant(v);
            idx++;
        } else if (nodes[i].IsVariable()) {
            params[i] = parameters ? parameters[idx] : T(nodes[i].Value);
            vals[i] = dataset.GetValues(nodes[i].HashValue);
            idx++;
        }
        treeContainsNonlinearSymbols |= static_cast<bool>(nodes[i].Type & ~PrimitiveSet::Arithmetic);
    }

    auto lastCol = m.col(nodes.size() - 1);

    size_t numRows = range.Size();
    for (size_t row = 0; row < numRows; row += S) {
        auto remainingRows = std::min(S, numRows - row);

        for (size_t i = 0; i < nodes.size(); ++i) {
            typename decltype(m)::ColXpr r = m.col(i);
            auto const& s = nodes[i];

            if (GSL_LIKELY(s.IsLeaf())) {
                if (s.IsVariable()) {
                    Eigen::Map<const Eigen::Array<Operon::Scalar, Eigen::Dynamic, 1, Eigen::ColMajor>> seg(vals[i].data() + range.Start() + row, remainingRows);
                    r.segment(0, remainingRows) = params[i] * seg.cast<T>();
                }
            } else {
                switch (s.Type) {
                case NodeType::Add: {
                    dispatch_op<T, S, NodeType::Add>(m, nodes, i);
                    break;
                }
                case NodeType::Sub: {
                    dispatch_op<T, S, NodeType::Sub>(m, nodes, i);
                    break;
                }
                case NodeType::Mul: {
                    dispatch_op<T, S, NodeType::Mul>(m, nodes, i);
                    break;
                }
                case NodeType::Div: {
                    dispatch_op<T, S, NodeType::Div>(m, nodes, i);
                    break;
                }
                default: {
                    if (treeContainsNonlinearSymbols) {
                        switch (s.Type) {
                        case NodeType::Log: {
                            r = m.col(i - 1).log();
                            break;
                        }
                        case NodeType::Exp: {
                            r = m.col(i - 1).exp();
                            break;
                        }
                        case NodeType::Sin: {
                            r = m.col(i - 1).sin();
                            break;
                        }
                        case NodeType::Cos: {
                            r = m.col(i - 1).cos();
                            break;
                        }
                        case NodeType::Tan: {
                            r = m.col(i - 1).tan();
                            break;
                        }
                        case NodeType::Sqrt: {
                            r = m.col(i - 1).sqrt();
                            break;
                        }
                        case NodeType::Cbrt: {
                            r = m.col(i - 1).unaryExpr([](T v) { return T(ceres::cbrt(v)); });
                            break;
                        }
                        case NodeType::Square: {
                            r = m.col(i - 1).square();
                            break;
                        }
                        default: {
                            break;
                        }
                        }
                    }
                    break;
                }
                }
            }
        }
        // the final result is found in the last section of the buffer corresponding to the root node
        auto seg = lastCol.segment(0, remainingRows);
        auto max_ = Operon::Numeric::Max<T>();
#if EIGEN_MINOR_VERSION > 7
        res.segment(row, remainingRows) = (seg.isFinite()).select(seg, max_);
#else
        // less efficient
        res.segment(row, remainingRows) = seg.unaryExpr([&](auto v) { return ceres::IsFinite(v) ? v : max_; });
#endif
    }
}

struct TreeEvaluator {
    TreeEvaluator(Tree const& tree, Dataset const& dataset, const Range range)
        : tree_ref(tree)
        , dataset_ref(dataset)
        , range(range)
    {
    }

    template <typename T>
    bool operator()(T const* const* parameters, T* result) const
    {
        gsl::span<T> view(result, range.Size());
        Evaluate(tree_ref, dataset_ref, range, view, parameters[0]);
        return true;
    }

private:
    std::reference_wrapper<const Tree> tree_ref;
    std::reference_wrapper<const Dataset> dataset_ref;
    Range range;
};

struct ResidualEvaluator {
    ResidualEvaluator(const Tree& tree, const Dataset& dataset, const gsl::span<const Operon::Scalar> targetValues, const Range range)
        : treeEvaluator(tree, dataset, range)
        , target_ref(targetValues)
    {
    }

    template <typename T>
    bool operator()(T const* const* parameters, T* residuals) const
    {
        treeEvaluator(parameters, residuals);
        Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1, Eigen::ColMajor>> resMap(residuals, target_ref.size());
        Eigen::Map<const Eigen::Array<Operon::Scalar, Eigen::Dynamic, 1, Eigen::ColMajor>> targetMap(target_ref.data(), target_ref.size());
        resMap -= targetMap.cast<T>();
        return true;
    }

private:
    TreeEvaluator treeEvaluator;
    gsl::span<const Operon::Scalar> target_ref;
};
}
#endif
