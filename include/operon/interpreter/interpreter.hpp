#ifndef OPERON_INTERPRETER_HPP
#define OPERON_INTERPRETER_HPP

#include <algorithm>

#include "core/dataset.hpp"
#include "core/tree.hpp"
#include "dispatch_table.hpp"

namespace Operon {
struct Interpreter {
    Interpreter(DispatchTable const& ft)
        : ftable(ft)
    {
    }

    Interpreter() { }

    // evaluate a tree and return a vector of values
    template <typename T>
    Operon::Vector<T> Evaluate(Tree const& tree, Dataset const& dataset, Range const range, T const* const parameters = nullptr) const noexcept
    {
        Operon::Vector<T> result(range.Size());
        Evaluate<T>(tree, dataset, range, gsl::span<T>(result), parameters);
        return result;
    }

    template <typename T>
    Operon::Vector<T> Evaluate(Tree const& tree, Dataset const& dataset, Range const range, size_t const batchSize, T const* const parameters = nullptr) const noexcept
    {
        Operon::Vector<T> result(range.Size());
        gsl::span<T> view(result);

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
    void Evaluate(Tree const& tree, Dataset const& dataset, Range const range, gsl::span<T> result, T const* const parameters = nullptr) const noexcept
    {
        const auto& nodes = tree.Nodes();
        EXPECT(nodes.size() > 0);

        constexpr int S = static_cast<int>(detail::BatchSize<T>());

        Eigen::Array<T, S, Eigen::Dynamic, Eigen::ColMajor> m(S, nodes.size());
        Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1, Eigen::ColMajor>> res(result.data(), result.size(), 1);

        Operon::Vector<T> params(nodes.size());
        Operon::Vector<gsl::span<const Operon::Scalar>> vals(nodes.size());
        size_t idx = 0;

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
        }

        auto lastCol = m.col(nodes.size() - 1);

        int numRows = static_cast<int>(range.Size());
        for (int row = 0; row < numRows; row += S) {
            auto remainingRows = std::min(S, numRows - row);

            for (size_t i = 0; i < nodes.size(); ++i) {
                auto const& s = nodes[i];

                if (s.IsConstant()) {
                    continue; // constants already filled above
                } else if (s.IsVariable()) {
                    Eigen::Map<const Eigen::Array<Operon::Scalar, Eigen::Dynamic, 1, Eigen::ColMajor>> seg(vals[i].data() + range.Start() + row, remainingRows);
                    m.col(i).segment(0, remainingRows) = params[i] * seg.cast<T>();
                } else {
                    ftable.Get<T>(nodes[i].Type)(m, nodes, i, range.Start() + row);
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

    template<typename T>
    static void Evaluate(DispatchTable& ft, Tree const& tree, Dataset const& dataset, Range const range, gsl::span<T> result, T const* const parameters = nullptr) noexcept {
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
