// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_INTERPRETER_HPP
#define OPERON_INTERPRETER_HPP

#include <algorithm>
#include <gsl/pointers>
#include <optional>
#include <span>

#include "operon/core/dataset.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/formatter/formatter.hpp"
#include "derivatives.hpp"

// #include "tape.hpp"

namespace Operon {

enum class LikelihoodType : uint8_t { Gaussian, Poisson };

template<typename T>
struct InterpreterBase {
    InterpreterBase() = default;
    InterpreterBase(const InterpreterBase&) = default;
    InterpreterBase(InterpreterBase&&) = default;
    auto operator=(const InterpreterBase&) -> InterpreterBase& = default;
    auto operator=(InterpreterBase&&) -> InterpreterBase& = default;
    virtual ~InterpreterBase() = default;

    // evaluate model output
    virtual auto Evaluate(Operon::Span<T const> coeff, Operon::Range range, Operon::Span<T> result) const -> void = 0;
    virtual auto Evaluate(Operon::Span<T const> coeff, Operon::Range range) const -> Operon::Vector<T> = 0;

    // evaluate model jacobian in reverse mode
    virtual auto JacRev(Operon::Span<T const> coeff, Operon::Range range, Operon::Span<T> jacobian) const -> void = 0;
    virtual auto JacRev(Operon::Span<T const> coeff, Operon::Range range) const -> Eigen::Array<T, -1, -1> = 0;

    // evaluate model jacobian in forward mode
    virtual auto JacFwd(Operon::Span<T const> coeff, Operon::Range range, Operon::Span<T> jacobian) const -> void = 0;
    virtual auto JacFwd(Operon::Span<T const> coeff, Operon::Range range) const -> Eigen::Array<T, -1, -1> = 0;

    // getters
    [[nodiscard]] virtual auto GetTree() const -> Operon::Tree const* = 0;
    [[nodiscard]] virtual auto GetDataset() const -> Operon::Dataset const* = 0;
};

template<typename T = Operon::Scalar, typename DTable = DefaultDispatch>
requires DTable::template SupportsType<T>
struct Interpreter : public InterpreterBase<T> {
    using DispatchTable = DTable;
    static constexpr auto BatchSize = DTable::template BatchSize<T>;

    Interpreter(gsl::not_null<DTable const*> dtable, gsl::not_null<Operon::Dataset const*> dataset, gsl::not_null<Operon::Tree const*> tree)
        : dtable_(dtable)
        , dataset_(dataset)
        , tree_(tree) { }

    auto Primal() const { return primal_; }
    auto Trace() const { return trace_; }

    auto Evaluate(Operon::Span<T const> coeff, Operon::Range range, Operon::Span<T> result) const -> void final {
        InitContext(coeff, range);

        auto const len{ static_cast<int64_t>(range.Size()) };

        constexpr int64_t S{ BatchSize };
        auto* ptr = primal_.data() + ((primal_.extent(1) - 1) * S);

        for (auto row = 0L; row < len; row += S) {
            ForwardPass(range, row, /*trace=*/false);

            if (std::ssize(result) == len) {
                auto rem = std::min(S, len - row);
                std::ranges::copy(std::span(ptr, rem), result.data() + row);
            }
        }
    }

    auto Evaluate(Operon::Span<T const> coeff, Operon::Range range) const -> Operon::Vector<T> final {
        Operon::Vector<T> res(range.Size());
        this->Evaluate(coeff, range, {res.data(), res.size()});
        ENSURE(res.size() == range.Size());
        return res;
    }

    auto JacRev(Operon::Span<T const> coeff, Operon::Range range, Operon::Span<T> jacobian) const -> void final {
        InitContext(coeff, range);
        auto const len{ static_cast<int64_t>(range.Size()) };
        auto const& nodes = tree_->Nodes();
        auto const nn { std::ssize(nodes) };

        constexpr int64_t S{ BatchSize };
        trace_ = Backend::Buffer<T, S>(S, nn);
        Backend::Fill<T, S>(trace_, nn-1, T{1});

        Eigen::Map<Eigen::Array<T, -1, -1>> jac(jacobian.data(), len, coeff.size());

        for (auto row = 0L; row < len; row += S) {
            ForwardPass(range, row, /*trace=*/true);
            ReverseTrace(range, row, jac);
        }

        // Replace NaN values with zeros
        jac = jac.isNaN().select(T{0}, jac);
    }

    auto JacRev(Operon::Span<T const> coeff, Operon::Range range) const -> Eigen::Array<T, -1, -1> final {
        auto const nr{ static_cast<int64_t>(range.Size()) };
        Eigen::Array<T, -1, -1> jacobian(nr, coeff.size());
        JacRev(coeff, range, { jacobian.data(), static_cast<size_t>(jacobian.size()) });
        return jacobian;
    }

    auto JacFwd(Operon::Span<T const> coeff, Operon::Range range, Operon::Span<T> jacobian) const -> void final {
        InitContext(coeff, range);
        auto const& nodes = tree_->Nodes();
        auto const nNodes = std::ssize(nodes); 
        auto const nRows  = static_cast<int>(range.Size());

        trace_ = Backend::Buffer<T, BatchSize>(BatchSize, nNodes);
        Backend::Fill<T, BatchSize>(trace_, nNodes-1, T{1});

        Eigen::Map<Eigen::Array<T, -1, -1>> jac(jacobian.data(), nRows, coeff.size());

        for (int row = 0; row < nRows; row += BatchSize) {
            ForwardPass(range, row, /*trace=*/true);
            ForwardTrace(range, row, jac);
        }

        // Replace NaN values with zeros
        jac = jac.isNaN().select(T{0}, jac);
    }

    auto JacFwd(Operon::Span<T const> coeff, Operon::Range range) const -> Eigen::Array<T, -1, -1> final {
        auto const nRows = static_cast<int64_t>(range.Size());
        Eigen::Array<T, -1, -1> jacobian(nRows, coeff.size());
        JacFwd(coeff, range, { jacobian.data(), static_cast<size_t>(jacobian.size()) });
        return jacobian;
    }

    [[nodiscard]] auto GetTree() const -> Operon::Tree const* { return tree_.get(); }
    [[nodiscard]] auto GetDataset() const -> Operon::Dataset const* { return dataset_.get(); }

    auto GetDispatchTable() const { return dtable_.get(); }

    static auto Evaluate(Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Range const range) -> Operon::Vector<T> {
        auto coeff = tree.GetCoefficients();
        DTable dt;
        return Interpreter{&dt, &dataset, &tree}.Evaluate(coeff, range);
    }

    static auto Evaluate(Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Range const range, Operon::Span<T const> coeff) -> Operon::Vector<T> {
        DTable dt;
        return Interpreter{&dt, &dataset, &tree}.Evaluate(coeff, range);
    }

private:
    // private members
    using Data = std::tuple<T,
          std::span<T const>,
          std::optional<Dispatch::Callable<T, BatchSize> const>,
          std::optional<Dispatch::CallableDiff<T, BatchSize> const> >;

    gsl::not_null<DTable const*> dtable_;
    gsl::not_null<Operon::Dataset const*> dataset_;
    gsl::not_null<Operon::Tree const*> tree_;

    // mutable internal state (used by all the forward/reverse passes)
    mutable Operon::Vector<Data> context_;
    mutable Backend::Buffer<T, BatchSize> primal_;
    mutable Backend::Buffer<T, BatchSize> trace_;

    // private methods
    auto ForwardPass(Operon::Range range, int row, bool trace = false) const -> void {
        auto const& nodes     = tree_->Nodes();
        auto const nNodes     = std::ssize(nodes);
        auto const rangeStart = static_cast<int64_t>(range.Start());
        auto const rangeSize  = static_cast<int64_t>(range.Size());
        constexpr int64_t S   = BatchSize;

        auto rem = std::min(S, rangeSize - row);
        Operon::Range rg(rangeStart + row, rangeStart + row + rem);

        // forward pass - compute primal and trace
        for (auto i = 0L; i < nNodes; ++i) {
            if (nodes[i].IsConstant()) { continue; }

            auto const& [ p, v, f, df ] = context_[i];
            auto* ptr = primal_.data() + (i * S);

            if (nodes[i].IsVariable()) {
                std::ranges::transform(v.subspan(row, rem), ptr, [p](auto x) { return x * p; });
            } else {
                std::invoke(*f, nodes, primal_, i, rg);

                // first compute the partials
                if (trace && df) {
                    for (auto j : Tree::Indices(nodes, i)) {
                        std::invoke(*df, nodes, primal_, trace_, i, j);
                    }
                }

                // apply weight after partials are computed
                //if (p != T{1}) {
                //    std::ranges::transform(std::span(ptr, rem), ptr, [p](auto x) { return x * p; });
                //}
            }
        }
    }

    auto ForwardTrace(Operon::Range range, int row, Eigen::Ref<Eigen::Array<T, -1, -1>> jac) const -> void {
        auto const rangeSize     = static_cast<int64_t>(range.Size());
        auto const& nodes        = tree_->Nodes();
        auto const nNodes        = std::ssize(nodes);
        constexpr int64_t S      = BatchSize;
        auto const remainingRows = std::min(S, rangeSize - row);

        Eigen::Array<T, S, -1> dot(S, nNodes);
        Operon::Vector<int64_t> cidx(jac.cols());

        Eigen::Map<Eigen::Array<T, S, -1>> primal(primal_.data(), S, nNodes);
        Eigen::Map<Eigen::Array<T, S, -1>> trace(trace_.data(), S, nNodes);

        for (auto i = 0L, j = 0L; i < nNodes; ++i) {
            if (nodes[i].Optimize) { cidx[j++] = i; }
        }

        auto k{0};
        for (auto c : cidx) {
            dot.topRows(remainingRows).setConstant(T{0});
            dot.col(c).head(remainingRows).setConstant(T{1});

            for (auto i = 0; i < nNodes; ++i) {
                if (nodes[i].IsLeaf()) { continue; }
                for (auto x : Tree::Indices(nodes, i)) {
                    auto j{ static_cast<int64_t>(x) };
                    if (nodes[j].IsLeaf() && j != c) { continue; }
                    dot.col(i).head(remainingRows) += dot.col(j).head(remainingRows) * trace.col(j).head(remainingRows) * std::get<0>(context_[i]);
                }
            }

            jac.col(k++).segment(row, remainingRows) = dot.col(nNodes-1).head(remainingRows) * primal.col(c).head(remainingRows) / std::get<0>(context_[c]);
        }
    }

    auto ReverseTrace(Operon::Range range, int row, Eigen::Ref<Eigen::Array<T, -1, -1>> jac) const -> void {
        auto const rangeSize     = static_cast<int64_t>(range.Size());
        auto const& nodes        = tree_->Nodes();
        auto const nNodes        = std::ssize(nodes);
        constexpr int64_t S      = BatchSize;
        auto const remainingRows = std::min(S, rangeSize - row);

        auto k{jac.cols()};
        Eigen::Map<Eigen::Array<T, S, -1>> primal(primal_.data(), S, nNodes);
        Eigen::Map<Eigen::Array<T, S, -1>> trace(trace_.data(), S, nNodes);

        for (auto i = nNodes-1; i >= 0L; --i) {
            auto w = std::get<0>(context_[i]);

            if (nodes[i].Optimize) {
                jac.col(--k).segment(row, remainingRows) = trace.col(i).head(remainingRows) * primal.col(i).head(remainingRows) / w;
            }

            if (nodes[i].IsLeaf()) { continue; }

            for (auto j : Tree::Indices(nodes, i)) {
                auto const x { static_cast<int64_t>(j) };
                trace.col(x).head(remainingRows) *= trace.col(i).head(remainingRows) * w;
            }
        }
    }

    // init tree info into context_ and initializes primal_ columns
    auto InitContext(Operon::Span<T const> coeff, Operon::Range range) const {
        auto const& nodes = tree_->Nodes();
        auto const nRows  = static_cast<int64_t>(range.Size());
        auto const nNodes = std::ssize(nodes);

        constexpr int64_t S{ BatchSize };
        primal_ = Backend::Buffer<T, S>(S, nNodes);
        std::ranges::fill_n(primal_.data(), S * nNodes, T{0});

        context_.clear();
        context_.reserve(nNodes);

        auto const& dt = dtable_.get();
        // aggregate necessary info about the tree into a context object
        for (int64_t i = 0, j = 0; i < nNodes; ++i) {
            auto const& n = nodes[i];
            auto const* ptr      = n.IsVariable() ? dataset_->GetValues(n.HashValue).subspan(range.Start(), range.Size()).data() : nullptr;
            auto variableValues  = std::tuple_element_t<1, Data>(ptr, nRows);
            auto nodeCoefficient = (!coeff.empty() && n.Optimize) ? T{coeff[j++]} : T{n.Value};
            auto nodeFunction    = dt->template TryGetFunction<T>(n.HashValue);
            auto nodeDerivative  = dt->template TryGetDerivative<T>(n.HashValue);

            if (!n.IsLeaf() && !nodeFunction) {
                throw std::runtime_error(fmt::format("Missing primitive for node {}\n", n.Name()));
            }

            context_.emplace_back(nodeCoefficient, variableValues, nodeFunction, nodeDerivative);

            if (n.IsConstant()) {
                Backend::Fill<T, S>(primal_, i, T{nodeCoefficient});
            }
        }
    }
};

// convenience method to interpret many trees in parallel (mostly useful from the python wrapper)
auto OPERON_EXPORT EvaluateTrees(Operon::Vector<Operon::Tree> const& trees, Operon::Dataset const* dataset, Operon::Range range, size_t nthread = 0) -> Operon::Vector<Operon::Vector<Operon::Scalar>>;
auto OPERON_EXPORT EvaluateTrees(Operon::Vector<Operon::Tree> const& trees, Operon::Dataset const* dataset, Operon::Range range, std::span<Operon::Scalar> result, size_t nthread = 0) -> void;
} // namespace Operon
#endif
