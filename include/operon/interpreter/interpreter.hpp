// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_INTERPRETER_HPP
#define OPERON_INTERPRETER_HPP

#include <algorithm>
#include <optional>
#include <span>

#include "operon/core/dataset.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "dispatch_table.hpp"
// #include "tape.hpp"

namespace Operon {

namespace detail {
    // aligned allocation
    template<class T>
    struct Deleter {
        auto operator()(T* p) const -> void {
            std::free(p); // NOLINT
        }
    };

    template<class T>
    using AlignedUnique = std::unique_ptr<T[], Deleter<T>>;

    template<class ElementType, size_t ByteAlignment>
    auto AllocateAligned(auto const numElements) -> AlignedUnique<ElementType>
    {
        auto const numBytes = numElements * sizeof(ElementType);
        auto* ptr = std::aligned_alloc(ByteAlignment, numBytes);
        return AlignedUnique<ElementType>(static_cast<ElementType*>(ptr));
    }
}  // namespace detail

enum class LikelihoodType : int { Gaussian, Poisson };

template<typename T>
struct InterpreterBase {
    // evaluate model output
    virtual auto Evaluate(Operon::Span<T const> coeff, Operon::Range range, Operon::Span<T> result) const -> void = 0;
    virtual auto Evaluate(Operon::Span<T const> coeff, Operon::Range range) const -> std::vector<T> = 0;

    // evaluate model jacobian in reverse mode
    virtual auto JacRev(Operon::Span<T const> coeff, Operon::Range range, Operon::Span<T> jacobian) const -> void = 0;
    virtual auto JacRev(Operon::Span<T const> coeff, Operon::Range range) const -> Eigen::Array<T, -1, -1> = 0;

    // evaluate model jacobian in forward mode
    virtual auto JacFwd(Operon::Span<T const> coeff, Operon::Range range, Operon::Span<T> jacobian) const -> void = 0;
    virtual auto JacFwd(Operon::Span<T const> coeff, Operon::Range range) const -> Eigen::Array<T, -1, -1> = 0;

    // getters
    [[nodiscard]] virtual auto GetTree() const -> Operon::Tree const& = 0;
    [[nodiscard]] virtual auto GetDataset() const -> Operon::Dataset const& = 0;

    virtual ~InterpreterBase() = default;
};

template<typename T = Operon::Scalar, typename DTable = DefaultDispatch>
requires DTable::template SupportsType<T>
struct Interpreter : public InterpreterBase<T> {
    using DispatchTable = DTable;
    static constexpr auto BatchSize = DTable::template BatchSize<T>;

    Interpreter(DTable const& dtable, Operon::Dataset const& dataset, Operon::Tree const& tree)
        : dtable_(dtable)
        , dataset_(dataset)
        , tree_(tree) { }

    auto Primal() const { return primal_; }
    auto Trace() const { return trace_; }

    inline auto Evaluate(Operon::Span<T const> coeff, Operon::Range range, Operon::Span<T> result) const -> void final {
        InitContext(coeff, range);

        auto const len{ static_cast<int64_t>(range.Size()) };

        constexpr int64_t S{ BatchSize };
        auto* ptr = primal_.data_handle() + (primal_.extent(1) - 1) * S;

        for (auto row = 0L; row < len; row += S) {
            ForwardPass(range, row, /*trace=*/false);

            if (std::ssize(result) == len) {
                auto rem = std::min(S, len - row);
                std::ranges::copy(std::span(ptr, rem), result.data() + row);
            }
        }
    }

    inline auto Evaluate(Operon::Span<T const> coeff, Operon::Range range) const -> std::vector<T> final {
        std::vector<T> res(range.Size());
        Evaluate(coeff, range, {res.data(), res.size()});
        return res;
    }

    inline auto JacRev(Operon::Span<T const> coeff, Operon::Range range, Operon::Span<T> jacobian) const -> void final {
        InitContext(coeff, range);
        auto const len{ static_cast<int64_t>(range.Size()) };
        auto const& nodes = tree_.get().Nodes();
        auto const nn { std::ssize(nodes) };

        constexpr int64_t S{ BatchSize };
        traceStorage_ = detail::AllocateAligned<T, Backend::DefaultAlignment>(S * nn);
        trace_ = Backend::View<T, S>(traceStorage_.get(), S, nn);
        Fill<T, S>(trace_, nn-1, T{1});

        Eigen::Map<Eigen::Array<T, -1, -1>> jac(jacobian.data(), len, coeff.size());

        for (auto row = 0L; row < len; row += S) {
            ForwardPass(range, row, /*trace=*/true);
            ReverseTrace(range, row, jac);
        }
    }

    inline auto JacRev(Operon::Span<T const> coeff, Operon::Range range) const -> Eigen::Array<T, -1, -1> final {
        auto const nr{ static_cast<int64_t>(range.Size()) };
        Eigen::Array<T, -1, -1> jacobian(nr, coeff.size());
        JacRev(coeff, range, { jacobian.data(), static_cast<size_t>(jacobian.size()) });
        return jacobian;
    }

    auto JacFwd(Operon::Span<T const> coeff, Operon::Range range, Operon::Span<T> jacobian) const -> void final {
        InitContext(coeff, range);
        auto const len{ static_cast<int>(range.Size()) };
        auto const& nodes = tree_.get().Nodes();
        auto const nn   { std::ssize(nodes) };

        constexpr int64_t S{ BatchSize };
        traceStorage_ = detail::AllocateAligned<T, Backend::DefaultAlignment>(S * nn);
        trace_ = Backend::View<T, S>(traceStorage_.get(), S, nn);
        Fill<T, S>(trace_, nn-1, T{1});

        Eigen::Map<Eigen::Array<T, -1, -1>> jac(jacobian.data(), len, coeff.size());

        for (int row = 0; row < len; row += S) {
            ForwardPass(range, row, /*trace=*/true);
            ForwardTrace(range, row, jac);
        }
    }

    inline auto JacFwd(Operon::Span<T const> coeff, Operon::Range range) const -> Eigen::Array<T, -1, -1> final {
        auto const nr{ static_cast<int64_t>(range.Size()) };
        Eigen::Array<T, -1, -1> jacobian(nr, coeff.size());
        JacFwd(coeff, range, { jacobian.data(), static_cast<size_t>(jacobian.size()) });
        return jacobian;
    }

    [[nodiscard]] auto GetTree() const -> Operon::Tree const& final { return tree_.get(); }
    [[nodiscard]] auto GetDataset() const -> Operon::Dataset const& final { return dataset_.get(); }

    auto GetDispatchTable() const { return dtable_.get(); }

    static inline auto Evaluate(Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Range const range) {
        auto coeff = tree.GetCoefficients();
        DTable dt;
        return Interpreter{dt, dataset, tree}.Evaluate(coeff, range);
    }

    static inline auto Evaluate(Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Range const range, Operon::Span<T const> coeff) {
        DTable dt;
        return Interpreter{dt, dataset, tree}.Evaluate(coeff, range);
    }

private:
    // private members
    using Data = std::tuple<T,
          std::span<Operon::Scalar const>,
          std::optional<Dispatch::Callable<T, BatchSize> const>,
          std::optional<Dispatch::CallableDiff<T, BatchSize> const> >;

    std::reference_wrapper<DTable const> dtable_;
    std::reference_wrapper<Operon::Dataset const> dataset_;
    std::reference_wrapper<Operon::Tree const> tree_;

    // mutable internal state (used by all the forward/reverse passes)
    mutable std::vector<Data> context_;

    mutable Backend::View<T, BatchSize> primal_;
    mutable Backend::View<T, BatchSize> trace_;

    mutable detail::AlignedUnique<T> primalStorage_;
    mutable detail::AlignedUnique<T> traceStorage_;

    // private methods
    inline auto ForwardPass(Operon::Range range, int row, bool trace = false) const -> void {
        auto const start { static_cast<int64_t>(range.Start()) };
        auto const len   { static_cast<int64_t>(range.Size()) };
        auto const& nodes = tree_.get().Nodes();
        auto const nn = std::ssize(nodes);
        constexpr int64_t S{ BatchSize };

        auto rem = std::min(S, len - row);
        Operon::Range rg(start + row, start + row + rem);

        // forward pass - compute primal and trace
        for (auto i = 0L; i < nn; ++i) {
            auto const& [ p, v, f, df ] = context_[i];
            auto* ptr = primal_.data_handle() + i * S;

            if (nodes[i].IsVariable()) {
                std::ranges::transform(v.subspan(row, rem), ptr, [p](auto x) { return x * p; });
            } else if (f) {
                std::invoke(*f, nodes, primal_, i, rg);

                // first compute the partials
                if (trace && df) {
                    for (auto j : Tree::Indices(nodes, i)) {
                        std::invoke(*df, nodes, primal_, trace_, i, j);
                    }
                }

                // apply weight after partials are computed
                if (p != T{1}) {
                    std::ranges::transform(std::span(ptr, rem), ptr, [p](auto x) { return x * p; });
                }
            }
        }
    }

    inline auto ForwardTrace(Operon::Range range, int row, Eigen::Ref<Eigen::Array<T, -1, -1>> jac) const -> void {
        auto const len   { static_cast<int64_t>(range.Size()) };
        auto const& nodes{ tree_.get().Nodes() };
        auto const nn { std::ssize(nodes) };
        constexpr int64_t S{ BatchSize };
        auto const rem   { std::min(S, len - row) };

        Eigen::Array<T, S, -1> dot(S, nn);
        std::vector<int64_t> cidx(jac.cols());

        Eigen::Map<Eigen::Array<T, S, -1>> primal(primal_.data_handle(), S, nn);
        Eigen::Map<Eigen::Array<T, S, -1>> trace(trace_.data_handle(), S, nn);

        for (auto i = 0L, j = 0L; i < nn; ++i) {
            if (nodes[i].Optimize) { cidx[j++] = i; }
        }

        auto k{0};
        for (auto c : cidx) {
            dot.topRows(rem).setConstant(T{0});
            dot.col(c).head(rem).setConstant(T{1});

            for (auto i = 0; i < nn; ++i) {
                if (nodes[i].IsLeaf()) { continue; }
                for (auto x : Tree::Indices(nodes, i)) {
                    auto j{ static_cast<int64_t>(x) };
                    if (nodes[j].IsLeaf() && j != c) { continue; }
                    dot.col(i).head(rem) += dot.col(j).head(rem) * trace.col(j).head(rem) * std::get<0>(context_[i]);
                }
            }

            jac.col(k++).segment(row, rem) = dot.col(nn-1).head(rem) * primal.col(c).head(rem) / std::get<0>(context_[c]);
        }
    }

    auto ReverseTrace(Operon::Range range, int row, Eigen::Ref<Eigen::Array<T, -1, -1>> jac) const -> void {
        auto const len   { static_cast<int64_t>(range.Size()) };
        auto const& nodes{ tree_.get().Nodes() };
        auto const nn    { std::ssize(nodes) };
        constexpr int64_t S{ BatchSize };
        auto const rem   { std::min(S, len - row) };

        auto k{jac.cols()};
        Eigen::Map<Eigen::Array<T, S, -1>> primal(primal_.data_handle(), S, nn);
        Eigen::Map<Eigen::Array<T, S, -1>> trace(trace_.data_handle(), S, nn);

        for (auto i = nn-1; i >= 0L; --i) {
            auto w = std::get<0>(context_[i]);

            if (nodes[i].Optimize) {
                jac.col(--k).segment(row, rem) = trace.col(i).head(rem) * primal.col(i).head(rem) / w;
            }

            if (nodes[i].IsLeaf()) { continue; }

            for (auto j : Tree::Indices(nodes, i)) {
                auto const x { static_cast<int64_t>(j) };
                trace.col(x).head(rem) *= trace.col(i).head(rem) * w;
            }
        }
    }

    // init tree info into context_ and initializes primal_ columns
    auto InitContext(Operon::Span<T const> coeff, Operon::Range range) const {
        auto const& nodes{ tree_.get().Nodes() };
        auto const nr { static_cast<int64_t>(range.Size()) };
        auto const nn { std::ssize(nodes) };

        constexpr int64_t S{ BatchSize };
        primalStorage_ = detail::AllocateAligned<T, Backend::DefaultAlignment>(S * nn);
        std::ranges::fill_n(primalStorage_.get(), S * nn, T{0});
        primal_ = Backend::View<T, BatchSize>(primalStorage_.get(), S, nn);

        context_.clear();
        context_.reserve(nn);

        auto const& dt = dtable_.get();
        // aggregate necessary info about the tree into a context object
        for (int64_t i = 0, j = 0; i < nn; ++i) {
            auto const& n = nodes[i];
            auto const* ptr      = n.IsVariable() ? dataset_.get().GetValues(n.HashValue).subspan(range.Start(), range.Size()).data() : nullptr;
            auto variableValues  = std::tuple_element_t<1, Data>(ptr, nr);
            auto nodeCoefficient = (!coeff.empty() && n.Optimize) ? T{coeff[j++]} : T{n.Value};
            auto nodeFunction    = dt.template TryGetFunction<T>(n.HashValue);
            auto nodeDerivative  = dt.template TryGetDerivative<T>(n.HashValue);

            if (!n.IsLeaf() && !nodeFunction) {
                throw std::runtime_error(fmt::format("Missing primitive for node {}\n", n.Name()));
            }

            context_.push_back({ nodeCoefficient, variableValues, nodeFunction, nodeDerivative });

            if (n.IsConstant()) {
                Fill<T, S>(primal_, i, T{nodeCoefficient});
            }
        }
    }
};

// convenience method to interpret many trees in parallel (mostly useful from the python wrapper)
auto OPERON_EXPORT EvaluateTrees(std::vector<Operon::Tree> const& trees, Operon::Dataset const& dataset, Operon::Range range, size_t nthread = 0) -> std::vector<std::vector<Operon::Scalar>>;
auto OPERON_EXPORT EvaluateTrees(std::vector<Operon::Tree> const& trees, Operon::Dataset const& dataset, Operon::Range range, std::span<Operon::Scalar> result, size_t nthread = 0) -> void;
} // namespace Operon
#endif
