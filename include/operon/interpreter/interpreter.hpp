// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

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

template<typename T = Operon::Scalar, typename DTable = ScalarDispatch>
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

        std::size_t j = 0;
        auto const cols = BuildColumns([&](std::size_t i) -> std::size_t { return nodes[i].Optimize ? j++ : NoColumn; });

        Eigen::Map<Eigen::Array<T, -1, -1>> jac(jacobian.data(), len, coeff.size());
        // No zero-init needed: each Optimize node maps to a unique column
        // (built via BuildColumns above), so every column is written
        // exactly once (Accumulate=false, plain `=`).

        for (auto row = 0L; row < len; row += S) {
            ForwardPass(range, row, /*trace=*/true);
            ReverseTraceGeneric<false>(range, row, cols.colOf,
                [](std::size_t i, auto const& primal, T w) { return primal.col(static_cast<Eigen::Index>(i)) / w; },
                jac);
        }
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

        std::size_t j = 0;
        auto const cols = BuildColumns([&](std::size_t i) -> std::size_t { return nodes[i].Optimize ? j++ : NoColumn; });

        Eigen::Map<Eigen::Array<T, -1, -1>> jac(jacobian.data(), nRows, coeff.size());
        // No zero-init needed — see JacRev.

        for (int row = 0; row < nRows; row += BatchSize) {
            ForwardPass(range, row, /*trace=*/true);
            ForwardTraceGeneric<false>(range, row, cols.seeds,
                [](std::size_t i, auto const& primal, T w) { return primal.col(static_cast<Eigen::Index>(i)) / w; },
                jac);
        }
    }

    auto JacFwd(Operon::Span<T const> coeff, Operon::Range range) const -> Eigen::Array<T, -1, -1> final {
        auto const nRows = static_cast<int64_t>(range.Size());
        Eigen::Array<T, -1, -1> jacobian(nRows, coeff.size());
        JacFwd(coeff, range, { jacobian.data(), static_cast<size_t>(jacobian.size()) });
        return jacobian;
    }

    // Reverse-mode derivative of the tree output w.r.t. the raw value of
    // one input Variable (identified by hash), summed over every
    // occurrence of that variable in the tree via the multivariate chain
    // rule. Complements JacRev, which differentiates w.r.t. Node::Optimize
    // coefficients (weights) rather than variable values — same underlying
    // adjoint sweep (ReverseTraceGeneric), different extraction point.
    auto JacRevVariable(Operon::Span<T const> coeff, Operon::Range range, Operon::Hash variable, Operon::Span<T> result) const -> void {
        InitContext(coeff, range);
        auto const len{ static_cast<int64_t>(range.Size()) };
        auto const& nodes = tree_->Nodes();
        auto const nn { std::ssize(nodes) };

        constexpr int64_t S{ BatchSize };
        trace_ = Backend::Buffer<T, S>(S, nn);
        Backend::Fill<T, S>(trace_, nn-1, T{1});

        auto const cols = BuildColumns([&](std::size_t i) -> std::size_t { return (nodes[i].IsVariable() && nodes[i].HashValue == variable) ? 0 : NoColumn; });

        Eigen::Map<Eigen::Array<T, -1, -1>> jac(result.data(), len, 1);
        jac.setZero(); // needed: multiple occurrences of `variable` can share column 0 (Accumulate=true) — cheap, single-column buffer

        for (auto row = 0L; row < len; row += S) {
            ForwardPass(range, row, /*trace=*/true);
            ReverseTraceGeneric<true>(range, row, cols.colOf,
                [](std::size_t /*i*/, auto const& /*primal*/, T w) { return Eigen::Array<T, S, 1>::Constant(w); },
                jac);
        }
    }

    auto JacRevVariable(Operon::Span<T const> coeff, Operon::Range range, Operon::Hash variable) const -> Operon::Vector<T> {
        Operon::Vector<T> result(range.Size());
        JacRevVariable(coeff, range, variable, { result.data(), result.size() });
        return result;
    }

    // Forward-mode counterpart to JacRevVariable — same relationship as
    // JacFwd has to JacRev.
    auto JacFwdVariable(Operon::Span<T const> coeff, Operon::Range range, Operon::Hash variable, Operon::Span<T> result) const -> void {
        InitContext(coeff, range);
        auto const& nodes = tree_->Nodes();
        auto const nNodes = std::ssize(nodes);
        auto const nRows  = static_cast<int>(range.Size());

        trace_ = Backend::Buffer<T, BatchSize>(BatchSize, nNodes);
        Backend::Fill<T, BatchSize>(trace_, nNodes-1, T{1});

        auto const cols = BuildColumns([&](std::size_t i) -> std::size_t { return (nodes[i].IsVariable() && nodes[i].HashValue == variable) ? 0 : NoColumn; });

        Eigen::Map<Eigen::Array<T, -1, -1>> jac(result.data(), nRows, 1);
        jac.setZero(); // needed — see JacRevVariable

        for (int row = 0; row < nRows; row += BatchSize) {
            ForwardPass(range, row, /*trace=*/true);
            ForwardTraceGeneric<true>(range, row, cols.seeds,
                [](std::size_t /*i*/, auto const& /*primal*/, T w) { return Eigen::Array<T, BatchSize, 1>::Constant(w); },
                jac);
        }
    }

    auto JacFwdVariable(Operon::Span<T const> coeff, Operon::Range range, Operon::Hash variable) const -> Operon::Vector<T> {
        Operon::Vector<T> result(range.Size());
        JacFwdVariable(coeff, range, variable, { result.data(), result.size() });
        return result;
    }

    // Evaluate the full tree and extract values at multiple node indices.
    // Roots set to SIZE_MAX produce zero columns.
    auto EvaluateRoots(Operon::Span<T const> coeff, Operon::Range range,
                       Operon::Span<std::size_t const> roots) const -> Eigen::Array<T, -1, -1>
    {
        InitContext(coeff, range);

        auto const len    = static_cast<int64_t>(range.Size());
        auto const nRoots = static_cast<Eigen::Index>(roots.size());
        constexpr int64_t S = BatchSize;
        constexpr auto NoRoot = std::numeric_limits<std::size_t>::max();

        // Not zero-initialized: every row is written exactly once by the
        // batch loop below; NoRoot columns are zeroed explicitly.
        Eigen::Array<T, -1, -1> result(len, nRoots);
        for (Eigen::Index k = 0; k < nRoots; ++k) {
            if (roots[k] == NoRoot) { result.col(k).setZero(); }
        }

        auto const nNodes = static_cast<std::size_t>(tree_->Nodes().size());
        for (Eigen::Index k = 0; k < nRoots; ++k) {
            EXPECT(roots[k] == NoRoot || roots[k] < nNodes);
        }

        for (auto row = 0L; row < len; row += S) {
            ForwardPass(range, row, /*trace=*/false);
            auto const rem = std::min(S, len - row);
            for (Eigen::Index k = 0; k < nRoots; ++k) {
                if (roots[k] == NoRoot) { continue; }
                auto const* src = primal_.data() + (static_cast<int64_t>(roots[k]) * S);
                std::copy_n(src, rem, result.col(k).data() + row);
            }
        }
        return result;
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
    mutable Operon::Range range_{};

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

            if (nodes[i].IsRef()) {
                EXPECT(static_cast<int64_t>(nodes[i].RefTo) < i); // backward reference invariant
                auto const* src = primal_.data() + (static_cast<int64_t>(nodes[i].RefTo) * S);
                std::copy_n(src, S, ptr);
            } else if (nodes[i].IsVariable()) {
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

    // Sentinel meaning "this node doesn't contribute to any output column",
    // used by BuildColumns/*TraceGeneric below.
    static constexpr std::size_t NoColumn = std::numeric_limits<std::size_t>::max();

    // Built once per JacRev/JacFwd/JacRevVariable/JacFwdVariable call (not
    // per row-batch — it only depends on tree structure, not on which rows
    // are being processed): `colOf[i]` is the output column node i
    // contributes to, or NoColumn — used by ReverseTraceGeneric, which
    // visits every node anyway so an O(1) lookup is free. `seeds` is the
    // compact list of (node, column) pairs with colOf[i] != NoColumn — used
    // by ForwardTraceGeneric, which should only iterate actual targets
    // instead of scanning every node in the tree.
    struct Columns {
        Operon::Vector<std::size_t> colOf;
        Operon::Vector<std::pair<std::size_t, std::size_t>> seeds;
    };

    template <typename Predicate>
    auto BuildColumns(Predicate predicate) const -> Columns {
        auto const nNodes = static_cast<std::size_t>(tree_->Nodes().size());
        Columns cols;
        cols.colOf.assign(nNodes, NoColumn);
        for (std::size_t i = 0; i < nNodes; ++i) {
            if (auto const col = predicate(i); col != NoColumn) {
                cols.colOf[i] = col;
                cols.seeds.emplace_back(i, col);
            }
        }
        return cols;
    }

    // Shared forward-mode sweep behind JacFwd/JacFwdVariable. `seeds`: the
    // (node, column) pairs to seed-and-propagate (see BuildColumns).
    // `factor(i, primal, w)`: the local d(primal_i)/d(target) multiplier
    // converting the node's root-adjoint into the derivative w.r.t.
    // whatever `target` is for this call (a coefficient's weight, or a
    // variable's raw value) — e.g. primal_i/w for a coefficient target
    // (primal_i = w * x_i), or w for a variable target. `Accumulate`:
    // false writes `=` (the coefficient case — each column has exactly one
    // contributing node, so no zero-init is needed by the caller); true
    // writes `+=` (the variable case, where multiple nodes/occurrences can
    // share one column — caller must zero-init `jac` first).
    template <bool Accumulate, typename LocalFactor>
    auto ForwardTraceGeneric(Operon::Range range, int row, Operon::Vector<std::pair<std::size_t, std::size_t>> const& seeds, LocalFactor factor, Eigen::Ref<Eigen::Array<T, -1, -1>> jac) const -> void {
        auto const rangeSize     = static_cast<int64_t>(range.Size());
        auto const& nodes        = tree_->Nodes();
        auto const nNodes        = std::ssize(nodes);
        constexpr int64_t S      = BatchSize;
        auto const remainingRows = std::min(S, rangeSize - row);

        Eigen::Array<T, S, -1> dot(S, nNodes);

        Eigen::Map<Eigen::Array<T, S, -1>> primal(primal_.data(), S, nNodes);
        Eigen::Map<Eigen::Array<T, S, -1>> trace(trace_.data(), S, nNodes);

        for (auto const& [c, col] : seeds) {
            auto const cc = static_cast<int64_t>(c);

            dot.topRows(remainingRows).setConstant(T{0});
            dot.col(cc).head(remainingRows).setConstant(T{1});

            for (auto i = 0; i < nNodes; ++i) {
                if (nodes[i].IsRef()) {
                    EXPECT(static_cast<int64_t>(nodes[i].RefTo) < i); // backward reference invariant
                    dot.col(i).head(remainingRows) = dot.col(static_cast<int64_t>(nodes[i].RefTo)).head(remainingRows);
                    continue;
                }
                if (nodes[i].IsLeaf()) { continue; }
                for (auto x : Tree::Indices(nodes, i)) {
                    auto j{ static_cast<int64_t>(x) };
                    if (nodes[j].IsLeaf() && j != cc) { continue; }
                    dot.col(i).head(remainingRows) += dot.col(j).head(remainingRows) * trace.col(j).head(remainingRows) * std::get<0>(context_[i]);
                }
            }

            auto const w = std::get<0>(context_[c]);
            auto const contribution = dot.col(nNodes-1).head(remainingRows) * factor(c, primal, w).head(remainingRows);
            if constexpr (Accumulate) {
                jac.col(static_cast<Eigen::Index>(col)).segment(row, remainingRows) += contribution;
            } else {
                jac.col(static_cast<Eigen::Index>(col)).segment(row, remainingRows) = contribution;
            }
        }
    }

    // Shared reverse-mode sweep behind JacRev/JacRevVariable — see
    // ForwardTraceGeneric above for the `factor`/`Accumulate` contract.
    // `colOf`: per-node column lookup (see BuildColumns). One backward pass
    // computes every node's root-adjoint regardless of how many columns are
    // extracted, unlike the forward-mode sweep above (one seeded pass per
    // column) — this is why reverse mode is preferred when there are many
    // targets (JacRev vs JacFwd for coefficients).
    template <bool Accumulate, typename LocalFactor>
    auto ReverseTraceGeneric(Operon::Range range, int row, Operon::Vector<std::size_t> const& colOf, LocalFactor factor, Eigen::Ref<Eigen::Array<T, -1, -1>> jac) const -> void {
        auto const rangeSize     = static_cast<int64_t>(range.Size());
        auto const& nodes        = tree_->Nodes();
        auto const nNodes        = std::ssize(nodes);
        constexpr int64_t S      = BatchSize;
        auto const remainingRows = std::min(S, rangeSize - row);

        Eigen::Map<Eigen::Array<T, S, -1>> primal(primal_.data(), S, nNodes);
        Eigen::Map<Eigen::Array<T, S, -1>> trace(trace_.data(), S, nNodes);

        for (auto i = nNodes-1; i >= 0L; --i) {
            auto w = std::get<0>(context_[i]);

            if (auto const col = colOf[static_cast<std::size_t>(i)]; col != NoColumn) {
                auto const contribution = trace.col(i).head(remainingRows) * factor(static_cast<std::size_t>(i), primal, w).head(remainingRows);
                if constexpr (Accumulate) {
                    jac.col(static_cast<Eigen::Index>(col)).segment(row, remainingRows) += contribution;
                } else {
                    jac.col(static_cast<Eigen::Index>(col)).segment(row, remainingRows) = contribution;
                }
            }

            if (nodes[i].IsRef()) {
                EXPECT(static_cast<int64_t>(nodes[i].RefTo) < i); // backward ref: target processed after us in reverse sweep
                // Accumulate gradient into the referenced node (may be referenced >1 time)
                trace.col(static_cast<int64_t>(nodes[i].RefTo)).head(remainingRows) +=
                    trace.col(i).head(remainingRows);
                continue;
            }
            if (nodes[i].IsLeaf()) { continue; }

            for (auto j : Tree::Indices(nodes, i)) {
                auto const x { static_cast<int64_t>(j) };
                trace.col(x).head(remainingRows) *= trace.col(i).head(remainingRows) * w;
            }
        }
    }


    // Full bind: allocate primal_, build context_ with function/derivative pointers
    // and variable data spans. Called once per unique (tree, range) pair.
    auto BindTree(Operon::Range range) const {
        auto const& nodes = tree_->Nodes();
        auto const nRows  = static_cast<int64_t>(range.Size());
        auto const nNodes = std::ssize(nodes);

        constexpr int64_t S{ BatchSize };
        primal_ = Backend::Buffer<T, S>(S, nNodes);
        std::ranges::fill_n(primal_.data(), S * nNodes, T{0});

        context_.clear();
        context_.reserve(nNodes);

        auto const& dt = dtable_.get();
        for (int64_t i = 0; i < nNodes; ++i) {
            auto const& n = nodes[i];
            auto variableValues = n.IsVariable()
                ? std::tuple_element_t<1, Data>(dataset_->GetValues(n.HashValue).subspan(range.Start(), range.Size()).data(), nRows)
                : std::tuple_element_t<1, Data>{};
            auto nodeFunction   = dt->template TryGetFunction<T>(n.HashValue);
            auto nodeDerivative = dt->template TryGetDerivative<T>(n.HashValue);

            if (!n.IsLeaf() && !nodeFunction) {
                throw std::runtime_error(fmt::format("Missing primitive for node {}\n", n.Name()));
            }

            context_.emplace_back(T{n.Value}, variableValues, nodeFunction, nodeDerivative);
        }
        range_ = range;
    }

    // Cheap update: patch coefficient values in context_ and re-fill constant columns.
    // Called on every optimizer step once BindTree has been called for this range.
    auto UpdateCoefficients(Operon::Span<T const> coeff) const {
        auto const& nodes = tree_->Nodes();
        auto const nNodes = std::ssize(nodes);
        constexpr int64_t S{ BatchSize };

        for (int64_t i = 0, j = 0; i < nNodes; ++i) {
            auto const& n = nodes[i];
            if (!coeff.empty() && n.Optimize) {
                std::get<0>(context_[i]) = T{coeff[j++]};
            }
            if (n.IsConstant()) {
                Backend::Fill<T, S>(primal_, i, std::get<0>(context_[i]));
            }
        }
    }

    auto InitContext(Operon::Span<T const> coeff, Operon::Range range) const {
        if (context_.empty() || range_ != range) { BindTree(range); }
        UpdateCoefficients(coeff);
    }
};

// convenience method to interpret many trees in parallel (mostly useful from the python wrapper)
auto OPERON_EXPORT EvaluateTrees(Operon::Vector<Operon::Tree> const& trees, Operon::Dataset const* dataset, Operon::Range range, size_t nthread = 0) -> Operon::Vector<Operon::Vector<Operon::Scalar>>;
auto OPERON_EXPORT EvaluateTrees(Operon::Vector<Operon::Tree> const& trees, Operon::Dataset const* dataset, Operon::Range range, std::span<Operon::Scalar> result, size_t nthread = 0) -> void;
} // namespace Operon
#endif
