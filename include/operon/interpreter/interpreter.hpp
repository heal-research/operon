// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_INTERPRETER_HPP
#define OPERON_INTERPRETER_HPP

#include <Eigen/Core>
#include <algorithm>
#include <gsl/pointers>
#include <optional>
#include <span>

#include "derivatives.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/core/interpreter_error.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/operon_export.hpp"
#include <string>
#include <tl/expected.hpp>

// #include "tape.hpp"

namespace Operon {

enum class LikelihoodType : uint8_t { Gaussian, Poisson };

struct TreeEvaluationError {
    std::size_t Index {};
    InterpreterError Error;
};

template <typename T> struct InterpreterBase {
    InterpreterBase() = default;
    InterpreterBase(const InterpreterBase&) = default;
    InterpreterBase(InterpreterBase&&) = default;
    auto operator=(const InterpreterBase&) -> InterpreterBase& = default;
    auto operator=(InterpreterBase&&) -> InterpreterBase& = default;
    virtual ~InterpreterBase() = default;

    // evaluate model output
    [[nodiscard]] virtual auto Evaluate(Operon::Span<T const> coeff, Operon::Range range, Operon::Span<T> result) const
        -> tl::expected<void, InterpreterError>
        = 0;
    [[nodiscard]] virtual auto Evaluate(Operon::Span<T const> coeff, Operon::Range range) const
        -> tl::expected<Operon::Vector<T>, InterpreterError>
        = 0;

    // evaluate model jacobian in reverse mode
    [[nodiscard]] virtual auto JacRev(Operon::Span<T const> coeff, Operon::Range range, Operon::Span<T> jacobian) const
        -> tl::expected<void, InterpreterError>
        = 0;
    [[nodiscard]] virtual auto JacRev(Operon::Span<T const> coeff, Operon::Range range) const
        -> tl::expected<Eigen::Array<T, -1, -1>, InterpreterError>
        = 0;

    // evaluate model jacobian in forward mode
    [[nodiscard]] virtual auto JacFwd(Operon::Span<T const> coeff, Operon::Range range, Operon::Span<T> jacobian) const
        -> tl::expected<void, InterpreterError>
        = 0;
    [[nodiscard]] virtual auto JacFwd(Operon::Span<T const> coeff, Operon::Range range) const
        -> tl::expected<Eigen::Array<T, -1, -1>, InterpreterError>
        = 0;

    // evaluate model derivative w.r.t. a single input variable's raw value (identified by hash), summed over every
    // occurrence of that variable in the tree — reverse mode
    [[nodiscard]] virtual auto JacRevVariable(Operon::Span<T const> coeff, Operon::Range range, Operon::Hash variable,
        Operon::Span<T> result) const -> tl::expected<void, InterpreterError>
        = 0;
    [[nodiscard]] virtual auto JacRevVariable(Operon::Span<T const> coeff, Operon::Range range,
        Operon::Hash variable) const -> tl::expected<Operon::Vector<T>, InterpreterError>
        = 0;

    // same as above, forward mode
    [[nodiscard]] virtual auto JacFwdVariable(Operon::Span<T const> coeff, Operon::Range range, Operon::Hash variable,
        Operon::Span<T> result) const -> tl::expected<void, InterpreterError>
        = 0;
    [[nodiscard]] virtual auto JacFwdVariable(Operon::Span<T const> coeff, Operon::Range range,
        Operon::Hash variable) const -> tl::expected<Operon::Vector<T>, InterpreterError>
        = 0;

    // getters
    [[nodiscard]] virtual auto GetTree() const -> Operon::Tree const* = 0;
    [[nodiscard]] virtual auto GetDataset() const -> Operon::Dataset const* = 0;
};

// Not thread-safe: binds lazily into mutable scratch state reused across calls. Use one instance per worker thread.
template <typename T = Operon::Scalar, typename DTable = ScalarDispatch>
    requires(DTable::template SupportsType<T>)
struct Interpreter : public InterpreterBase<T> {
    using DispatchTable = DTable;
    static constexpr auto BatchSize = DTable::template BatchSize<T>;

    Interpreter(gsl::not_null<DTable const*> dtable, gsl::not_null<Operon::Dataset const*> dataset,
        gsl::not_null<Operon::Tree const*> tree)
        : dtable_(dtable)
        , dataset_(dataset)
        , tree_(tree)
    {
    }

    [[nodiscard]] auto Evaluate(Operon::Span<T const> coeff, Operon::Range range, Operon::Span<T> result) const
        -> tl::expected<void, InterpreterError> final
    {
        if (auto prepared = Prepare(coeff, range, /*needDerivatives=*/false); !prepared) {
            return prepared;
        }
        if (auto sized = CheckOutputSize(range.Size(), result.size()); !sized) {
            return sized;
        }
        UpdateCoefficients(coeff);
        auto const len = static_cast<int64_t>(range.Size());
        constexpr int64_t S = BatchSize;
        auto* ptr = primal_.data() + ((primal_.extent(1) - 1) * S);
        for (auto row = 0L; row < len; row += S) {
            ForwardPass(range, row, false);
            auto const rem = std::min(S, len - row);
            std::ranges::copy(std::span(ptr, rem), result.data() + row);
        }
        return {};
    }

    [[nodiscard]] auto Evaluate(Operon::Span<T const> coeff, Operon::Range range) const
        -> tl::expected<Operon::Vector<T>, InterpreterError> final
    {
        Operon::Vector<T> result(range.Size());
        auto evaluated = Evaluate(coeff, range, { result.data(), result.size() });
        if (!evaluated) {
            return tl::unexpected(std::move(evaluated.error()));
        }
        return result;
    }

    [[nodiscard]] auto JacRev(Operon::Span<T const> coeff, Operon::Range range, Operon::Span<T> jacobian) const
        -> tl::expected<void, InterpreterError> final
    {
        if (auto prepared = Prepare(coeff, range, /*needDerivatives=*/true); !prepared) {
            return prepared;
        }
        if (auto sized = CheckOutputSize(range.Size() * coeff.size(), jacobian.size()); !sized) {
            return sized;
        }
        UpdateCoefficients(coeff);
        auto const len = static_cast<int64_t>(range.Size());
        auto const& nodes = tree_->Nodes();
        auto const nn = std::ssize(nodes);
        constexpr int64_t S = BatchSize;
        trace_ = Backend::Buffer<T, S>(S, nn);
        Backend::Fill<T, S>(trace_, nn - 1, T { 1 });
        std::size_t j = 0;
        auto const cols = BuildColumns([&](std::size_t i) -> std::size_t { return nodes[i].Optimize ? j++ : NoIndex; });
        Eigen::Map<Eigen::Array<T, -1, -1>> jac(jacobian.data(), len, coeff.size());
        for (auto row = 0L; row < len; row += S) {
            ForwardPass(range, row, true);
            ReverseTrace<false>(
                range, row, cols.colOf,
                [&](std::size_t i, auto const& primal, T w) -> Eigen::Array<T, S, 1> {
                    if (nodes[i].IsConstant()) {
                        return Eigen::Array<T, S, 1>::Ones();
                    }
                    if (nodes[i].IsVariable() && w == T { 0 }) {
                        Eigen::Array<T, S, 1> values = Eigen::Array<T, S, 1>::Zero();
                        auto const input = dataset_->GetValues(nodes[i].HashValue)
                                               .subspan(range.Start() + static_cast<std::size_t>(row));
                        auto const count = std::min<std::size_t>(S, input.size());
                        std::copy_n(input.data(), count, values.data());
                        return values;
                    }
                    return primal.col(static_cast<Eigen::Index>(i)) / w;
                },
                jac);
        }
        return {};
    }

    [[nodiscard]] auto JacRev(Operon::Span<T const> coeff, Operon::Range range) const
        -> tl::expected<Eigen::Array<T, -1, -1>, InterpreterError> final
    {
        auto const nr = static_cast<int64_t>(range.Size());
        Eigen::Array<T, -1, -1> jacobian(nr, coeff.size());
        auto result = JacRev(coeff, range, { jacobian.data(), static_cast<size_t>(jacobian.size()) });
        if (!result) {
            return tl::unexpected(std::move(result.error()));
        }
        return jacobian;
    }

    [[nodiscard]] auto JacFwd(Operon::Span<T const> coeff, Operon::Range range, Operon::Span<T> jacobian) const
        -> tl::expected<void, InterpreterError> final
    {
        if (auto prepared = Prepare(coeff, range, /*needDerivatives=*/true); !prepared) {
            return prepared;
        }
        if (auto sized = CheckOutputSize(range.Size() * coeff.size(), jacobian.size()); !sized) {
            return sized;
        }
        UpdateCoefficients(coeff);
        auto const& nodes = tree_->Nodes();
        auto const nNodes = std::ssize(nodes);
        auto const nRows = static_cast<int>(range.Size());
        trace_ = Backend::Buffer<T, BatchSize>(BatchSize, nNodes);
        Backend::Fill<T, BatchSize>(trace_, nNodes - 1, T { 1 });
        std::size_t j = 0;
        auto const cols = BuildColumns([&](std::size_t i) -> std::size_t { return nodes[i].Optimize ? j++ : NoIndex; });
        Eigen::Map<Eigen::Array<T, -1, -1>> jac(jacobian.data(), nRows, coeff.size());
        for (int row = 0; row < nRows; row += BatchSize) {
            ForwardPass(range, row, true);
            ForwardTraceGeneric<false>(
                range, row, cols.seeds,
                [&](std::size_t i, auto const& primal, T w) -> Eigen::Array<T, BatchSize, 1> {
                    if (nodes[i].IsConstant()) {
                        return Eigen::Array<T, BatchSize, 1>::Ones();
                    }
                    if (nodes[i].IsVariable() && w == T { 0 }) {
                        Eigen::Array<T, BatchSize, 1> values = Eigen::Array<T, BatchSize, 1>::Zero();
                        auto const input = dataset_->GetValues(nodes[i].HashValue)
                                               .subspan(range.Start() + static_cast<std::size_t>(row));
                        auto const count = std::min<std::size_t>(BatchSize, input.size());
                        std::copy_n(input.data(), count, values.data());
                        return values;
                    }
                    return primal.col(static_cast<Eigen::Index>(i)) / w;
                },
                jac);
        }
        return {};
    }

    [[nodiscard]] auto JacFwd(Operon::Span<T const> coeff, Operon::Range range) const
        -> tl::expected<Eigen::Array<T, -1, -1>, InterpreterError> final
    {
        auto const nRows = static_cast<int64_t>(range.Size());
        Eigen::Array<T, -1, -1> jacobian(nRows, coeff.size());
        auto result = JacFwd(coeff, range, { jacobian.data(), static_cast<size_t>(jacobian.size()) });
        if (!result) {
            return tl::unexpected(std::move(result.error()));
        }
        return jacobian;
    }

    [[nodiscard]] auto JacRevVariable(Operon::Span<T const> coeff, Operon::Range range, Operon::Hash variable,
        Operon::Span<T> result) const -> tl::expected<void, InterpreterError> final
    {
        if (auto prepared = Prepare(coeff, range, /*needDerivatives=*/true); !prepared) {
            return prepared;
        }
        if (auto sized = CheckOutputSize(range.Size(), result.size()); !sized) {
            return sized;
        }
        UpdateCoefficients(coeff);
        auto const len = static_cast<int64_t>(range.Size());
        auto const& nodes = tree_->Nodes();
        auto const nn = std::ssize(nodes);
        constexpr int64_t S = BatchSize;
        trace_ = Backend::Buffer<T, S>(S, nn);
        Backend::Fill<T, S>(trace_, nn - 1, T { 1 });
        auto const cols = BuildColumns([&](std::size_t i) -> std::size_t {
            return (nodes[i].IsVariable() && nodes[i].HashValue == variable) ? 0 : NoIndex;
        });
        Eigen::Map<Eigen::Array<T, -1, -1>> jac(result.data(), len, 1);
        jac.setZero();
        for (auto row = 0L; row < len; row += S) {
            ForwardPass(range, row, true);
            ReverseTrace<true>(
                range, row, cols.colOf,
                [](std::size_t, auto const&, T w) { return Eigen::Array<T, S, 1>::Constant(w); }, jac);
        }
        return {};
    }

    [[nodiscard]] auto JacRevVariable(Operon::Span<T const> coeff, Operon::Range range, Operon::Hash variable) const
        -> tl::expected<Operon::Vector<T>, InterpreterError> final
    {
        Operon::Vector<T> result(range.Size());
        auto evaluated = JacRevVariable(coeff, range, variable, { result.data(), result.size() });
        if (!evaluated) {
            return tl::unexpected(std::move(evaluated.error()));
        }
        return result;
    }

    [[nodiscard]] auto JacFwdVariable(Operon::Span<T const> coeff, Operon::Range range, Operon::Hash variable,
        Operon::Span<T> result) const -> tl::expected<void, InterpreterError> final
    {
        if (auto prepared = Prepare(coeff, range, /*needDerivatives=*/true); !prepared) {
            return prepared;
        }
        if (auto sized = CheckOutputSize(range.Size(), result.size()); !sized) {
            return sized;
        }
        UpdateCoefficients(coeff);
        auto const& nodes = tree_->Nodes();
        auto const nNodes = std::ssize(nodes);
        auto const nRows = static_cast<int>(range.Size());
        trace_ = Backend::Buffer<T, BatchSize>(BatchSize, nNodes);
        Backend::Fill<T, BatchSize>(trace_, nNodes - 1, T { 1 });
        auto const cols = BuildColumns([&](std::size_t i) -> std::size_t {
            return (nodes[i].IsVariable() && nodes[i].HashValue == variable) ? 0 : NoIndex;
        });
        Eigen::Map<Eigen::Array<T, -1, -1>> jac(result.data(), nRows, 1);
        jac.setZero();
        for (int row = 0; row < nRows; row += BatchSize) {
            ForwardPass(range, row, true);
            ForwardTraceGeneric<true>(
                range, row, cols.seeds,
                [](std::size_t, auto const&, T w) { return Eigen::Array<T, BatchSize, 1>::Constant(w); }, jac);
        }
        return {};
    }

    [[nodiscard]] auto JacFwdVariable(Operon::Span<T const> coeff, Operon::Range range, Operon::Hash variable) const
        -> tl::expected<Operon::Vector<T>, InterpreterError> final
    {
        Operon::Vector<T> result(range.Size());
        auto evaluated = JacFwdVariable(coeff, range, variable, { result.data(), result.size() });
        if (!evaluated) {
            return tl::unexpected(std::move(evaluated.error()));
        }
        return result;
    }

    // Evaluates the full tree and extracts values at multiple node indices; roots set to SIZE_MAX produce zero columns.
    [[nodiscard]] auto EvaluateRoots(Operon::Span<T const> coeff, Operon::Range range,
        Operon::Span<std::size_t const> roots) const -> tl::expected<Eigen::Array<T, -1, -1>, InterpreterError>
    {
        if (auto prepared = Prepare(coeff, range, /*needDerivatives=*/false); !prepared) {
            return tl::unexpected(std::move(prepared.error()));
        }
        auto const nNodes = static_cast<std::size_t>(tree_->Nodes().size());
        for (auto root : roots) {
            if (root != NoIndex && root >= nNodes) {
                return tl::unexpected(InterpreterError { InterpreterError::Code::InvalidRootIndex, {}, nNodes, root });
            }
        }
        UpdateCoefficients(coeff);
        auto const len = static_cast<int64_t>(range.Size());
        auto const nRoots = static_cast<Eigen::Index>(roots.size());
        constexpr int64_t S = BatchSize;

        // Not zero-initialized: every row is written exactly once by the
        // batch loop below; NoIndex columns are zeroed explicitly.
        Eigen::Array<T, -1, -1> result(len, nRoots);
        for (Eigen::Index k = 0; k < nRoots; ++k) {
            if (roots[k] == NoIndex) {
                result.col(k).setZero();
            }
        }

        for (auto row = 0L; row < len; row += S) {
            ForwardPass(range, row, /*trace=*/false);
            auto const rem = std::min(S, len - row);
            for (Eigen::Index k = 0; k < nRoots; ++k) {
                if (roots[k] == NoIndex) {
                    continue;
                }
                auto const* src = primal_.data() + (static_cast<int64_t>(roots[k]) * S);
                std::copy_n(src, rem, result.col(k).data() + row);
            }
        }
        return result;
    }

    [[nodiscard]] auto GetTree() const -> Operon::Tree const* { return tree_.get(); }
    [[nodiscard]] auto GetDataset() const -> Operon::Dataset const* { return dataset_.get(); }

    auto GetDispatchTable() const { return dtable_.get(); }

    static auto Evaluate(Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Range const range)
        -> tl::expected<Operon::Vector<T>, InterpreterError>
    {
        auto coeff = tree.GetCoefficients();
        DTable dt;
        return Interpreter { &dt, &dataset, &tree }.Evaluate(coeff, range);
    }

    static auto Evaluate(Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Range const range,
        Operon::Span<T const> coeff) -> tl::expected<Operon::Vector<T>, InterpreterError>
    {
        DTable dt;
        return Interpreter { &dt, &dataset, &tree }.Evaluate(coeff, range);
    }

private:
    // Everything a node needs to be evaluated for the bound range: its (mutable) weight, the variable's column slice,
    // and the resolved primitive/derivative callables (empty where not applicable: leaves have neither, and a function
    // without a registered derivative has no Derivative).
    struct NodeContext {
        T Weight;
        std::span<T const> Values;
        Dispatch::Callable<T, BatchSize> Function;
        Dispatch::CallableDiff<T, BatchSize> Derivative;
    };

    gsl::not_null<DTable const*> dtable_;
    gsl::not_null<Operon::Dataset const*> dataset_;
    gsl::not_null<Operon::Tree const*> tree_;

    // Mutable scratch state for forward/reverse passes; not synchronized (see thread-affinity note above).
    mutable Operon::Vector<NodeContext> context_;
    mutable Backend::Buffer<T, BatchSize> primal_;
    mutable Backend::Buffer<T, BatchSize> trace_;
    mutable Backend::Buffer<T, BatchSize> refAdj_; // adjoints Refs send to their targets (reverse sweep only)
    // Distinct Ref targets of the bound tree and a per-node mask of them; both stay empty for trees without Refs.
    mutable Operon::Vector<std::size_t> refTargets_;
    mutable Operon::Vector<char> isRefTarget_;
    mutable Operon::Range range_ {};
    // First function node (in tree order) without a usable derivative, if any; Jacobian entry points report it.
    mutable std::optional<Operon::Hash> missingDerivative_;

    // private methods
    auto ForwardPass(Operon::Range range, int row, bool trace = false) const -> void
    {
        auto const& nodes = tree_->Nodes();
        auto const nNodes = std::ssize(nodes);
        auto const rangeStart = static_cast<int64_t>(range.Start());
        auto const rangeSize = static_cast<int64_t>(range.Size());
        constexpr int64_t S = BatchSize;

        auto rem = std::min(S, rangeSize - row);
        Operon::Range rg(rangeStart + row, rangeStart + row + rem);

        // The reverse sweep accumulates Ref adjoints into their targets. A target with a function parent has its column
        // overwritten by that parent's partials below; any other target must start every batch from zero.
        if (trace && !refTargets_.empty()) {
            for (auto const target : refTargets_) {
                Backend::Fill<T, S>(trace_, target, T { 0 });
            }
        }

        // forward pass - compute primal and trace
        for (auto i = 0L; i < nNodes; ++i) {
            if (nodes[i].IsConstant()) {
                continue;
            }

            auto const& [p, v, f, df] = context_[i];
            auto* ptr = primal_.data() + (i * S);

            if (nodes[i].IsRef()) {
                EXPECT(static_cast<int64_t>(nodes[i].RefTo) < i); // backward reference invariant
                auto const* src = primal_.data() + (static_cast<int64_t>(nodes[i].RefTo) * S);
                std::copy_n(src, S, ptr);
            } else if (nodes[i].IsVariable()) {
                std::ranges::transform(v.subspan(row, rem), ptr, [p](auto x) { return x * p; });
            } else {
                std::invoke(f, nodes, primal_, i, rg);

                // first compute the partials
                if (trace) {
                    EXPECT(df); // Prepare rejects tracing a tree with a missing derivative
                    for (auto j : Tree::Indices(nodes, i)) {
                        std::invoke(df, nodes, primal_, trace_, i, j);
                    }
                }

                // apply weight after partials are computed
                // if (p != T{1}) {
                //    std::ranges::transform(std::span(ptr, rem), ptr, [p](auto x) { return x * p; });
                //}
            }
        }
    }

    // Sentinel: "no such node/column/root index".
    static constexpr std::size_t NoIndex = std::numeric_limits<std::size_t>::max();

    // Per-node output-column mapping: colOf[i] for ReverseTraceGeneric, seeds (node, column) pairs for
    // ForwardTraceGeneric.
    struct Columns {
        Operon::Vector<std::size_t> colOf;
        Operon::Vector<std::pair<std::size_t, std::size_t>> seeds;
    };

    template <typename Predicate> auto BuildColumns(Predicate predicate) const -> Columns
    {
        auto const nNodes = static_cast<std::size_t>(tree_->Nodes().size());
        Columns cols;
        cols.colOf.assign(nNodes, NoIndex);
        for (std::size_t i = 0; i < nNodes; ++i) {
            if (auto const col = predicate(i); col != NoIndex) {
                cols.colOf[i] = col;
                cols.seeds.emplace_back(i, col);
            }
        }
        return cols;
    }

    // Shared forward-mode sweep behind JacFwd/JacFwdVariable; one seeded pass per output column (see BuildColumns).
    template <bool Accumulate, typename LocalFactor>
    auto ForwardTraceGeneric(Operon::Range range, int row,
        Operon::Vector<std::pair<std::size_t, std::size_t>> const& seeds, LocalFactor factor,
        Eigen::Ref<Eigen::Array<T, -1, -1>> jac) const -> void
    {
        auto const rangeSize = static_cast<int64_t>(range.Size());
        auto const& nodes = tree_->Nodes();
        auto const nNodes = std::ssize(nodes);
        constexpr int64_t S = BatchSize;
        auto const remainingRows = std::min(S, rangeSize - row);

        Eigen::Array<T, S, -1> dot(S, nNodes);

        Eigen::Map<Eigen::Array<T, S, -1>> primal(primal_.data(), S, nNodes);
        Eigen::Map<Eigen::Array<T, S, -1>> trace(trace_.data(), S, nNodes);

        for (auto const& [c, col] : seeds) {
            auto const cc = static_cast<int64_t>(c);

            dot.topRows(remainingRows).setConstant(T { 0 });
            dot.col(cc).head(remainingRows).setConstant(T { 1 });

            for (auto i = 0; i < nNodes; ++i) {
                if (nodes[i].IsRef()) {
                    EXPECT(static_cast<int64_t>(nodes[i].RefTo) < i); // backward reference invariant
                    dot.col(i).head(remainingRows) = dot.col(static_cast<int64_t>(nodes[i].RefTo)).head(remainingRows);
                    continue;
                }
                if (nodes[i].IsLeaf()) {
                    continue;
                }
                for (auto x : Tree::Indices(nodes, i)) {
                    auto j { static_cast<int64_t>(x) };
                    // A leaf child other than the seeded node cc has a zero tangent and can be skipped — except a Ref,
                    // which is a leaf by arity but an alias: its dot was already copied from its (possibly seeded)
                    // target above and must not be dropped just because j != cc.
                    if (nodes[j].IsLeaf() && !nodes[j].IsRef() && j != cc) {
                        continue;
                    }
                    dot.col(i).head(remainingRows)
                        += dot.col(j).head(remainingRows) * trace.col(j).head(remainingRows) * context_[i].Weight;
                }
            }

            auto const w = context_[c].Weight;
            auto const factorValues = factor(c, primal, w);
            auto const contribution
                = (dot.col(nNodes - 1).head(remainingRows) * factorValues.head(remainingRows)).eval();
            if constexpr (Accumulate) {
                jac.col(static_cast<Eigen::Index>(col)).segment(row, remainingRows) += contribution;
            } else {
                jac.col(static_cast<Eigen::Index>(col)).segment(row, remainingRows) = contribution;
            }
        }
    }

    // Shared reverse-mode sweep behind JacRev/JacRevVariable; one backward pass covers every output column.
    // Dispatches once per batch so that trees without Refs (the default) run a sweep with no Ref handling at all.
    template <bool Accumulate, typename LocalFactor>
    auto ReverseTrace(Operon::Range range, int row, Operon::Vector<std::size_t> const& colOf, LocalFactor factor,
        Eigen::Ref<Eigen::Array<T, -1, -1>> jac) const -> void
    {
        if (refTargets_.empty()) {
            ReverseTraceGeneric<Accumulate, false>(range, row, colOf, factor, jac);
        } else {
            ReverseTraceGeneric<Accumulate, true>(range, row, colOf, factor, jac);
        }
    }

    template <bool Accumulate, bool HasRefs, typename LocalFactor>
    auto ReverseTraceGeneric(Operon::Range range, int row, Operon::Vector<std::size_t> const& colOf, LocalFactor factor,
        Eigen::Ref<Eigen::Array<T, -1, -1>> jac) const -> void
    {
        auto const rangeSize = static_cast<int64_t>(range.Size());
        auto const& nodes = tree_->Nodes();
        auto const nNodes = std::ssize(nodes);
        constexpr int64_t S = BatchSize;
        auto const remainingRows = std::min(S, rangeSize - row);

        Eigen::Map<Eigen::Array<T, S, -1>> primal(primal_.data(), S, nNodes);
        Eigen::Map<Eigen::Array<T, S, -1>> trace(trace_.data(), S, nNodes);

        // The adjoint a Ref sends to its target is collected apart from the local partials held in trace: the target's
        // own parent only turns its partial into an adjoint when that parent is processed, which can be after the Ref.
        // The collected adjoint is added to the target's own when the sweep reaches the target.
        if constexpr (HasRefs) {
            if (refAdj_.extent(1) != nNodes) {
                refAdj_ = Backend::Buffer<T, S>(S, nNodes);
            }
            for (auto const target : refTargets_) {
                Backend::Fill<T, S>(refAdj_, target, T { 0 });
            }
        }
        [[maybe_unused]] Eigen::Map<Eigen::Array<T, S, -1>> refAdj(refAdj_.data(), S, HasRefs ? nNodes : 0);

        for (auto i = nNodes - 1; i >= 0L; --i) {
            auto w = context_[i].Weight;

            if constexpr (HasRefs) {
                if (isRefTarget_[static_cast<std::size_t>(i)] != 0) {
                    trace.col(i).head(remainingRows) += refAdj.col(i).head(remainingRows);
                }
            }

            if (auto const col = colOf[static_cast<std::size_t>(i)]; col != NoIndex) {
                auto const factorValues = factor(static_cast<std::size_t>(i), primal, w);
                auto const contribution = (trace.col(i).head(remainingRows) * factorValues.head(remainingRows)).eval();
                if constexpr (Accumulate) {
                    jac.col(static_cast<Eigen::Index>(col)).segment(row, remainingRows) += contribution;
                } else {
                    jac.col(static_cast<Eigen::Index>(col)).segment(row, remainingRows) = contribution;
                }
            }

            if constexpr (HasRefs) {
                if (nodes[i].IsRef()) {
                    EXPECT(static_cast<int64_t>(nodes[i].RefTo)
                        < i); // backward ref: target processed after us in reverse sweep
                    // Accumulate gradient into the referenced node (may be referenced >1 time)
                    refAdj.col(static_cast<int64_t>(nodes[i].RefTo)).head(remainingRows)
                        += trace.col(i).head(remainingRows);
                    continue;
                }
            }
            if (nodes[i].IsLeaf()) {
                continue;
            }

            for (auto j : Tree::Indices(nodes, i)) {
                auto const x { static_cast<int64_t>(j) };
                trace.col(x).head(remainingRows) *= trace.col(i).head(remainingRows) * w;
            }
        }
    }

public:
    // Full bind: validate the tree against the dataset and dispatch table, resolve every node's callables, and allocate
    // primal_. A missing derivative is only an error if requireDerivatives is set; otherwise it is remembered and
    // reported by the Jacobian entry points.
    auto BindTree(Operon::Range range, bool requireDerivatives = false) const -> tl::expected<void, InterpreterError>
    {
        auto const& nodes = tree_->Nodes();
        auto const nRows = static_cast<int64_t>(range.Size());
        auto const nNodes = std::ssize(nodes);
        auto const& dt = dtable_.get();

        Operon::Vector<NodeContext> context;
        context.reserve(nNodes);
        std::optional<Operon::Hash> missingDerivative;
        for (auto const& n : nodes) {
            NodeContext node { .Weight = T { n.Value }, .Values = {}, .Function = {}, .Derivative = {} };
            if (n.IsVariable()) {
                if (!dataset_->FindVariableName(n.HashValue)) {
                    return tl::unexpected(
                        InterpreterError { .Kind = InterpreterError::Code::MissingVariable, .Hash = n.HashValue });
                }
                node.Values = std::span<T const>(
                    dataset_->GetValues(n.HashValue).subspan(range.Start(), range.Size()).data(), nRows);
            } else if (!n.IsLeaf()) {
                auto const* function = dt->template FindFunction<T>(n.HashValue);
                if (function == nullptr) {
                    return tl::unexpected(
                        InterpreterError { .Kind = InterpreterError::Code::MissingPrimitive, .Hash = n.HashValue });
                }
                node.Function = *function;
                auto const* derivative = dt->template FindDerivative<T>(n.HashValue);
                if (derivative != nullptr) {
                    node.Derivative = *derivative;
                } else if (!missingDerivative) {
                    missingDerivative = n.HashValue;
                }
            }
            context.push_back(std::move(node));
        }
        if (requireDerivatives && missingDerivative) {
            return tl::unexpected(
                InterpreterError { .Kind = InterpreterError::Code::MissingDerivative, .Hash = *missingDerivative });
        }
        constexpr int64_t S = BatchSize;
        primal_ = Backend::Buffer<T, S>(S, nNodes);
        std::ranges::fill_n(primal_.data(), S * nNodes, T { 0 });
        context_ = std::move(context);
        missingDerivative_ = missingDerivative;
        refTargets_.clear();
        isRefTarget_.clear();
        for (auto const& n : nodes) {
            if (n.IsRef()) {
                if (isRefTarget_.empty()) {
                    isRefTarget_.assign(static_cast<std::size_t>(nNodes), 0);
                }
                if (isRefTarget_[n.RefTo] == 0) {
                    isRefTarget_[n.RefTo] = 1;
                    refTargets_.push_back(n.RefTo);
                }
            }
        }
        range_ = range;
        return {};
    }

private:
    // Validates the coefficient count, (re)binds the tree if the range changed, and rejects a derivative request on a
    // tree with a missing derivative.
    [[nodiscard]] auto Prepare(Operon::Span<T const> coeff, Operon::Range range, bool needDerivatives) const
        -> tl::expected<void, InterpreterError>
    {
        if (auto valid = ValidateCoefficients(coeff); !valid) {
            return valid;
        }
        if (context_.empty() || range_ != range) {
            if (auto bound = BindTree(range); !bound) {
                return bound;
            }
        }
        if (needDerivatives && missingDerivative_) {
            return tl::unexpected(
                InterpreterError { .Kind = InterpreterError::Code::MissingDerivative, .Hash = *missingDerivative_ });
        }
        return {};
    }

    [[nodiscard]] static auto CheckOutputSize(std::size_t expected, std::size_t actual)
        -> tl::expected<void, InterpreterError>
    {
        if (actual != expected) {
            return tl::unexpected(InterpreterError { InterpreterError::Code::InvalidOutputSize, {}, expected, actual });
        }
        return {};
    }

    [[nodiscard]] auto ValidateCoefficients(Operon::Span<T const> coeff) const -> tl::expected<void, InterpreterError>
    {
        if (coeff.empty()) {
            return {};
        }
        auto const expected = static_cast<std::size_t>(
            std::ranges::count_if(tree_->Nodes(), [](auto const& node) { return node.Optimize; }));
        if (coeff.size() != expected) {
            return tl::unexpected(
                InterpreterError { InterpreterError::Code::InvalidCoefficientSize, {}, expected, coeff.size() });
        }
        return {};
    }

    auto UpdateCoefficients(Operon::Span<T const> coeff) const
    {
        auto const& nodes = tree_->Nodes();
        auto const nNodes = std::ssize(nodes);
        constexpr int64_t S = BatchSize;
        for (int64_t i = 0, j = 0; i < nNodes; ++i) {
            auto const& n = nodes[i];
            if (!coeff.empty() && n.Optimize) {
                context_[i].Weight = T { coeff[j++] };
            }
            if (n.IsConstant()) {
                Backend::Fill<T, S>(primal_, i, context_[i].Weight);
            }
        }
    }
};

// Convenience methods to interpret many trees in parallel (mostly useful from the Python wrapper).
auto OPERON_EXPORT EvaluateTrees(
    Operon::Vector<Operon::Tree> const& trees, Operon::Dataset const* dataset, Operon::Range range, size_t nthread = 0)
    -> tl::expected<Operon::Vector<Operon::Vector<Operon::Scalar>>, TreeEvaluationError>;
auto OPERON_EXPORT EvaluateTrees(Operon::Vector<Operon::Tree> const& trees, Operon::Dataset const* dataset,
    Operon::Range range, std::span<Operon::Scalar> result, size_t nthread = 0)
    -> tl::expected<void, TreeEvaluationError>;
} // namespace Operon
#endif
