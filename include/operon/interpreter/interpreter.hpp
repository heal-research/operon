// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_INTERPRETER_HPP
#define OPERON_INTERPRETER_HPP

#include <algorithm>
#include <gsl/pointers>
#include <optional>
#include <span>

#include "derivatives.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/formatter/formatter.hpp"
#include <string>
#include <tl/expected.hpp>

// #include "tape.hpp"

namespace Operon {

enum class LikelihoodType : uint8_t { Gaussian,
    Poisson };

struct InterpreterError {
    enum class Code {
        MissingVariable,
        MissingPrimitive,
        MissingDerivative,
        InvalidOutputSize,
        InvalidCoefficientSize,

    };

    Code Kind;
    Operon::Hash Hash{};
    std::size_t ExpectedSize{};
    std::size_t ActualSize{};
};

[[nodiscard]] inline auto FormatInterpreterError(InterpreterError const& error) -> std::string {
    switch (error.Kind) {
    case InterpreterError::Code::MissingVariable: return fmt::format("missing dataset variable with hash {}", error.Hash);
    case InterpreterError::Code::MissingPrimitive: return fmt::format("missing primitive with hash {}", error.Hash);
    case InterpreterError::Code::MissingDerivative: return fmt::format("missing derivative for primitive with hash {}", error.Hash);
    case InterpreterError::Code::InvalidOutputSize: return fmt::format("invalid output size: expected {}, got {}", error.ExpectedSize, error.ActualSize);
    case InterpreterError::Code::InvalidCoefficientSize: return fmt::format("invalid coefficient size: expected {}, got {}", error.ExpectedSize, error.ActualSize);
    }
    std::unreachable();
}

struct TreeEvaluationError {
    std::size_t Index{};
    InterpreterError Error;
};

template <typename T>
struct InterpreterBase {
    InterpreterBase() = default;
    InterpreterBase(const InterpreterBase&) = default;
    InterpreterBase(InterpreterBase&&) = default;
    auto operator=(const InterpreterBase&) -> InterpreterBase& = default;
    auto operator=(InterpreterBase&&) -> InterpreterBase& = default;
    virtual ~InterpreterBase() = default;

    // evaluate model output
    [[nodiscard]] virtual auto Evaluate(Operon::Span<T const> coeff, Operon::Range range, Operon::Span<T> result) const -> tl::expected<void, InterpreterError> = 0;
    [[nodiscard]] virtual auto Evaluate(Operon::Span<T const> coeff, Operon::Range range) const -> tl::expected<Operon::Vector<T>, InterpreterError> = 0;

    // evaluate model jacobian in reverse mode
    [[nodiscard]] virtual auto JacRev(Operon::Span<T const> coeff, Operon::Range range, Operon::Span<T> jacobian) const -> tl::expected<void, InterpreterError> = 0;
    [[nodiscard]] virtual auto JacRev(Operon::Span<T const> coeff, Operon::Range range) const -> tl::expected<Eigen::Array<T, -1, -1>, InterpreterError> = 0;

    // evaluate model jacobian in forward mode
    [[nodiscard]] virtual auto JacFwd(Operon::Span<T const> coeff, Operon::Range range, Operon::Span<T> jacobian) const -> tl::expected<void, InterpreterError> = 0;
    [[nodiscard]] virtual auto JacFwd(Operon::Span<T const> coeff, Operon::Range range) const -> tl::expected<Eigen::Array<T, -1, -1>, InterpreterError> = 0;

    // evaluate model derivative w.r.t. a single input variable's raw value (identified by hash), summed over every occurrence of that variable in the tree — reverse mode
    [[nodiscard]] virtual auto JacRevVariable(Operon::Span<T const> coeff, Operon::Range range, Operon::Hash variable, Operon::Span<T> result) const -> tl::expected<void, InterpreterError> = 0;
    [[nodiscard]] virtual auto JacRevVariable(Operon::Span<T const> coeff, Operon::Range range, Operon::Hash variable) const -> tl::expected<Operon::Vector<T>, InterpreterError> = 0;

    // same as above, forward mode
    [[nodiscard]] virtual auto JacFwdVariable(Operon::Span<T const> coeff, Operon::Range range, Operon::Hash variable, Operon::Span<T> result) const -> tl::expected<void, InterpreterError> = 0;
    [[nodiscard]] virtual auto JacFwdVariable(Operon::Span<T const> coeff, Operon::Range range, Operon::Hash variable) const -> tl::expected<Operon::Vector<T>, InterpreterError> = 0;

    // getters
    [[nodiscard]] virtual auto GetTree() const -> Operon::Tree const* = 0;
    [[nodiscard]] virtual auto GetDataset() const -> Operon::Dataset const* = 0;
};

// Not thread-safe: binds lazily into mutable scratch state reused across calls. Use one instance per worker thread.
template <typename T = Operon::Scalar, typename DTable = ScalarDispatch>
    requires DTable::template
SupportsType<T> struct Interpreter : public InterpreterBase<T> {
    using DispatchTable = DTable;
    static constexpr auto BatchSize = DTable::template BatchSize<T>;

    Interpreter(gsl::not_null<DTable const*> dtable, gsl::not_null<Operon::Dataset const*> dataset, gsl::not_null<Operon::Tree const*> tree)
        : dtable_(dtable)
        , dataset_(dataset)
        , tree_(tree)
    {
    }

    auto Primal() const { return primal_; }
    auto Trace() const { return trace_; }

    [[nodiscard]] auto Evaluate(Operon::Span<T const> coeff, Operon::Range range, Operon::Span<T> result) const -> tl::expected<void, InterpreterError> final
    {
        if (auto valid = ValidateCoefficients(coeff); !valid) { return tl::unexpected(std::move(valid.error())); }
        if (context_.empty() || range_ != range) {
            auto bound = BindTree(range, false);
            if (!bound) { return tl::unexpected(std::move(bound.error())); }
        }
        if (result.size() != range.Size()) {
            return tl::unexpected(InterpreterError { InterpreterError::Code::InvalidOutputSize, {}, range.Size(), result.size() });
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

    [[nodiscard]] auto Evaluate(Operon::Span<T const> coeff, Operon::Range range) const -> tl::expected<Operon::Vector<T>, InterpreterError> final
    {
        Operon::Vector<T> result(range.Size());
        auto evaluated = Evaluate(coeff, range, { result.data(), result.size() });
        if (!evaluated) { return tl::unexpected(std::move(evaluated.error())); }
        return result;
    }

    [[nodiscard]] auto JacRev(Operon::Span<T const> coeff, Operon::Range range, Operon::Span<T> jacobian) const -> tl::expected<void, InterpreterError> final
    {
        if (auto valid = ValidateCoefficients(coeff); !valid) { return tl::unexpected(std::move(valid.error())); }
        if (context_.empty() || range_ != range || !derivativesBound_) {
            auto bound = BindTree(range, true);
            if (!bound) { return tl::unexpected(std::move(bound.error())); }
        }
        auto const expected = range.Size() * coeff.size();
        if (jacobian.size() != expected) {
            return tl::unexpected(InterpreterError { InterpreterError::Code::InvalidOutputSize, {}, expected, jacobian.size() });
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
            ReverseTraceGeneric<false>(range, row, cols.colOf, [&](std::size_t i, auto const& primal, T w) -> Eigen::Array<T, S, 1> {
                if (nodes[i].IsConstant()) { return Eigen::Array<T, S, 1>::Ones(); }
                if (nodes[i].IsVariable() && w == T { 0 }) {
                    Eigen::Array<T, S, 1> values = Eigen::Array<T, S, 1>::Zero();
                    auto const input = dataset_->GetValues(nodes[i].HashValue).subspan(range.Start() + static_cast<std::size_t>(row));
                    auto const count = std::min<std::size_t>(S, input.size());
                    std::copy_n(input.data(), count, values.data());
                    return values;
                }
                return primal.col(static_cast<Eigen::Index>(i)) / w;
            }, jac);
        }
        return {};
    }

    [[nodiscard]] auto JacRev(Operon::Span<T const> coeff, Operon::Range range) const -> tl::expected<Eigen::Array<T, -1, -1>, InterpreterError> final
    {
        auto const nr = static_cast<int64_t>(range.Size());
        Eigen::Array<T, -1, -1> jacobian(nr, coeff.size());
        auto result = JacRev(coeff, range, { jacobian.data(), static_cast<size_t>(jacobian.size()) });
        if (!result) { return tl::unexpected(std::move(result.error())); }
        return jacobian;
    }

    [[nodiscard]] auto JacFwd(Operon::Span<T const> coeff, Operon::Range range, Operon::Span<T> jacobian) const -> tl::expected<void, InterpreterError> final
    {
        if (auto valid = ValidateCoefficients(coeff); !valid) { return tl::unexpected(std::move(valid.error())); }
        if (context_.empty() || range_ != range || !derivativesBound_) {
            auto bound = BindTree(range, true);
            if (!bound) { return tl::unexpected(std::move(bound.error())); }
        }
        auto const expected = range.Size() * coeff.size();
        if (jacobian.size() != expected) {
            return tl::unexpected(InterpreterError { InterpreterError::Code::InvalidOutputSize, {}, expected, jacobian.size() });
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
            ForwardTraceGeneric<false>(range, row, cols.seeds, [&](std::size_t i, auto const& primal, T w) -> Eigen::Array<T, BatchSize, 1> {
                if (nodes[i].IsConstant()) { return Eigen::Array<T, BatchSize, 1>::Ones(); }
                if (nodes[i].IsVariable() && w == T { 0 }) {
                    Eigen::Array<T, BatchSize, 1> values = Eigen::Array<T, BatchSize, 1>::Zero();
                    auto const input = dataset_->GetValues(nodes[i].HashValue).subspan(range.Start() + static_cast<std::size_t>(row));
                    auto const count = std::min<std::size_t>(BatchSize, input.size());
                    std::copy_n(input.data(), count, values.data());
                    return values;
                }
                return primal.col(static_cast<Eigen::Index>(i)) / w;
            }, jac);
        }
        return {};
    }

    [[nodiscard]] auto JacFwd(Operon::Span<T const> coeff, Operon::Range range) const -> tl::expected<Eigen::Array<T, -1, -1>, InterpreterError> final
    {
        auto const nRows = static_cast<int64_t>(range.Size());
        Eigen::Array<T, -1, -1> jacobian(nRows, coeff.size());
        auto result = JacFwd(coeff, range, { jacobian.data(), static_cast<size_t>(jacobian.size()) });
        if (!result) { return tl::unexpected(std::move(result.error())); }
        return jacobian;
    }

    [[nodiscard]] auto JacRevVariable(Operon::Span<T const> coeff, Operon::Range range, Operon::Hash variable, Operon::Span<T> result) const -> tl::expected<void, InterpreterError> final
    {
        if (auto valid = ValidateCoefficients(coeff); !valid) { return tl::unexpected(std::move(valid.error())); }
        if (context_.empty() || range_ != range || !derivativesBound_) {
            auto bound = BindTree(range, true);
            if (!bound) { return tl::unexpected(std::move(bound.error())); }
        }
        if (result.size() != range.Size()) {
            return tl::unexpected(InterpreterError { InterpreterError::Code::InvalidOutputSize, {}, range.Size(), result.size() });
        }
        UpdateCoefficients(coeff);
        auto const len = static_cast<int64_t>(range.Size());
        auto const& nodes = tree_->Nodes();
        auto const nn = std::ssize(nodes);
        constexpr int64_t S = BatchSize;
        trace_ = Backend::Buffer<T, S>(S, nn);
        Backend::Fill<T, S>(trace_, nn - 1, T { 1 });
        auto const cols = BuildColumns([&](std::size_t i) -> std::size_t { return (nodes[i].IsVariable() && nodes[i].HashValue == variable) ? 0 : NoIndex; });
        Eigen::Map<Eigen::Array<T, -1, -1>> jac(result.data(), len, 1);
        jac.setZero();
        for (auto row = 0L; row < len; row += S) {
            ForwardPass(range, row, true);
            ReverseTraceGeneric<true>(range, row, cols.colOf, [](std::size_t, auto const&, T w) { return Eigen::Array<T, S, 1>::Constant(w); }, jac);
        }
        return {};
    }

    [[nodiscard]] auto JacRevVariable(Operon::Span<T const> coeff, Operon::Range range, Operon::Hash variable) const -> tl::expected<Operon::Vector<T>, InterpreterError> final
    {
        Operon::Vector<T> result(range.Size());
        auto evaluated = JacRevVariable(coeff, range, variable, { result.data(), result.size() });
        if (!evaluated) { return tl::unexpected(std::move(evaluated.error())); }
        return result;
    }

    [[nodiscard]] auto JacFwdVariable(Operon::Span<T const> coeff, Operon::Range range, Operon::Hash variable, Operon::Span<T> result) const -> tl::expected<void, InterpreterError> final
    {
        if (auto valid = ValidateCoefficients(coeff); !valid) { return tl::unexpected(std::move(valid.error())); }
        if (context_.empty() || range_ != range || !derivativesBound_) {
            auto bound = BindTree(range, true);
            if (!bound) { return tl::unexpected(std::move(bound.error())); }
        }
        if (result.size() != range.Size()) {
            return tl::unexpected(InterpreterError { InterpreterError::Code::InvalidOutputSize, {}, range.Size(), result.size() });
        }
        UpdateCoefficients(coeff);
        auto const& nodes = tree_->Nodes();
        auto const nNodes = std::ssize(nodes);
        auto const nRows = static_cast<int>(range.Size());
        trace_ = Backend::Buffer<T, BatchSize>(BatchSize, nNodes);
        Backend::Fill<T, BatchSize>(trace_, nNodes - 1, T { 1 });
        auto const cols = BuildColumns([&](std::size_t i) -> std::size_t { return (nodes[i].IsVariable() && nodes[i].HashValue == variable) ? 0 : NoIndex; });
        Eigen::Map<Eigen::Array<T, -1, -1>> jac(result.data(), nRows, 1);
        jac.setZero();
        for (int row = 0; row < nRows; row += BatchSize) {
            ForwardPass(range, row, true);
            ForwardTraceGeneric<true>(range, row, cols.seeds, [](std::size_t, auto const&, T w) { return Eigen::Array<T, BatchSize, 1>::Constant(w); }, jac);
        }
        return {};
    }

    [[nodiscard]] auto JacFwdVariable(Operon::Span<T const> coeff, Operon::Range range, Operon::Hash variable) const -> tl::expected<Operon::Vector<T>, InterpreterError> final
    {
        Operon::Vector<T> result(range.Size());
        auto evaluated = JacFwdVariable(coeff, range, variable, { result.data(), result.size() });
        if (!evaluated) { return tl::unexpected(std::move(evaluated.error())); }
        return result;
    }

    // Evaluates the full tree and extracts values at multiple node indices; roots set to SIZE_MAX produce zero columns.
    auto EvaluateRoots(Operon::Span<T const> coeff, Operon::Range range,
        Operon::Span<std::size_t const> roots) const -> Eigen::Array<T, -1, -1>
    {
        InitContext(coeff, range);

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

        auto const nNodes = static_cast<std::size_t>(tree_->Nodes().size());
        for (Eigen::Index k = 0; k < nRoots; ++k) {
            EXPECT(roots[k] == NoIndex || roots[k] < nNodes);
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

    static auto Evaluate(Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Range const range) -> tl::expected<Operon::Vector<T>, InterpreterError>
    {
        auto coeff = tree.GetCoefficients();
        DTable dt;
        return Interpreter { &dt, &dataset, &tree }.Evaluate(coeff, range);
    }

    static auto Evaluate(Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Range const range, Operon::Span<T const> coeff) -> tl::expected<Operon::Vector<T>, InterpreterError>
    {
        DTable dt;
        return Interpreter { &dt, &dataset, &tree }.Evaluate(coeff, range);
    }
    using Data = std::tuple<T,
        std::span<T const>,
        std::optional<Dispatch::Callable<T, BatchSize> const>,
        std::optional<Dispatch::CallableDiff<T, BatchSize> const>>;

    gsl::not_null<DTable const*> dtable_;
    gsl::not_null<Operon::Dataset const*> dataset_;
    gsl::not_null<Operon::Tree const*> tree_;

    // Mutable scratch state for forward/reverse passes; not synchronized (see thread-affinity note above).
    mutable Operon::Vector<Data> context_;
    mutable Backend::Buffer<T, BatchSize> primal_;
    mutable Backend::Buffer<T, BatchSize> trace_;
    mutable Operon::Range range_ {};
    mutable bool derivativesBound_ {};

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
                std::invoke(*f, nodes, primal_, i, rg);

                // first compute the partials
                if (trace && df) {
                    for (auto j : Tree::Indices(nodes, i)) {
                        std::invoke(*df, nodes, primal_, trace_, i, j);
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

    // Per-node output-column mapping: colOf[i] for ReverseTraceGeneric, seeds (node, column) pairs for ForwardTraceGeneric.
    struct Columns {
        Operon::Vector<std::size_t> colOf;
        Operon::Vector<std::pair<std::size_t, std::size_t>> seeds;
    };

    template <typename Predicate>
    auto BuildColumns(Predicate predicate) const -> Columns
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
    auto ForwardTraceGeneric(Operon::Range range, int row, Operon::Vector<std::pair<std::size_t, std::size_t>> const& seeds, LocalFactor factor, Eigen::Ref<Eigen::Array<T, -1, -1>> jac) const -> void
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
                    // A leaf child other than the seeded node cc has a zero tangent and can be skipped — except a Ref, which is a
                    // leaf by arity but an alias: its dot was already copied from its (possibly seeded) target above and must not
                    // be dropped just because j != cc.
                    if (nodes[j].IsLeaf() && !nodes[j].IsRef() && j != cc) {
                        continue;
                    }
                    dot.col(i).head(remainingRows) += dot.col(j).head(remainingRows) * trace.col(j).head(remainingRows) * std::get<0>(context_[i]);
                }
            }

            auto const w = std::get<0>(context_[c]);
            auto const contribution = dot.col(nNodes - 1).head(remainingRows) * factor(c, primal, w).head(remainingRows);
            if constexpr (Accumulate) {
                jac.col(static_cast<Eigen::Index>(col)).segment(row, remainingRows) += contribution;
            } else {
                jac.col(static_cast<Eigen::Index>(col)).segment(row, remainingRows) = contribution;
            }
        }
    }

    // Shared reverse-mode sweep behind JacRev/JacRevVariable; one backward pass covers every output column.
    template <bool Accumulate, typename LocalFactor>
    auto ReverseTraceGeneric(Operon::Range range, int row, Operon::Vector<std::size_t> const& colOf, LocalFactor factor, Eigen::Ref<Eigen::Array<T, -1, -1>> jac) const -> void
    {
        auto const rangeSize = static_cast<int64_t>(range.Size());
        auto const& nodes = tree_->Nodes();
        auto const nNodes = std::ssize(nodes);
        constexpr int64_t S = BatchSize;
        auto const remainingRows = std::min(S, rangeSize - row);

        Eigen::Map<Eigen::Array<T, S, -1>> primal(primal_.data(), S, nNodes);
        Eigen::Map<Eigen::Array<T, S, -1>> trace(trace_.data(), S, nNodes);

        for (auto i = nNodes - 1; i >= 0L; --i) {
            auto w = std::get<0>(context_[i]);

            if (auto const col = colOf[static_cast<std::size_t>(i)]; col != NoIndex) {
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
                trace.col(static_cast<int64_t>(nodes[i].RefTo)).head(remainingRows) += trace.col(i).head(remainingRows);
                continue;
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
    // Full bind: allocate primal_, build context_ with function/derivative pointers and variable data spans.
    auto BindTree(Operon::Range range, bool requireDerivatives = false) const -> tl::expected<void, InterpreterError>
    {
        auto const& nodes = tree_->Nodes();
        auto const nRows = static_cast<int64_t>(range.Size());
        auto const nNodes = std::ssize(nodes);
        auto const& dt = dtable_.get();
        for (auto const& n : nodes) {
            if (n.IsVariable() && !dataset_->GetVariable(n.HashValue)) {
                return tl::unexpected(InterpreterError { .Kind=InterpreterError::Code::MissingVariable, .Hash=n.HashValue });
            }
            if (!n.IsLeaf() && !dt->template TryGetFunction<T>(n.HashValue)) {
                return tl::unexpected(InterpreterError { .Kind=InterpreterError::Code::MissingPrimitive, .Hash=n.HashValue });
            }
            if (requireDerivatives && !n.IsLeaf()) {
                auto derivative = dt->template TryGetDerivative<T>(n.HashValue);
                if (!derivative || !*derivative) {
                    return tl::unexpected(InterpreterError { .Kind=InterpreterError::Code::MissingDerivative, .Hash=n.HashValue });
                }
            }
        }
        constexpr int64_t S = BatchSize;
        primal_ = Backend::Buffer<T, S>(S, nNodes);
        std::ranges::fill_n(primal_.data(), S * nNodes, T { 0 });
        context_.clear();
        context_.reserve(nNodes);
        for (int64_t i = 0; i < nNodes; ++i) {
            auto const& n = nodes[i];
            auto variableValues = n.IsVariable()
                ? std::tuple_element_t<1, Data>(dataset_->GetValues(n.HashValue).subspan(range.Start(), range.Size()).data(), nRows)
                : std::tuple_element_t<1, Data> {};
            context_.emplace_back(T { n.Value }, variableValues,
                dt->template TryGetFunction<T>(n.HashValue),
                dt->template TryGetDerivative<T>(n.HashValue));
        }
        derivativesBound_ = requireDerivatives;
        range_ = range;
        return {};
    }

private:
    [[nodiscard]] auto ValidateCoefficients(Operon::Span<T const> coeff) const -> tl::expected<void, InterpreterError>
    {
        if (coeff.empty()) { return {}; }
        auto const expected = static_cast<std::size_t>(std::ranges::count_if(tree_->Nodes(), [](auto const& node) { return node.Optimize; }));
        if (coeff.size() != expected) {
            return tl::unexpected(InterpreterError { InterpreterError::Code::InvalidCoefficientSize, {}, expected, coeff.size() });
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
            if (!coeff.empty() && n.Optimize) { std::get<0>(context_[i]) = T { coeff[j++] }; }
            if (n.IsConstant()) { Backend::Fill<T, S>(primal_, i, std::get<0>(context_[i])); }
        }
    }

    auto InitContext(Operon::Span<T const> coeff, Operon::Range range) const
    {
        if (context_.empty() || range_ != range) {
            auto bound = BindTree(range);
            if (!bound) { throw std::runtime_error(FormatInterpreterError(bound.error())); }
        }
        UpdateCoefficients(coeff);
    }
};

// Convenience methods to interpret many trees in parallel (mostly useful from the Python wrapper).
auto OPERON_EXPORT TryEvaluateTrees(Operon::Vector<Operon::Tree> const& trees, Operon::Dataset const* dataset, Operon::Range range, size_t nthread = 0)
    -> tl::expected<Operon::Vector<Operon::Vector<Operon::Scalar>>, TreeEvaluationError>;
auto OPERON_EXPORT TryEvaluateTrees(Operon::Vector<Operon::Tree> const& trees, Operon::Dataset const* dataset, Operon::Range range, std::span<Operon::Scalar> result, size_t nthread = 0)
    -> tl::expected<void, TreeEvaluationError>;
auto OPERON_EXPORT EvaluateTrees(Operon::Vector<Operon::Tree> const& trees, Operon::Dataset const* dataset, Operon::Range range, size_t nthread = 0) -> Operon::Vector<Operon::Vector<Operon::Scalar>>;
auto OPERON_EXPORT EvaluateTrees(Operon::Vector<Operon::Tree> const& trees, Operon::Dataset const* dataset, Operon::Range range, std::span<Operon::Scalar> result, size_t nthread = 0) -> void;
} // namespace Operon
#endif
