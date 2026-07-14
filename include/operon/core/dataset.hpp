// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef DATASET_H
#define DATASET_H

#include <optional>
#include <string>
#include <vector>

#include <gsl/pointers>
#include <tl/expected.hpp>

#include "operon/operon_export.hpp"
#include "contracts.hpp"
#include "range.hpp"
#include "types.hpp"
#include "variable.hpp"

namespace Operon {

enum class DatasetError : std::uint8_t {
    VariableNotFound = 0,
};

class OPERON_EXPORT Dataset {
public:
    using Variables = Operon::Map<Operon::Hash, Operon::Variable>;
    using Extents   = std::dextents<int, 2>;
    using Storage   = MDArray<Scalar, Extents>;
    using View      = MDSpan<Scalar const, Extents>;

private:
    Variables variables_;
    Storage   storage_;
    View      view_;
    int       rows_{0}; // logical row count; storage_ has (rows_+7)&~7 rows for owning datasets

    std::optional<Vector<Scalar>> weights_;

    struct ViewTag {};
    Dataset(ViewTag, gsl::not_null<Scalar const*> data, int rows, int cols);

    auto ReadCsv(std::string const& path, bool hasHeader) -> std::pair<Storage, int>;
    void InitializeVariables(std::vector<std::string> const&);

    [[nodiscard]] auto ColSpan(int idx) const noexcept -> Span<Scalar const> {
        return { view_.data_handle() + (static_cast<ptrdiff_t>(idx) * view_.extent(0)), static_cast<size_t>(rows_) };
    }

public:
    explicit Dataset(std::string const& path, bool hasHeader = false);
    Dataset(std::vector<std::string> const& vars, std::vector<std::vector<Scalar>> const& vals);
    explicit Dataset(std::vector<std::vector<Scalar>> const& vals);
    Dataset(gsl::not_null<Scalar const*> data, int rows, int cols);

    Dataset(Dataset const& rhs);
    Dataset(Dataset&& rhs) noexcept;

    ~Dataset() = default;

    auto operator=(Dataset rhs) -> Dataset& { Swap(rhs); return *this; }
    auto operator=(Dataset&& rhs) noexcept -> Dataset&;

    void Swap(Dataset& rhs) noexcept;

    auto operator==(Dataset const& rhs) const noexcept -> bool;

    // Returns true when the dataset wraps external memory (created via Dataset::Wrap).
    // Non-owning datasets do not support mutations or GetPaddedValues.
    [[nodiscard]] auto IsView() const noexcept -> bool { return storage_.size() == 0; }

    // Expert factory: wraps an externally-owned, already-padded buffer without copying.
    // Caller must guarantee:
    //   - buffer has ((rows+7)&~7) * cols float32 elements in column-major order
    //   - tail rows [rows, (rows+7)&~7) are zeroed
    //   - buffer outlives the Dataset
    static auto Wrap(gsl::not_null<Scalar const*> data, int rows, int cols) -> Dataset;

    template<std::integral T = int>
    [[nodiscard]] auto Rows() const -> T { return static_cast<T>(rows_); }

    template<std::integral T = int>
    [[nodiscard]] auto Cols() const -> T { return static_cast<T>(view_.extent(1)); }

    template<std::integral T = int>
    [[nodiscard]] auto Dimensions() const -> std::pair<T, T> { return { Rows<T>(), Cols<T>() }; }

    [[nodiscard]] auto Data() const noexcept -> View { return view_; }

    [[nodiscard]] auto VariableNames() const -> std::vector<std::string>;
    void SetVariableNames(std::vector<std::string> const& names);
    [[nodiscard]] auto VariableHashes() const -> std::vector<Operon::Hash>;
    [[nodiscard]] auto VariableIndices() const -> std::vector<std::size_t>;

    [[nodiscard]] auto GetValues(std::string const& name) const noexcept -> Span<Scalar const>;
    [[nodiscard]] auto GetValues(Operon::Hash hash) const noexcept -> Span<Scalar const>;
    [[nodiscard]] auto GetValues(int64_t index) const noexcept -> Span<Scalar const>;
    [[nodiscard]] auto GetValues(Variable const& var) const noexcept -> Span<Scalar const> { return GetValues(var.Hash); }

    // Padded column accessors for SIMD consumers. For owning datasets, storage_ has
    // (nRows+7)&~7 rows; for Wrap()-created views, the caller pre-padded the buffer.
    // In both cases slots [nRows, PaddedRows()) are zero. Not valid for IsView() datasets
    // created via means other than Wrap() (i.e., there is no un-padded view constructor).
    [[nodiscard]] auto PaddedRows() const noexcept -> int { return static_cast<int>(view_.extent(0)); }
    [[nodiscard]] auto GetPaddedValues(int64_t index) const noexcept -> Scalar const*;
    [[nodiscard]] auto GetPaddedValues(Operon::Hash hash) const noexcept -> Scalar const*;
    [[nodiscard]] auto GetPaddedValues(std::string const& name) const noexcept -> Scalar const*;
    [[nodiscard]] auto GetPaddedValues(Variable const& var) const noexcept -> Scalar const* { return GetPaddedValues(var.Hash); }

    [[nodiscard]] auto GetVariable(std::string const& name) const noexcept -> tl::expected<Variable, DatasetError>;
    [[nodiscard]] auto GetVariable(Operon::Hash hash) const noexcept -> tl::expected<Variable, DatasetError>;
    [[nodiscard]] auto GetVariables() const noexcept -> std::vector<Operon::Variable>;

    void SetWeights(Span<Scalar const> w);
    [[nodiscard]] auto Weights() const noexcept -> std::optional<Span<Scalar const>>;

    void Shuffle(Operon::RandomGenerator& random);
    void Normalize(size_t i, Range range);
    void Standardize(size_t i, Range range);
    void PermuteRows(std::vector<int> const& perm);

    // Overwrites the [range.Start(), range.Start() + range.Size()) slice of
    // one column in place - e.g. for permutation-importance-style analyses
    // that need to swap a column's values out, evaluate, then swap the
    // originals back in without cloning the rest of the dataset (see
    // analyzers/permutation_importance.hpp). Rows outside `range` and every
    // other column are untouched.
    void SetValues(Operon::Hash hash, Range range, Span<Scalar const> values);
};

} // namespace Operon

#endif
