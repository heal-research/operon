// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef DATASET_H
#define DATASET_H

#include <optional>
#include <string>
#include <vector>

#include <gsl/pointers>

#include "operon/operon_export.hpp"
#include "contracts.hpp"
#include "range.hpp"
#include "types.hpp"
#include "variable.hpp"

namespace Operon {

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

    std::optional<Vector<Scalar>> weights_;

    auto ReadCsv(std::string const& path, bool hasHeader) -> Storage;
    void InitializeVariables(std::vector<std::string> const&);

    [[nodiscard]] auto ColSpan(int idx) const noexcept -> Span<Scalar const> {
        return { view_.data_handle() + (static_cast<ptrdiff_t>(idx) * view_.extent(0)), static_cast<size_t>(view_.extent(0)) };
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

    [[nodiscard]] auto IsView() const noexcept -> bool { return storage_.size() == 0; }

    template<std::integral T = int>
    [[nodiscard]] auto Rows() const -> T { return static_cast<T>(view_.extent(0)); }

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

    [[nodiscard]] auto GetVariable(std::string const& name) const noexcept -> std::optional<Variable>;
    [[nodiscard]] auto GetVariable(Operon::Hash hash) const noexcept -> std::optional<Variable>;
    [[nodiscard]] auto GetVariables() const noexcept -> std::vector<Operon::Variable>;

    void SetWeights(Span<Scalar const> w);
    [[nodiscard]] auto Weights() const noexcept -> std::optional<Span<Scalar const>>;

    void Shuffle(Operon::RandomGenerator& random);
    void Normalize(size_t i, Range range);
    void Standardize(size_t i, Range range);
    void PermuteRows(std::vector<int> const& perm);
};

} // namespace Operon

#endif
