// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include <fast_float/fast_float.h>
#include <fmt/format.h>
#include <parser.hpp>
#include <vstat/vstat.hpp>

#include "operon/core/dataset.hpp"
#include "operon/core/types.hpp"
#include "operon/hash/hash.hpp"

namespace Operon {

namespace {
    auto VariablesFromNames(auto const& names) -> Dataset::Variables
    {
        Hasher const hasher;
        Dataset::Variables vars;
        for (auto i = 0; i < std::ssize(names); ++i) {
            auto const& name = names[i];
            auto const hash = hasher(name);
            vars.insert({ hash, { name, hash, i } });
        }
        return vars;
    }

    auto DefaultVariables(int count) -> Dataset::Variables
    {
        std::vector<std::string> names(count);
        for (auto i = 0; i < count; ++i) {
            names[i] = fmt::format("X{}", i + 1);
        }
        return VariablesFromNames(names);
    }

    auto StorageFromCols(std::vector<std::vector<Operon::Scalar>> const& cols) -> Dataset::Storage
    {
        auto const ncols = static_cast<int>(cols.size());
        auto const nrows = static_cast<int>(cols.front().size());
        auto const pr = (nrows + 7) & ~7; // NOLINT(hicpp-signed-bitwise) padded rows; tail is zero (value-init)
        Dataset::Storage s(pr, ncols);
        auto* dst = s.container().data();
        for (auto j = 0; j < ncols; ++j) {
            std::copy_n(cols[j].data(), nrows, dst + (static_cast<ptrdiff_t>(j) * pr));
        }
        return s;
    }

    auto MakeView(Dataset::Storage const& s) -> Dataset::View
    {
        return s.to_mdspan();
    }
} // namespace

auto Dataset::ReadCsv(std::string const& path, bool hasHeader) -> std::pair<Dataset::Storage, int>
{
    std::ifstream f(path);
    aria::csv::CsvParser parser(f);

    auto nrow = static_cast<int>(std::count(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>(), '\n'));
    f.clear();
    f.seekg(0);

    auto ncol { 0 };

    if (hasHeader) {
        --nrow;
        for (auto const& row : parser) {
            for (auto const& field : row) {
                Hasher const hash;
                auto h = hash(field);
                variables_.insert({ h, { .Name = field, .Hash = h, .Index = ncol++ } });
            }
            break;
        }
    }

    // flat column-major buffer: buf[col * nrow + row]
    std::vector<Operon::Scalar> buf;
    if (ncol > 0) {
        buf.resize(static_cast<size_t>(nrow) * ncol);
    }
    std::vector<Operon::Scalar> rowBuf;
    auto rowIdx = 0;

    for (auto const& row : parser) {
        rowBuf.clear();
        for (auto const& field : row) {
            Operon::Scalar v { 0 };
            auto const status = fast_float::from_chars(field.data(), field.data() + field.size(), v);
            if (status.ec != std::errc()) {
                throw std::runtime_error(fmt::format("failed to parse field at line {}\n", rowIdx));
            }
            rowBuf.push_back(v);
        }

        if (ncol == 0) {
            ENSURE(!hasHeader);
            ncol = static_cast<int>(rowBuf.size());
            buf.resize(static_cast<size_t>(nrow) * ncol);
            variables_ = DefaultVariables(ncol);
        }

        for (auto j = 0; j < ncol; ++j) {
            buf[(static_cast<size_t>(j) * nrow) + rowIdx] = rowBuf[j];
        }
        ++rowIdx;
    }

    auto const pr = (nrow + 7) & ~7; // NOLINT(hicpp-signed-bitwise) padded rows
    Storage s(pr, ncol);
    auto* dst = s.container().data();
    for (auto j = 0; j < ncol; ++j) {
        std::copy_n(buf.data() + (static_cast<ptrdiff_t>(j) * nrow), nrow, dst + (static_cast<ptrdiff_t>(j) * pr));
    }
    return { std::move(s), nrow };
}

Dataset::Dataset(std::string const& path, bool hasHeader)
{
    auto [stor, nr] = ReadCsv(path, hasHeader);
    storage_ = std::move(stor);
    rows_ = nr;
    view_ = MakeView(storage_);
}

Dataset::Dataset(std::vector<std::string> const& vars, std::vector<std::vector<Scalar>> const& vals)
    : variables_(VariablesFromNames(vars))
    , storage_(StorageFromCols(vals))
    , view_(MakeView(storage_))
    , rows_(static_cast<int>(vals.front().size()))
{
}

Dataset::Dataset(std::vector<std::vector<Scalar>> const& vals)
    : variables_(DefaultVariables(static_cast<int>(vals.size())))
    , storage_(StorageFromCols(vals))
    , view_(MakeView(storage_))
    , rows_(static_cast<int>(vals.front().size()))
{
}

auto Dataset::Wrap(gsl::not_null<Scalar const*> data, int rows, int cols) -> Dataset
{
    EXPECT(rows > 0 && cols > 0);
    return Dataset(ViewTag {}, data, rows, cols);
}

Dataset::Dataset(ViewTag /*unused*/, gsl::not_null<Scalar const*> data, int rows, int cols)
    : variables_(DefaultVariables(cols))
    , view_(data, (rows + 7) & ~7, cols) // NOLINT(hicpp-signed-bitwise)
    , rows_(rows)
{
    // storage_ stays empty → IsView() == true; caller guarantees padding
}

Dataset::Dataset(gsl::not_null<Scalar const*> data, int rows, int cols)
    : variables_(DefaultVariables(cols))
{
    auto const pr = (rows + 7) & ~7; // NOLINT(hicpp-signed-bitwise)
    storage_ = Storage(pr, cols);
    auto* dst = storage_.container().data();
    for (auto j = 0; j < cols; ++j) {
        std::copy_n(data.get() + (static_cast<ptrdiff_t>(j) * rows), rows,
            dst + (static_cast<ptrdiff_t>(j) * pr));
    }
    view_ = MakeView(storage_);
    rows_ = rows;
}

Dataset::Dataset(Dataset const& rhs)
    : variables_(rhs.variables_)
    , rows_(rhs.rows_)
    , weights_(rhs.weights_)
{
    auto const pr = static_cast<ptrdiff_t>(rhs.view_.extent(0)); // paddedRows
    auto const ncols = static_cast<int>(rhs.view_.extent(1));
    storage_ = Storage(pr, ncols);
    std::copy_n(rhs.view_.data_handle(), static_cast<size_t>(pr) * ncols, storage_.container().data());
    view_ = MakeView(storage_);
}

Dataset::Dataset(Dataset&& rhs) noexcept
    : variables_(std::move(rhs.variables_))
    , storage_(std::move(rhs.storage_))
    , view_(rhs.view_)
    , rows_(rhs.rows_)
    , weights_(std::move(rhs.weights_))
{
}

auto Dataset::operator=(Dataset&& rhs) noexcept -> Dataset&
{
    if (this != &rhs) {
        variables_ = std::move(rhs.variables_);
        storage_ = std::move(rhs.storage_);
        view_ = rhs.view_;
        rows_ = rhs.rows_;
        weights_ = std::move(rhs.weights_);
    }
    return *this;
}

void Dataset::Swap(Dataset& rhs) noexcept
{
    variables_.swap(rhs.variables_);
    std::swap(storage_, rhs.storage_);
    std::swap(view_, rhs.view_);
    std::swap(rows_, rhs.rows_);
    std::swap(weights_, rhs.weights_);
}

auto Dataset::operator==(Dataset const& rhs) const noexcept -> bool
{
    if (Rows() != rhs.Rows() || Cols() != rhs.Cols()) {
        return false;
    }
    if (variables_.size() != rhs.variables_.size()) {
        return false;
    }
    if (!std::equal(variables_.begin(), variables_.end(), rhs.variables_.begin())) {
        return false;
    }
    for (auto j = 0; j < Cols(); ++j) {
        auto a = ColSpan(j);
        auto b = rhs.ColSpan(j);
        if (!std::equal(a.begin(), a.end(), b.begin(),
                [](auto x, auto y) -> bool { return std::abs(x - y) < static_cast<Scalar>(1e-6); })) {
            return false;
        }
    }
    return true;
}

void Dataset::SetVariableNames(std::vector<std::string> const& names)
{
    if (std::ssize(names) != Cols()) {
        throw std::runtime_error(fmt::format(
            "The number of columns ({}) does not match the number of column names ({}).",
            Cols(), names.size()));
    }
    variables_ = VariablesFromNames(names);
}

auto Dataset::VariableNames() const -> std::vector<std::string>
{
    std::vector<std::string> names;
    names.reserve(variables_.size());
    std::transform(variables_.begin(), variables_.end(), std::back_inserter(names),
        [](auto const& p) -> auto { return p.second.Name; });
    return names;
}

auto Dataset::VariableHashes() const -> std::vector<Operon::Hash>
{
    std::vector<Operon::Hash> hashes;
    hashes.reserve(variables_.size());
    std::transform(variables_.begin(), variables_.end(), std::back_inserter(hashes),
        [](auto const& p) -> auto { return p.second.Hash; });
    return hashes;
}

auto Dataset::VariableIndices() const -> std::vector<std::size_t>
{
    std::vector<std::size_t> indices;
    indices.reserve(variables_.size());
    std::transform(variables_.begin(), variables_.end(), std::back_inserter(indices),
        [](auto const& p) -> auto { return static_cast<std::size_t>(p.second.Index); });
    return indices;
}

auto Dataset::GetValues(std::string const& name) const noexcept -> Span<Scalar const>
{
    return GetValues(Hasher {}(name));
}

auto Dataset::GetValues(Operon::Hash hash) const noexcept -> Span<Scalar const>
{
    auto it = variables_.find(hash);
    if (it == variables_.end()) {
        fmt::print(stderr, "GetValues: cannot find variable with hash value {}", hash);
        std::abort();
    }
    return ColSpan(static_cast<int>(it->second.Index));
}

auto Dataset::GetValues(int64_t index) const noexcept -> Span<Scalar const>
{
    return ColSpan(static_cast<int>(index)); // NOLINT(cppcoreguidelines-narrowing-conversions)
}

auto Dataset::GetVariable(std::string const& name) const noexcept -> tl::expected<Variable, DatasetError>
{
    return GetVariable(Hasher {}(name));
}

auto Dataset::GetVariable(Operon::Hash hash) const noexcept -> tl::expected<Variable, DatasetError>
{
    auto it = variables_.find(hash);
    if (it == variables_.end()) {
        return tl::make_unexpected(DatasetError::VariableNotFound);
    }
    return it->second;
}

auto Dataset::GetVariables() const noexcept -> std::vector<Operon::Variable>
{
    std::vector<Operon::Variable> variables;
    variables.reserve(variables_.size());
    auto const& vals = variables_.values();
    std::transform(vals.begin(), vals.end(), std::back_inserter(variables),
        [](auto const& p) -> auto { return p.second; });
    return variables;
}

void Dataset::SetWeights(Span<Scalar const> w)
{
    ENSURE(std::ssize(w) == Rows());
    // NOT validated for non-negativity here (unlike the size check above):
    // rows outside any particular Problem's training range are never read by
    // the coefficient optimizers and may legitimately hold negative/sentinel
    // placeholder values (see GaussianLoss's/LMCostFunction's ctors, which
    // validate only the in-range slice they will actually use).
    weights_.emplace(w.begin(), w.end());
}

auto Dataset::Weights() const noexcept -> std::optional<Span<Scalar const>>
{
    if (!weights_) {
        return std::nullopt;
    }
    return Span<Scalar const> { weights_->data(), weights_->size() };
}

void Dataset::Shuffle(Operon::RandomGenerator& random)
{
    if (IsView()) {
        throw std::runtime_error("Cannot shuffle a non-owning dataset.\n");
    }
    std::vector<int> perm(Rows());
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), random);
    PermuteRows(perm);
    if (weights_) {
        Vector<Scalar> tmp(weights_->size());
        for (auto i = 0; i < Rows(); ++i) {
            tmp[i] = (*weights_)[perm[i]];
        }
        *weights_ = std::move(tmp);
    }
}

void Dataset::Normalize(size_t i, Range range)
{
    if (IsView()) {
        throw std::runtime_error("Cannot normalize a non-owning dataset.\n");
    }
    EXPECT(range.Start() + range.Size() <= static_cast<size_t>(Rows()));
    auto const stride = static_cast<ptrdiff_t>(view_.extent(0)); // paddedRows
    auto* col = storage_.container().data() + (static_cast<ptrdiff_t>(i) * stride); // NOLINT(bugprone-implicit-widening-of-multiplication-result)
    auto* begin = col + range.Start();
    auto* end = begin + range.Size();
    auto [minIt, maxIt] = std::minmax_element(begin, end);
    auto const min = *minIt;
    auto const rng = *maxIt - min;
    std::transform(col, col + Rows(), col, [min, rng](auto v) -> Scalar { return (v - min) / rng; });
}

void Dataset::Standardize(size_t i, Range range)
{
    if (IsView()) {
        throw std::runtime_error("Cannot standardize a non-owning dataset.\n");
    }
    EXPECT(range.Start() + range.Size() <= static_cast<size_t>(Rows()));
    auto const stride = static_cast<ptrdiff_t>(view_.extent(0)); // paddedRows
    auto* col = storage_.container().data() + (static_cast<ptrdiff_t>(i) * stride); // NOLINT(bugprone-implicit-widening-of-multiplication-result)
    auto* begin = col + range.Start();
    auto const stats = vstat::univariate::accumulate<Scalar>(begin, begin + range.Size());
    auto const stddev = std::sqrt(stats.variance);
    auto const mu = stats.mean;
    std::transform(col, col + Rows(), col, [mu, stddev](auto v) -> Scalar { return (v - mu) / stddev; });
}

void Dataset::SetValues(Operon::Hash hash, Range range, Span<Scalar const> values)
{
    if (IsView()) {
        throw std::runtime_error("Cannot mutate a non-owning dataset.\n");
    }
    EXPECT(values.size() == range.Size());
    EXPECT(range.Start() + range.Size() <= static_cast<size_t>(Rows()));
    auto it = variables_.find(hash);
    if (it == variables_.end()) {
        fmt::print(stderr, "SetValues: cannot find variable with hash value {}", hash);
        std::abort();
    }
    auto const stride = static_cast<ptrdiff_t>(view_.extent(0)); // paddedRows
    auto* col = storage_.container().data() + (static_cast<ptrdiff_t>(it->second.Index) * stride); // NOLINT(bugprone-implicit-widening-of-multiplication-result)
    std::copy(values.begin(), values.end(), col + range.Start());
}

void Dataset::PermuteRows(std::vector<int> const& perm)
{
    if (IsView()) {
        throw std::runtime_error("Cannot permute a non-owning dataset.\n");
    }
    ENSURE(std::ssize(perm) == Rows());
    auto const nrows = Rows();
    auto const stride = static_cast<ptrdiff_t>(view_.extent(0)); // paddedRows
    auto const ncols = Cols();
    auto* data = storage_.container().data();
    std::vector<Scalar> tmp(nrows);
    for (auto j = 0; j < ncols; ++j) {
        auto* col = data + (static_cast<ptrdiff_t>(j) * stride);
        for (auto k = 0; k < nrows; ++k) {
            tmp[k] = col[perm[k]];
        }
        std::copy(tmp.begin(), tmp.end(), col);
        // tail (col+nrows .. col+stride-1) remains zero
    }
}

auto Dataset::GetPaddedValues(int64_t index) const noexcept -> Scalar const*
{
    return view_.data_handle() + (static_cast<std::ptrdiff_t>(index) * view_.extent(0));
}

auto Dataset::GetPaddedValues(Operon::Hash hash) const noexcept -> Scalar const*
{
    auto it = variables_.find(hash);
    if (it == variables_.end()) {
        fmt::print(stderr, "GetPaddedValues: cannot find variable with hash value {}", hash);
        std::abort();
    }
    return GetPaddedValues(static_cast<int64_t>(it->second.Index));
}

auto Dataset::GetPaddedValues(std::string const& name) const noexcept -> Scalar const*
{
    return GetPaddedValues(Hasher {}(name));
}

} // namespace Operon
