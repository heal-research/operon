// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include <vstat/vstat.hpp>
#include <parser.hpp>
#include <fast_float/fast_float.h>
#include <fmt/format.h>

#include "operon/core/dataset.hpp"
#include "operon/core/types.hpp"
#include "operon/hash/hash.hpp"

namespace Operon {

void Dataset::BuildPaddedCols()
{
    auto const nrows = Rows<int>();
    auto const ncols = Cols<int>();
    paddedRows_ = (nrows + 7) & ~7;
    paddedCols_.assign(static_cast<std::size_t>(paddedRows_) * static_cast<std::size_t>(ncols), Scalar{0});
    for (auto j = 0; j < ncols; ++j) {
        auto const src = ColSpan(j);
        std::copy(src.begin(), src.end(),
                  paddedCols_.data() + static_cast<std::ptrdiff_t>(j) * paddedRows_);
    }
}

namespace {
    auto VariablesFromNames(auto const& names) -> Dataset::Variables {
        Hasher const hasher;
        Dataset::Variables vars;
        for (auto i = 0; i < std::ssize(names); ++i) {
            auto const& name = names[i];
            auto const hash = hasher(name);
            vars.insert({hash, {name, hash, i}});
        }
        return vars;
    }

    auto DefaultVariables(int count) -> Dataset::Variables {
        std::vector<std::string> names(count);
        for (auto i = 0; i < count; ++i) { names[i] = fmt::format("X{}", i + 1); }
        return VariablesFromNames(names);
    }

    auto StorageFromCols(std::vector<std::vector<Operon::Scalar>> const& cols) -> Dataset::Storage {
        auto const ncols = static_cast<int>(cols.size());
        auto const nrows = static_cast<int>(cols.front().size());
        Dataset::Storage s(nrows, ncols);
        auto* dst = s.container().data();
        for (auto j = 0; j < ncols; ++j) {
            std::copy_n(cols[j].data(), nrows, dst + (static_cast<ptrdiff_t>(j) * nrows));
        }
        return s;
    }

    auto MakeView(Dataset::Storage const& s) -> Dataset::View {
        return s.to_mdspan();
    }
} // namespace

auto Dataset::ReadCsv(std::string const& path, bool hasHeader) -> Dataset::Storage
{
    std::ifstream f(path);
    aria::csv::CsvParser parser(f);

    auto nrow = static_cast<int>(std::count(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>(), '\n'));
    f.clear();
    f.seekg(0);

    auto ncol{0};

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
    if (ncol > 0) { buf.resize(static_cast<size_t>(nrow) * ncol); }
    std::vector<Operon::Scalar> rowBuf;
    auto rowIdx = 0;

    for (auto const& row : parser) {
        rowBuf.clear();
        for (auto const& field : row) {
            Operon::Scalar v{0};
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

    Storage s(nrow, ncol);
    std::copy(buf.begin(), buf.end(), s.container().data());
    return s;
}

Dataset::Dataset(std::string const& path, bool hasHeader)
    : storage_(ReadCsv(path, hasHeader))
    , view_(MakeView(storage_))
{
    BuildPaddedCols();
}

Dataset::Dataset(std::vector<std::string> const& vars, std::vector<std::vector<Scalar>> const& vals)
    : variables_(VariablesFromNames(vars))
    , storage_(StorageFromCols(vals))
    , view_(MakeView(storage_))
{
    BuildPaddedCols();
}

Dataset::Dataset(std::vector<std::vector<Scalar>> const& vals)
    : variables_(DefaultVariables(static_cast<int>(vals.size())))
    , storage_(StorageFromCols(vals))
    , view_(MakeView(storage_))
{
    BuildPaddedCols();
}

Dataset::Dataset(gsl::not_null<Scalar const*> data, int rows, int cols)
    : variables_(DefaultVariables(cols))
    , view_(data, rows, cols)
{
    BuildPaddedCols();
}

Dataset::Dataset(Dataset const& rhs)
    : variables_(rhs.variables_)
    , weights_(rhs.weights_)
    , paddedRows_(rhs.paddedRows_)
{
    auto const nrows = rhs.view_.extent(0);
    auto const ncols = rhs.view_.extent(1);
    storage_ = Storage(nrows, ncols);
    auto const n = static_cast<size_t>(nrows) * static_cast<size_t>(ncols);
    std::copy_n(rhs.view_.data_handle(), n, storage_.container().data());
    view_ = MakeView(storage_);
    paddedCols_ = rhs.paddedCols_;
}

Dataset::Dataset(Dataset&& rhs) noexcept
    : variables_(std::move(rhs.variables_))
    , storage_(std::move(rhs.storage_))
    , view_(rhs.view_)
    , weights_(std::move(rhs.weights_))
    , paddedCols_(std::move(rhs.paddedCols_))
    , paddedRows_(rhs.paddedRows_)
{
}

auto Dataset::operator=(Dataset&& rhs) noexcept -> Dataset&
{
    if (this != &rhs) {
        variables_  = std::move(rhs.variables_);
        storage_    = std::move(rhs.storage_);
        view_       = rhs.view_;
        weights_    = std::move(rhs.weights_);
        paddedCols_ = std::move(rhs.paddedCols_);
        paddedRows_ = rhs.paddedRows_;
    }
    return *this;
}

void Dataset::Swap(Dataset& rhs) noexcept
{
    variables_.swap(rhs.variables_);
    std::swap(storage_, rhs.storage_);
    std::swap(view_, rhs.view_);
    std::swap(weights_, rhs.weights_);
    paddedCols_.swap(rhs.paddedCols_);
    std::swap(paddedRows_, rhs.paddedRows_);
}

auto Dataset::operator==(Dataset const& rhs) const noexcept -> bool
{
    if (Rows() != rhs.Rows() || Cols() != rhs.Cols()) { return false; }
    if (variables_.size() != rhs.variables_.size()) { return false; }
    if (!std::equal(variables_.begin(), variables_.end(), rhs.variables_.begin())) { return false; }
    auto const n = static_cast<size_t>(Rows()) * static_cast<size_t>(Cols());
    return std::equal(view_.data_handle(), view_.data_handle() + n, rhs.view_.data_handle(),
                      [](auto a, auto b) -> bool { return std::abs(a - b) < static_cast<Scalar>(1e-6); });
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
    return GetValues(Hasher{}(name));
}

auto Dataset::GetValues(Operon::Hash hash) const noexcept -> Span<Scalar const>
{
    auto it = variables_.find(hash);
    if (it == variables_.end()) {
        fmt::print(stderr, "GetValues: cannot find variable with hash value {}", hash);
        std::abort();
    }
    return ColSpan(it->second.Index);
}

auto Dataset::GetValues(int64_t index) const noexcept -> Span<Scalar const>
{
    return ColSpan(static_cast<int>(index)); // NOLINT(cppcoreguidelines-narrowing-conversions)
}

auto Dataset::GetVariable(std::string const& name) const noexcept -> std::optional<Variable>
{
    return GetVariable(Hasher{}(name));
}

auto Dataset::GetVariable(Operon::Hash hash) const noexcept -> std::optional<Variable>
{
    auto it = variables_.find(hash);
    if (it == variables_.end()) { return std::nullopt; }
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
    weights_.emplace(w.begin(), w.end());
}

auto Dataset::Weights() const noexcept -> std::optional<Span<Scalar const>>
{
    if (!weights_) { return std::nullopt; }
    return Span<Scalar const>{ weights_->data(), weights_->size() };
}

void Dataset::Shuffle(Operon::RandomGenerator& random)
{
    if (IsView()) { throw std::runtime_error("Cannot shuffle: dataset does not own the data.\n"); }
    std::vector<int> perm(Rows());
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), random);
    PermuteRows(perm);
    if (weights_) {
        Vector<Scalar> tmp(weights_->size());
        for (auto i = 0; i < Rows(); ++i) { tmp[i] = (*weights_)[perm[i]]; }
        *weights_ = std::move(tmp);
    }
    BuildPaddedCols();
}

void Dataset::Normalize(size_t i, Range range)
{
    if (IsView()) { throw std::runtime_error("Cannot normalize: dataset does not own the data.\n"); }
    EXPECT(range.Start() + range.Size() <= static_cast<size_t>(Rows()));
    auto* col   = storage_.container().data() + (static_cast<ptrdiff_t>(i) * Rows()); // NOLINT(bugprone-implicit-widening-of-multiplication-result)
    auto* begin = col + range.Start();
    auto* end   = begin + range.Size();
    auto [minIt, maxIt] = std::minmax_element(begin, end);
    auto const min = *minIt;
    auto const rng = *maxIt - min;
    std::transform(col, col + Rows(), col, [min, rng](auto v) -> Scalar { return (v - min) / rng; });
    BuildPaddedCols();
}

void Dataset::Standardize(size_t i, Range range)
{
    if (IsView()) { throw std::runtime_error("Cannot standardize: dataset does not own the data.\n"); }
    EXPECT(range.Start() + range.Size() <= static_cast<size_t>(Rows()));
    auto* col   = storage_.container().data() + (static_cast<ptrdiff_t>(i) * Rows()); // NOLINT(bugprone-implicit-widening-of-multiplication-result)
    auto* begin = col + range.Start();
    auto const stats  = vstat::univariate::accumulate<Scalar>(begin, begin + range.Size());
    auto const stddev = std::sqrt(stats.variance);
    auto const mu     = stats.mean;
    std::transform(col, col + Rows(), col, [mu, stddev](auto v) -> Scalar { return (v - mu) / stddev; });
    BuildPaddedCols();
}

void Dataset::PermuteRows(std::vector<int> const& perm)
{
    if (IsView()) { throw std::runtime_error("Cannot permute: dataset does not own the data.\n"); }
    ENSURE(std::ssize(perm) == Rows());
    auto const nrows = Rows();
    auto const ncols = Cols();
    auto* data = storage_.container().data();
    std::vector<Scalar> tmp(nrows);
    for (auto j = 0; j < ncols; ++j) {
        auto* col = data + (static_cast<ptrdiff_t>(j) * nrows);
        for (auto k = 0; k < nrows; ++k) { tmp[k] = col[perm[k]]; }
        std::copy(tmp.begin(), tmp.end(), col);
    }
    BuildPaddedCols();
}

auto Dataset::GetPaddedValues(int64_t index) const noexcept -> Scalar const*
{
    ENSURE(!IsView());
    return paddedCols_.data() + static_cast<std::ptrdiff_t>(index) * paddedRows_;
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
    return GetPaddedValues(Hasher{}(name));
}

} // namespace Operon
