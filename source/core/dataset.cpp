// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <vstat/vstat.hpp>
#include <parser.hpp>
#include <fast_float/fast_float.h>
#include <fmt/format.h>

#include "operon/core/constants.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/types.hpp"
#include "operon/hash/hash.hpp"

namespace Operon {

// internal implementation details
namespace {
    auto VariablesFromNames(auto const& names) {
        Hasher hasher;
        Dataset::Variables vars;
        for (auto i = 0; i < std::ssize(names); ++i) {
            auto const& name = names[i];
            auto hash = hasher(name);
            auto index = i;
            vars.insert({hash, {name, hash, index}});
        }
        return vars;
    }

    auto DefaultVariables(size_t count) {
        std::vector<std::string> names(count);
        for (auto i = 0UL; i < count; ++i) {
            names[i] = fmt::format("X{}", i+1);
        }
        return VariablesFromNames(names);
    }

    auto MatrixFromValues(auto const& values) {
        Dataset::Matrix m(std::ssize(values.front()), std::ssize(values));
        using M = Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1> const>;
        for (auto i = 0; i < m.cols(); ++i) {
            m.col(i) = M(values[i].data(), m.rows());
        }
        return m;
    }
} // namespace

auto Dataset::ReadCsv(std::string const& path, bool hasHeader) -> Dataset::Matrix
{
    std::ifstream f(path);
    aria::csv::CsvParser parser(f);

    auto nrow = std::count(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>(), '\n');
    // rewind the ifstream
    f.clear();
    f.seekg(0);

    auto ncol{0L};

    Hasher hash;
    Matrix m;

    if (hasHeader) {
        --nrow; // for matrix allocation, don't care about column names
        for (auto const& row : parser) {
            for (auto const& f : row) {
                auto h = hash(f);
                variables_.insert({ h, { f, h, ncol++ } });
            }
            break; // read only the first row
        }
        m.resize(nrow, static_cast<Eigen::Index>(ncol));
    }

    std::vector<Operon::Scalar> vec;
    Eigen::Index rowIdx = 0;

    for (auto const& row : parser) {
        size_t fieldIdx = 0;
        for (auto const& field : row) {
            Operon::Scalar v{0};
            auto status = fast_float::from_chars(field.data(), field.data() + field.size(), v);
            if(status.ec != std::errc()) {
                throw std::runtime_error(fmt::format("failed to parse field {} at line {}\n", fieldIdx, rowIdx));
            }
            vec.push_back(v);
            ++fieldIdx;
        }
        if (ncol == 0) {
            ENSURE(!hasHeader);
            ncol = static_cast<int64_t>(vec.size());
            m.resize(nrow, ncol);
            variables_ = DefaultVariables(ncol);
        }
        m.row(rowIdx) = Eigen::Map<Eigen::Array<Operon::Scalar, Eigen::Dynamic, 1>>(vec.data(), static_cast<Eigen::Index>(vec.size()));
        vec.clear();
        ++rowIdx;
    }
    return m;
}

Dataset::Dataset(std::vector<std::vector<Operon::Scalar>> const& vals)
    : variables_(DefaultVariables(vals.size()))
    , values_(MatrixFromValues(vals))
    , map_(values_.data(), values_.rows(), values_.cols())
{
}

Dataset::Dataset(std::string const& path, bool hasHeader)
    : values_(ReadCsv(path, hasHeader))
    , map_(values_.data(), values_.rows(), values_.cols())
{
}

Dataset::Dataset(Matrix vals)
    : variables_(DefaultVariables(static_cast<size_t>(vals.cols())))
    , values_(std::move(vals))
    , map_(values_.data(), values_.rows(), values_.cols())
{
}

Dataset::Dataset(std::vector<std::string> const& vars, std::vector<std::vector<Operon::Scalar>> const& vals)
    : values_(MatrixFromValues(vals))
    , map_(values_.data(), values_.rows(), values_.cols())
{
    Hasher hasher;
    for (auto i = 0; i < map_.cols(); ++i) {
        auto h = hasher(vars[i]);
        variables_.insert({h, { vars[i], h, i } });
    }
}

Dataset::Dataset(Matrix::Scalar const* data, Eigen::Index rows, Eigen::Index cols) // NOLINT
    : variables_(DefaultVariables(static_cast<size_t>(cols)))
    , map_(data, rows, cols)
{
}

void Dataset::SetVariableNames(std::vector<std::string> const& names)
{
    if (names.size() != static_cast<size_t>(map_.cols())) {
        auto msg = fmt::format("The number of columns ({}) does not match the number of column names ({}).", map_.cols(), names.size());
        throw std::runtime_error(msg);
    }
    variables_ = VariablesFromNames(names);
}

auto Dataset::VariableNames() const -> std::vector<std::string>
{
    std::vector<std::string> names;
    names.reserve(variables_.size());
    std::transform(variables_.begin(), variables_.end(), std::back_inserter(names), [](auto const& p) { return p.second.Name; });
    return names;
}

auto Dataset::VariableHashes() const -> std::vector<Operon::Hash>
{
    std::vector<Operon::Hash> hashes;
    hashes.reserve(variables_.size());
    std::transform(variables_.begin(), variables_.end(), std::back_inserter(hashes), [](auto const& p) { return p.second.Hash; });
    return hashes;
}

auto Dataset::VariableIndices() const -> std::vector<std::size_t>
{
    std::vector<std::size_t> indices;
    indices.reserve(variables_.size());
    std::transform(variables_.begin(), variables_.end(), std::back_inserter(indices), [](auto const& p) { return p.second.Index; });
    return indices;
}

auto Dataset::GetValues(std::string const& name) const noexcept -> Operon::Span<const Operon::Scalar>
{
    return GetValues(Hasher{}(name));
}

auto Dataset::GetValues(Operon::Hash hash) const noexcept -> Operon::Span<const Operon::Scalar>
{
    auto it = variables_.find(hash);
    if (it == variables_.end()) {
        fmt::print(stderr, "GetValues: cannot find variable with hash value {}", hash);
        std::abort();
    }
    return {map_.col(it->second.Index).data(), static_cast<size_t>(map_.rows())};
}

// this method needs to take an int argument to differentiate it from GetValues(Operon::Hash)
auto Dataset::GetValues(int64_t index) const noexcept -> Operon::Span<const Operon::Scalar>
{
    return {map_.col(index).data(), static_cast<size_t>(map_.rows())};
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

void Dataset::Shuffle(Operon::RandomGenerator& random)
{
    if (IsView()) { throw std::runtime_error("Cannot shuffle. Dataset does not own the data.\n"); }
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(values_.rows());
    perm.setIdentity();
    // generate a random permutation
    Operon::Span<decltype(perm)::IndicesType::Scalar> idx(perm.indices().data(), perm.indices().size());
    std::shuffle(idx.begin(), idx.end(), random);
    values_.matrix().applyOnTheLeft(perm); // permute rows
}

void Dataset::Normalize(size_t i, Range range)
{
    if (IsView()) { throw std::runtime_error("Cannot normalize. Dataset does not own the data.\n"); }
    EXPECT(range.Start() + range.Size() <= static_cast<size_t>(values_.rows()));
    auto j     = static_cast<Eigen::Index>(i);
    auto start = static_cast<Eigen::Index>(range.Start());
    auto size  = static_cast<Eigen::Index>(range.Size());
    auto seg   = values_.col(j).segment(start, size);
    auto min   = seg.minCoeff();
    auto max   = seg.maxCoeff();
    values_.col(j) = (values_.col(j).array() - min) / (max - min);
}

void Dataset::PermuteRows(std::vector<Eigen::Index> const& indices)
{
    if (IsView()) { throw std::runtime_error("Cannot shuffle. Dataset does not own the data.\n"); }
    ENSURE(values_.rows() == std::ssize(indices));
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(values_.rows());
    std::copy(indices.begin(), indices.end(), perm.indices().begin());
    values_.matrix().applyOnTheLeft(perm); // permute rows
}

// standardize column i using mean and stddev calculated over the specified range
void Dataset::Standardize(size_t i, Range range)
{
    if (IsView()) { throw std::runtime_error("Cannot standardize. Dataset does not own the data.\n"); }
    EXPECT(range.Start() + range.Size() <= static_cast<size_t>(values_.rows()));
    auto j = static_cast<Eigen::Index>(i);
    auto start = static_cast<Eigen::Index>(range.Start());
    auto n = static_cast<Eigen::Index>(range.Size());
    auto seg = values_.col(j).segment(start, n);
    auto stats = vstat::univariate::accumulate<Matrix::Scalar>(seg.begin(), seg.end());
    auto stddev = std::sqrt(stats.variance);
    values_.col(j) = (values_.col(j).array() - stats.mean) / stddev;
}
} // namespace Operon
