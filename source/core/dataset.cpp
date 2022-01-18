// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include <vstat/vstat.hpp>
#include <aria-csv/parser.hpp>
#include <fast_float/fast_float.h>

#include "operon/core/constants.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/types.hpp"
#include "operon/hash/hash.hpp"

namespace Operon {

// internal implementation details
namespace {
    const auto DefaultVariables = [](size_t count) {
        Hasher hash;

        std::vector<Variable> vars(count);
        for (size_t i = 0; i < vars.size(); ++i) {
            vars[i].Name = fmt::format("X{}", i+1);
            vars[i].Index = i;
            vars[i].Hash = hash(reinterpret_cast<uint8_t const*>(vars[i].Name.c_str()), vars[i].Name.size()); // NOLINT
        }
        std::sort(vars.begin(), vars.end(), [](auto &a, auto &b) { return a.Hash < b.Hash; });
        return vars;
    };
} // namespace

auto Dataset::ReadCsv(std::string const& path, bool hasHeader) -> Dataset::Matrix
{
    std::ifstream f(path);
    aria::csv::CsvParser parser(f);

    auto nrow = std::count(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>(), '\n');
    // rewind the ifstream
    f.clear();
    f.seekg(0);

    auto ncol{0UL};

    Hasher hash;
    Matrix m;

    if (hasHeader) {
        --nrow; // for matrix allocation, don't care about column names
        for (auto const& row : parser) {
            for (auto const& field : row) {
                auto h = hash(reinterpret_cast<uint8_t const*>(field.c_str()), field.size()); // NOLINT
                Variable v { field, h, ncol++ };
                variables_.push_back(v);
            }
            break; // read only the first row
        }
        std::sort(variables_.begin(), variables_.end(), [](auto& a, auto& b) { return a.Hash < b.Hash; });
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
            ncol = vec.size();
            m.resize(nrow, static_cast<Eigen::Index>(ncol));
            variables_ = DefaultVariables(ncol);
        }
        m.row(rowIdx) = Eigen::Map<Eigen::Array<Operon::Scalar, Eigen::Dynamic, 1>>(vec.data(), static_cast<Eigen::Index>(vec.size()));
        vec.clear();
        ++rowIdx;
    }
    return m;
}

Dataset::Dataset(std::vector<std::vector<Operon::Scalar>> const& vals)
    : Dataset(DefaultVariables(vals.size()), vals)
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

    auto ncol = static_cast<size_t>(map_.cols());

    for (size_t i = 0; i < ncol; ++i) {
        auto h = Hasher{}(reinterpret_cast<uint8_t const*>(names[i].c_str()), names[i].size()); // NOLINT
        Variable v { names[i], h, i };
        variables_[i] = v;
    }

    std::sort(variables_.begin(), variables_.end(), [&](auto& a, auto& b) { return a.Hash < b.Hash; });
}

auto Dataset::VariableNames() -> std::vector<std::string>
{
    std::vector<std::string> names;
    std::transform(variables_.begin(), variables_.end(), std::back_inserter(names), [](const Variable& v) { return v.Name; });
    return names;
}

auto Dataset::GetValues(const std::string& name) const noexcept -> Operon::Span<const Operon::Scalar>
{
    auto hashValue = Hasher{}(reinterpret_cast<uint8_t const*>(name.c_str()), name.size()); // NOLINT
    return GetValues(hashValue);
}

auto Dataset::GetValues(Operon::Hash hashValue) const noexcept -> Operon::Span<const Operon::Scalar>
{
    auto it = std::partition_point(variables_.begin(), variables_.end(), [&](const auto& v) { return v.Hash < hashValue; });
    bool variableExists = it != variables_.end() && it->Hash == hashValue;
    ENSURE(variableExists);
    auto idx = static_cast<Eigen::Index>(it->Index);
    return {map_.col(idx).data(), static_cast<size_t>(map_.rows())};
}

// this method needs to take an int argument to differentiate it from GetValues(Operon::Hash)
auto Dataset::GetValues(int index) const noexcept -> Operon::Span<const Operon::Scalar>
{
    return {map_.col(index).data(), static_cast<size_t>(map_.rows())};
}

auto Dataset::GetVariable(const std::string& name) const noexcept -> std::optional<Variable>
{
    auto hashValue = Hasher{}(reinterpret_cast<uint8_t const*>(name.c_str()), name.size()); // NOLINT
    return GetVariable(hashValue);
}

auto Dataset::GetVariable(Operon::Hash hashValue) const noexcept -> std::optional<Variable>
{
    auto it = std::partition_point(variables_.begin(), variables_.end(), [&](const auto& v) { return v.Hash < hashValue; });
    return it != variables_.end() && it->Hash == hashValue ? std::make_optional(*it) : std::nullopt;
}

void Dataset::Shuffle(Operon::RandomGenerator& random)
{
    if (IsView()) { throw std::runtime_error("Cannot shuffle. Dataset does not own the data.\n"); }
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(values_.rows());
    perm.setIdentity();
    // generate a random permutation
    Operon::Span<decltype(perm)::IndicesType::Scalar> idx(perm.indices().data(), perm.indices().size());
    std::shuffle(idx.begin(), idx.end(), random);
    values_ = perm * values_.matrix(); // permute rows
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

// standardize column i using mean and stddev calculated over the specified range
void Dataset::Standardize(size_t i, Range range)
{
    if (IsView()) { throw std::runtime_error("Cannot standardize. Dataset does not own the data.\n"); }
    EXPECT(range.Start() + range.Size() <= static_cast<size_t>(values_.rows()));
    auto j = static_cast<Eigen::Index>(i);
    auto start = static_cast<Eigen::Index>(range.Start());
    auto n = static_cast<Eigen::Index>(range.Size());
    auto seg = values_.col(j).segment(start, n);
    auto stats = vstat::univariate::accumulate<Matrix::Scalar>(seg.data(), seg.size());
    auto stddev = std::sqrt(stats.variance);
    values_.col(j) = (values_.col(j).array() - stats.mean) / stddev;
}
} // namespace Operon
