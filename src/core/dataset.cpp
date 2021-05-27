// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "core/dataset.hpp"
#include <fmt/core.h>

#include "core/constants.hpp"
#include "core/types.hpp"
#include "hash/hash.hpp"
#include <rapidcsv.h>

namespace Operon {

// internal implementation details
namespace {
    const auto defaultVariables = [](size_t count) {
        Hasher<HashFunction::XXHash> hash;

        std::vector<Variable> vars(count);
        for (size_t i = 0; i < vars.size(); ++i) {
            vars[i].Name = fmt::format("X{}", i+1);
            vars[i].Index = i;
            vars[i].Hash = hash(reinterpret_cast<uint8_t const*>(vars[i].Name.c_str()), vars[i].Name.size());
        }
        std::sort(vars.begin(), vars.end(), [](auto &a, auto &b) { return a.Hash < b.Hash; });
        return vars;
    };
}

Dataset::Matrix Dataset::ReadCsv(std::string const& path, bool hasHeader)
{
    rapidcsv::Document doc(path, rapidcsv::LabelParams(hasHeader-1, -1));

    // get row and column count
    size_t nrow = doc.GetRowCount();
    size_t ncol = doc.GetColumnCount();

    // fix column names and initialize the variables
    // allocate and fill in values
    if (hasHeader) {
        auto names = doc.GetColumnNames();

        Hasher<HashFunction::XXHash> hash;

        variables.resize(ncol);
        for (size_t i = 0; i < ncol; ++i) {
            auto h = hash(reinterpret_cast<uint8_t const*>(names[i].c_str()), names[i].size());
            Variable v { names[i], h, i };
            variables[i] = v;
        }
        std::sort(variables.begin(), variables.end(), [](auto& a, auto& b) { return a.Hash < b.Hash; });
    } else {
        variables = defaultVariables(ncol);
    }
    Matrix m(nrow, ncol);
    for (size_t i = 0; i < ncol; ++i) {
        auto col = m.col((Eigen::Index)i);
        for (size_t j = 0; j < nrow; ++j) {
            col((Eigen::Index)j) = doc.GetCell<Operon::Scalar>(i, j);
        }
    }
    return m;
}

Dataset::Dataset(std::vector<std::vector<Operon::Scalar>> const& vals)
    : Dataset(defaultVariables(vals.size()), vals)
{
}

Dataset::Dataset(std::string const& path, bool hasHeader)
    : values(ReadCsv(path, hasHeader))
    , map(values.data(), values.rows(), values.cols())
{
}

Dataset::Dataset(Matrix&& vals)
    : variables(defaultVariables((size_t)vals.cols()))
    , values(std::move(vals))
    , map(values.data(), values.rows(), values.cols())
{
}

Dataset::Dataset(Matrix const& vals)
    : variables(defaultVariables((size_t)vals.cols()))
    , values(vals)
    , map(values.data(), values.rows(), values.cols())
{
}

Dataset::Dataset(Eigen::Ref<Matrix const> ref)
    : variables(defaultVariables((size_t)ref.cols()))
    , map(ref.data(), ref.rows(), ref.cols()) 
{
}

void Dataset::SetVariableNames(std::vector<std::string> const& names)
{
    if (names.size() != (size_t)map.cols()) {
        auto msg = fmt::format("The number of columns ({}) does not match the number of column names ({}).", map.cols(), names.size());
        throw std::runtime_error(msg);
    }

    size_t ncol = (size_t)map.cols();

    for (size_t i = 0; i < ncol; ++i) {
        auto h = Hasher<HashFunction::XXHash>{}(reinterpret_cast<uint8_t const*>(names[i].c_str()), names[i].size());
        Variable v { names[i], h, i };
        variables[i] = v;
    }

    std::sort(variables.begin(), variables.end(), [&](auto& a, auto& b) { return a.Hash < b.Hash; });
}

std::vector<std::string> Dataset::VariableNames()
{
    std::vector<std::string> names;
    std::transform(variables.begin(), variables.end(), std::back_inserter(names), [](const Variable& v) { return v.Name; });
    return names;
}

Operon::Span<const Operon::Scalar> Dataset::GetValues(const std::string& name) const noexcept
{
    auto hashValue = Hasher<HashFunction::XXHash>{}(reinterpret_cast<uint8_t const*>(name.c_str()), name.size());
    return GetValues(hashValue);
}

Operon::Span<const Operon::Scalar> Dataset::GetValues(Operon::Hash hashValue) const noexcept
{
    auto it = std::partition_point(variables.begin(), variables.end(), [&](const auto& v) { return v.Hash < hashValue; });
    auto idx = static_cast<Eigen::Index>(it->Index);
    return Operon::Span<const Operon::Scalar>(map.col(idx).data(), static_cast<size_t>(map.rows()));
}

// this method needs to take an int argument to differentiate it from GetValues(Operon::Hash)
Operon::Span<const Operon::Scalar> Dataset::GetValues(int index) const noexcept
{
    return Operon::Span<const Operon::Scalar>(map.col(index).data(), static_cast<size_t>(map.rows()));
}

const std::optional<Variable> Dataset::GetVariable(const std::string& name) const noexcept
{
    auto hashValue = Hasher<HashFunction::XXHash>{}(reinterpret_cast<uint8_t const*>(name.c_str()), name.size());
    return GetVariable(hashValue);
}

const std::optional<Variable> Dataset::GetVariable(Operon::Hash hashValue) const noexcept
{
    auto it = std::partition_point(variables.begin(), variables.end(), [&](const auto& v) { return v.Hash < hashValue; });
    return it < variables.end() ? std::make_optional(*it) : std::nullopt; 
}

void Dataset::Shuffle(Operon::RandomGenerator& random)
{
    if (IsView()) { throw std::runtime_error("Cannot shuffle. Dataset does not own the data.\n"); }
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(values.rows());
    perm.setIdentity();
    // generate a random permutation
    std::shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size(), random);
    values = perm * values.matrix(); // permute rows
}

void Dataset::Normalize(size_t i, Range range)
{
    if (IsView()) { throw std::runtime_error("Cannot normalize. Dataset does not own the data.\n"); }
    EXPECT(range.Start() + range.Size() <= static_cast<size_t>(values.rows()));
    auto j     = static_cast<Eigen::Index>(i);
    auto start = static_cast<Eigen::Index>(range.Start());
    auto size  = static_cast<Eigen::Index>(range.Size());
    auto seg   = values.col(j).segment(start, size);
    auto min   = seg.minCoeff();
    auto max   = seg.maxCoeff();
    values.col(j) = (values.col(j).array() - min) / (max - min);
}

// standardize column i using mean and stddev calculated over the specified range
void Dataset::Standardize(size_t i, Range range)
{
    if (IsView()) { throw std::runtime_error("Cannot standardize. Dataset does not own the data.\n"); }
    EXPECT(range.Start() + range.Size() <= static_cast<size_t>(values.rows()));
    auto j = static_cast<Eigen::Index>(i);
    auto start = static_cast<Eigen::Index>(range.Start());
    auto n = static_cast<Eigen::Index>(range.Size());
    auto seg = values.col(j).segment(start, n);
    MeanVarianceCalculator calc;
    auto vals = Operon::Span<const Operon::Scalar>(seg.data(), range.Size());
    calc.Add(vals);
    values.col(j) = (values.col(j).array() - calc.Mean()) / calc.NaiveStandardDeviation();
}
} // namespace Operon

