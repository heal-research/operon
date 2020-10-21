/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2019 Bogdan Burlacu 
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE. 
 */

#include "core/dataset.hpp"
#include <fmt/core.h>

#include "core/types.hpp"
#include "rapidcsv.h"

namespace Operon {

// internal implementation details
namespace {
    // compare strings size first, as an attempt to have eg X1, X2, X10 in this order and not X1, X10, X2
    const auto compareWithSize = [](auto& lhs, auto& rhs) { return std::tuple(lhs.size(), lhs) < std::tuple(rhs.size(), rhs); };

    const auto defaultVariables = [](size_t count) {
        std::vector<Variable> vars(count);
        Operon::RandomGenerator rng(1234);
        for(auto& v : vars) { v.Hash = rng(); }
        std::sort(vars.begin(), vars.end(), [](auto& a, auto& b) { return a.Hash < b.Hash; });
        for (size_t i = 0; i < vars.size(); ++i) {
            vars[i].Name = fmt::format("X{}", i+1);
            vars[i].Index = i;
        }
        return vars;
    };
}

Dataset::Map Dataset::ReadCsv(std::string const& path, bool hasHeader)
{
    rapidcsv::Document doc(path, rapidcsv::LabelParams(hasHeader-1, -1));

    // get row and column count
    size_t nrow = doc.GetRowCount();
    size_t ncol = doc.GetColumnCount();

    // fix column names and initialize the variables
    // allocate and fill in values
    if (hasHeader) {
        auto names = doc.GetColumnNames();

        // generate a sequence of sorted hash values
        std::vector<Operon::Hash> hashes(ncol);
        Operon::RandomGenerator rng(1234);
        std::generate_n(hashes.begin(), ncol, [&]() { return rng(); });
        std::sort(hashes.begin(), hashes.end());

        variables.resize(ncol);
        for (size_t i = 0; i < ncol; ++i) {
            Variable v { names[i], 0, i };
            variables[i] = v;
        }

        std::sort(variables.begin(), variables.end(), [&](auto& a, auto& b) { return compareWithSize(a.Name, b.Name); });
        // assign hashes to the sorted variables (so that the ordering is the same and we can binary search by hash value)
        for (size_t i = 0; i < ncol; ++i) {
            variables[i].Hash = hashes[i];
        }
    } else {
        variables = defaultVariables(ncol);
    }
    values = Matrix(nrow, ncol);
    for (size_t i = 0; i < ncol; ++i) {
        auto col = values.col(i);
        for (size_t j = 0; j < nrow; ++j) {
            col(j) = doc.GetCell<Operon::Scalar>(i, j);
        }
    }
    return Map(values.data(), values.rows(), values.cols());
}

Dataset::Dataset(std::string const& path, bool hasHeader)
    : map(ReadCsv(path, hasHeader))
{
}

Dataset::Dataset(Matrix&& vals)
    : variables(defaultVariables(vals.cols()))
    , values(std::move(vals))
    , map(values.data(), values.rows(), values.cols())
{
}

Dataset::Dataset(Matrix const& vals)
    : variables(defaultVariables(vals.cols()))
    , values(vals)
    , map(values.data(), values.rows(), values.cols())
{
}

Dataset::Dataset(Eigen::Ref<Matrix const> ref)
    : variables(defaultVariables(ref.cols()))
    , map(ref.data(), ref.rows(), ref.cols()) 
{
}

void Dataset::SetVariableNames(std::vector<std::string> const& names)
{
    if (names.size() != (size_t)map.cols()) {
        auto msg = fmt::format("The number of columns ({}) does not match the number of column names ({}).", map.cols(), names.size());
        throw std::runtime_error(msg);
    }

    size_t ncol = map.cols();

    std::vector<Operon::Hash> hashes(ncol);
    Operon::RandomGenerator rng(1234);
    std::generate_n(hashes.begin(), ncol, [&]() { return rng(); });
    std::sort(hashes.begin(), hashes.end());

    for (size_t i = 0; i < ncol; ++i) {
        Variable v { names[i], 0, i };
        variables[i] = v;
    }

    std::sort(variables.begin(), variables.end(), [&](auto& a, auto& b) { return compareWithSize(a.Name, b.Name); });
    // assign hashes to the sorted variables (so that the ordering is the same and we can binary search by hash value)
    for (size_t i = 0; i < ncol; ++i) {
        variables[i].Hash = hashes[i];
    }
}

std::vector<std::string> Dataset::VariableNames()
{
    std::vector<std::string> names;
    std::transform(variables.begin(), variables.end(), std::back_inserter(names), [](const Variable& v) { return v.Name; });
    return names;
}

gsl::span<const Operon::Scalar> Dataset::GetValues(const std::string& name) const noexcept
{
    auto it = std::partition_point(variables.begin(), variables.end(), [&](const auto& v) { return compareWithSize(v.Name, name); });
    return gsl::span<const Operon::Scalar>(map.col(it->Index).data(), map.rows());
}

gsl::span<const Operon::Scalar> Dataset::GetValues(Operon::Hash hashValue) const noexcept
{
    auto it = std::partition_point(variables.begin(), variables.end(), [&](const auto& v) { return v.Hash < hashValue; });
    return gsl::span<const Operon::Scalar>(map.col(it->Index).data(), map.rows());
}

gsl::span<const Operon::Scalar> Dataset::GetValues(gsl::index index) const noexcept
{
    return gsl::span<const Operon::Scalar>(map.col(index).data(), map.rows());
}

const std::optional<Variable> Dataset::GetVariable(const std::string& name) const noexcept
{
    auto it = std::partition_point(variables.begin(), variables.end(), [&](const auto& v) { return compareWithSize(v.Name, name); });
    return it < variables.end() ? std::make_optional(*it) : std::nullopt; 
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

void Dataset::Normalize(gsl::index i, Range range)
{
    if (IsView()) { throw std::runtime_error("Cannot normalize. Dataset does not own the data.\n"); }
    Expects(range.Start() + range.Size() < static_cast<size_t>(values.rows()));
    auto seg = values.col(i).segment(range.Start(), range.Size());
    auto min = seg.minCoeff();
    auto max = seg.maxCoeff();
    values.col(i) = (values.col(i).array() - min) / (max - min);
}

// standardize column i using mean and stddev calculated over the specified range
void Dataset::Standardize(gsl::index i, Range range)
{
    if (IsView()) { throw std::runtime_error("Cannot standardize. Dataset does not own the data.\n"); }
    Expects(range.Start() + range.Size() < static_cast<size_t>(values.rows()));
    auto seg = values.col(i).segment(range.Start(), range.Size());
    MeanVarianceCalculator calc;
    auto vals = gsl::span<const Operon::Scalar>(seg.data(), seg.size());
    calc.Add(vals);

    values.col(i) = (values.col(i).array() - calc.Mean()) / calc.StandardDeviation();
}
} // namespace Operon

