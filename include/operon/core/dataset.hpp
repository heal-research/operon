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

#ifndef DATASET_H
#define DATASET_H

#include "core/common.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <algorithm>
#include <exception>
#include <fmt/core.h>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace Operon {
// compare strings size first, as an attempt to have eg X1, X2, X10 in this order and not X1, X10, X2
namespace {
    inline bool CompareWithSize(const std::string& lhs, const std::string& rhs)
    {
        return std::make_tuple(lhs.size(), lhs) < std::make_tuple(rhs.size(), rhs);
    }
}

class Dataset {
private:
    std::vector<Variable> variables;
    using MatrixType = Eigen::Matrix<operon::scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    MatrixType values;

    Dataset();

public:
    Dataset(const std::string& file, bool hasHeader = false);
    Dataset(const Dataset& rhs)
        : variables(rhs.variables)
        , values(rhs.values)
    {
    }
    Dataset(Dataset&& rhs) noexcept
        : variables(rhs.variables)
        , values(std::move(rhs.values))
    {
    }
    Dataset(const std::vector<Variable> vars, const std::vector<std::vector<operon::scalar_t>> vals)
        : variables(vars)
    {
        values = MatrixType(vals.front().size(), vals.size());
        for (size_t i = 0; i < vals.size(); ++i) {
            for (size_t j = 0; j < vals[i].size(); ++j) {
                values(j, i) = vals[i][j];
            }
        }
    }

    Dataset& operator=(Dataset rhs)
    {
        swap(rhs);
        return *this;
    }

    void swap(Dataset& rhs) noexcept
    {
        variables.swap(rhs.variables);
        values.swap(rhs.values);
    }

    size_t Rows() const { return values.rows(); }
    size_t Cols() const { return values.cols(); }
    std::pair<size_t, size_t> Dimensions() const { return { Rows(), Cols() }; }

    const MatrixType& Values() const { return values; }

    const std::vector<std::string> VariableNames() const
    {
        std::vector<std::string> names;
        std::transform(variables.begin(), variables.end(), std::back_inserter(names), [](const Variable& v) { return v.Name; });
        return names;
    }

    const gsl::span<const operon::scalar_t> GetValues(const std::string& name) const
    {
        auto needle = Variable { name, 0, 0 };
        auto record = std::equal_range(variables.begin(), variables.end(), needle, [&](const Variable& a, const Variable& b) { return CompareWithSize(a.Name, b.Name); });
        return gsl::span<const operon::scalar_t>(values.col(record.first->Index).data(), values.rows());
    }

    const gsl::span<const operon::scalar_t> GetValues(operon::hash_t hashValue) const
    {
        auto needle = Variable { "", hashValue, 0 };
        auto record = std::equal_range(variables.begin(), variables.end(), needle, [](const Variable& a, const Variable& b) { return a.Hash < b.Hash; });
        return gsl::span<const operon::scalar_t>(values.col(record.first->Index).data(), values.rows());
    }

    const gsl::span<const operon::scalar_t> GetValues(gsl::index index) const
    {
        return gsl::span<const operon::scalar_t>(values.col(index).data(), values.rows());
    }

    const std::string& GetName(operon::hash_t hashValue) const
    {
        auto needle = Variable { "", hashValue, 0 };
        auto record = std::equal_range(variables.begin(), variables.end(), needle, [](const Variable& a, const Variable& b) { return a.Hash < b.Hash; });
        return record.first->Name;
    }

    operon::hash_t GetHashValue(const std::string& name) const
    {
        auto needle = Variable { name, 0 /* hash value */, 0 /* index */ };
        auto record = std::equal_range(variables.begin(), variables.end(), needle, [](const Variable& a, const Variable& b) { return CompareWithSize(a.Name, b.Name); });
        return record.first->Hash;
    }

    gsl::index GetIndex(operon::hash_t hashValue) const
    {
        auto needle = Variable { "", hashValue, 0 };
        auto record = std::equal_range(variables.begin(), variables.end(), needle, [](const Variable& a, const Variable& b) { return a.Hash < b.Hash; });
        return record.first->Index;
    }
    const std::string& GetName(int index) const { return variables[index].Name; }
    const gsl::span<const Variable> Variables() const { return gsl::span<const Variable>(variables); }
};
}

#endif
