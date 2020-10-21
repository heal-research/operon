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
#include "stat/meanvariance.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <algorithm>
#include <exception>
#include <gsl/gsl_util>
#include <numeric>
#include <vector>
#include <optional>

namespace Operon {

class Dataset {
public:
    // some useful aliases
    using Matrix = Eigen::Array<Operon::Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    using Map = Eigen::Map<Matrix const>;


private:
    std::vector<Variable> variables;
    Matrix values;
    Map map;

    Dataset();

    // check if we own the data or if we are a view over someone else's data
    bool IsView() const noexcept { return values.data() != map.data(); }

    // read data from a csv file and return a map (view of the data)
    Map ReadCsv(std::string const& path, bool hasHeader);

    // this method ensures the same ordering of variables in the variables vector
    // based on index, name, hash value
    void InitializeVariables(std::vector<std::string> const&);

public:
    Dataset(const std::string& file, bool hasHeader = false);

    Dataset(Dataset const& rhs)
        : variables(rhs.variables)
        , values(rhs.values)
        , map(rhs.map)
    {
    }

    Dataset(Dataset&& rhs) noexcept
        : variables(rhs.variables)
        , values(std::move(rhs.values))
        , map(std::move(rhs.map))
    {
    }

    Dataset(std::vector<Variable> const& vars, std::vector<std::vector<Operon::Scalar>> const& vals)
        : variables(vars)
        , map(nullptr, vals[0].size(), vals.size())
    {
        values = Matrix(vals.front().size(), vals.size());
        for (size_t i = 0; i < vals.size(); ++i) {
            for (size_t j = 0; j < vals[i].size(); ++j) {
                values(j, i) = vals[i][j];
            }
        }
        new (&map) Map(values.data(), values.rows(), values.cols()); // we use placement new (no allocation)
    }

    Dataset(Eigen::Ref<Matrix const> ref);

    Dataset(Matrix const& vals);

    Dataset(Matrix&& vals);

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

    size_t Rows() const { return map.rows(); }
    size_t Cols() const { return map.cols(); }
    std::pair<size_t, size_t> Dimensions() const { return { Rows(), Cols() }; }

    Eigen::Ref<Matrix const> Values() const { return map; }

    std::vector<std::string> VariableNames();
    void SetVariableNames(std::vector<std::string> const&);

    gsl::span<const Operon::Scalar> GetValues(const std::string& name) const noexcept;
    gsl::span<const Operon::Scalar> GetValues(Operon::Hash hashValue) const noexcept;
    gsl::span<const Operon::Scalar> GetValues(gsl::index index) const noexcept;
    gsl::span<const Operon::Scalar> GetValues(Variable const& variable) const noexcept { return GetValues(variable.Hash); }

    const std::optional<Variable> GetVariable(const std::string& name) const noexcept;
    const std::optional<Variable> GetVariable(Operon::Hash hashValue) const noexcept;

    gsl::span<const Variable> Variables() const noexcept { return gsl::span<const Variable>(variables.data(), variables.size()); }

    void Shuffle(Operon::RandomGenerator& random);

    void Normalize(gsl::index i, Range range);

    // standardize column i using mean and stddev calculated over the specified range
    void Standardize(gsl::index i, Range range);
};
}

#endif
