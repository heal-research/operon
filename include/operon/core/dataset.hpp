// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef DATASET_H
#define DATASET_H

#include <algorithm>
#include <Eigen/Dense>
#include <exception>
#include <numeric>
#include <vector>
#include <optional>

#include "core/contracts.hpp"
#include "core/range.hpp"
#include "core/types.hpp"

namespace Operon {

// a dataset variable described by: name, hash value (for hashing), data column index
struct Variable {
    std::string Name = "";
    Operon::Hash Hash = Operon::Hash { 0 };
    size_t Index = size_t { 0 };

    constexpr bool operator==(Variable const& rhs) const noexcept {
        return std::tie(Name, Hash, Index) == std::tie(rhs.Name, rhs.Hash, rhs.Index);
    }

    constexpr bool operator!=(Variable const& rhs) const noexcept {
        return !(*this == rhs);
    }
};

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

    // read data from a csv file and return a map (view of the data)
    Matrix ReadCsv(std::string const& path, bool hasHeader);

    // this method ensures the same ordering of variables in the variables vector
    // based on index, name, hash value
    void InitializeVariables(std::vector<std::string> const&);

public:
    Dataset(const std::string& file, bool hasHeader = false);

    Dataset(Dataset const& rhs)
        : variables(rhs.variables)
        , values(rhs.values)
        , map(values.data(), values.rows(), values.cols())
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
        , map(nullptr, static_cast<Eigen::Index>(vals[0].size()), static_cast<Eigen::Index>(vals.size()))
    {
        values = Matrix(map.rows(), map.cols());

        for (Eigen::Index i = 0; i < values.cols(); ++i) {
            auto m = Eigen::Map<Eigen::Matrix<Operon::Scalar, Eigen::Dynamic, 1, Eigen::ColMajor> const>(vals[(size_t)i].data(), map.rows());
            values.col(i) = m;
        }
        new (&map) Map(values.data(), values.rows(), values.cols()); // we use placement new (no allocation)
    }

    Dataset(std::vector<std::vector<Operon::Scalar>> const& vals);

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
        new (&map) Map(values.data(), values.rows(), values.cols()); // we use placement new (no allocation)
    }

    bool operator==(Dataset const& rhs) const noexcept
    {
        return
            Rows() == rhs.Rows() &&
            Cols() == rhs.Cols() &&
            variables.size() == rhs.variables.size() &&
            std::equal(variables.begin(), variables.end(), rhs.variables.begin()) &&
            values.isApprox(rhs.values);
    }

    // check if we own the data or if we are a view over someone else's data
    bool IsView() const noexcept { return values.data() != map.data(); }

    size_t Rows() const { return (size_t)map.rows(); }
    size_t Cols() const { return (size_t)map.cols(); }
    std::pair<size_t, size_t> Dimensions() const { return { Rows(), Cols() }; }

    Eigen::Ref<Matrix const> Values() const { return map; }

    std::vector<std::string> VariableNames();
    void SetVariableNames(std::vector<std::string> const&);

    Operon::Span<const Operon::Scalar> GetValues(const std::string& name) const noexcept;
    Operon::Span<const Operon::Scalar> GetValues(Operon::Hash hashValue) const noexcept;
    Operon::Span<const Operon::Scalar> GetValues(int index) const noexcept;
    Operon::Span<const Operon::Scalar> GetValues(Variable const& variable) const noexcept { return GetValues(variable.Hash); }

    std::optional<Variable> GetVariable(const std::string& name) const noexcept;
    std::optional<Variable> GetVariable(Operon::Hash hashValue) const noexcept;

    Operon::Span<const Variable> Variables() const noexcept { return Operon::Span<const Variable>(variables.data(), variables.size()); }

    void Shuffle(Operon::RandomGenerator& random);

    void Normalize(size_t i, Range range);

    // standardize column i using mean and stddev calculated over the specified range
    void Standardize(size_t i, Range range);
};
}

#endif
