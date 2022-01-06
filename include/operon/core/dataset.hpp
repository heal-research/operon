// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef DATASET_H
#define DATASET_H

#include <Eigen/Core>

#include <optional>

#include "operon/operon_export.hpp"
#include "contracts.hpp"
#include "range.hpp"
#include "types.hpp"
#include "variable.hpp"

namespace Operon {

// a dataset variable described by: name, hash value (for hashing), data column index

class OPERON_EXPORT Dataset {
public:
    // some useful aliases
    using Matrix = Eigen::Array<Operon::Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    using Map = Eigen::Map<Matrix const>;

private:
    std::vector<Variable> variables_;
    Matrix values_;
    Map map_;

    Dataset();

    // read data from a csv file and return a map (view of the data)
    auto ReadCsv(std::string const& path, bool hasHeader) -> Matrix;

    // this method ensures the same ordering of variables in the variables vector
    // based on index, name, hash value
    void InitializeVariables(std::vector<std::string> const&);

public:
    explicit Dataset(const std::string& path, bool hasHeader = false);

    Dataset(Dataset const& rhs)
        : variables_(rhs.variables_)
        , values_(rhs.values_)
        , map_(values_.data(), values_.rows(), values_.cols())
    {
    }

    Dataset(Dataset&& rhs) noexcept
        : variables_(std::move(rhs.variables_))
        , values_(std::move(rhs.values_))
        , map_(rhs.map_)
    {
    }

    Dataset(std::vector<Variable> vars, std::vector<std::vector<Operon::Scalar>> const& vals)
        : variables_(std::move(vars))
        , map_(nullptr, static_cast<Eigen::Index>(vals[0].size()), static_cast<Eigen::Index>(vals.size()))
    {
        values_ = Matrix(map_.rows(), map_.cols());

        for (Eigen::Index i = 0; i < values_.cols(); ++i) {
            auto m = Eigen::Map<Eigen::Matrix<Operon::Scalar, Eigen::Dynamic, 1, Eigen::ColMajor> const>(vals[static_cast<size_t>(i)].data(), map_.rows());
            values_.col(i) = m;
        }
        new (&map_) Map(values_.data(), values_.rows(), values_.cols()); // we use placement new (no allocation)
    }

    explicit Dataset(std::vector<std::vector<Operon::Scalar>> const& vals);

    //explicit Dataset(Eigen::Ref<Matrix const> ref);
    Dataset(Matrix::Scalar const* data, Eigen::Index rows, Eigen::Index cols);

    explicit Dataset(Matrix vals);

    ~Dataset() = default;

    auto operator=(Dataset rhs) -> Dataset&
    {
        Swap(rhs);
        return *this;
    }

    auto operator=(Dataset&& rhs) noexcept -> Dataset&
    {
        if (this != &rhs) {
            variables_ = std::move(rhs.variables_);
            values_ = std::move(rhs.values_);
            new (&map_) Map(rhs.map_.data(), rhs.map_.rows(), rhs.map_.cols()); // we use placement new (no allocation)
        }
        return *this;
    }

    void Swap(Dataset& rhs) noexcept
    {
        variables_.swap(rhs.variables_);
        values_.swap(rhs.values_);
        new (&map_) Map(values_.data(), values_.rows(), values_.cols()); // we use placement new (no allocation)
    }

    auto operator==(Dataset const& rhs) const noexcept -> bool
    {
        return
            Rows() == rhs.Rows() &&
            Cols() == rhs.Cols() &&
            variables_.size() == rhs.variables_.size() &&
            std::equal(variables_.begin(), variables_.end(), rhs.variables_.begin()) &&
            values_.isApprox(rhs.values_);
    }

    // check if we own the data or if we are a view over someone else's data
    [[nodiscard]] auto IsView() const noexcept -> bool { return values_.data() != map_.data(); }

    [[nodiscard]] auto Rows() const -> size_t { return static_cast<size_t>(map_.rows()); }
    [[nodiscard]] auto Cols() const -> size_t { return static_cast<size_t>(map_.cols()); }
    [[nodiscard]] auto Dimensions() const -> std::pair<size_t, size_t> { return { Rows(), Cols() }; }

    [[nodiscard]] auto Values() const -> Eigen::Ref<Matrix const> { return map_; }

    auto VariableNames() -> std::vector<std::string>;
    void SetVariableNames(std::vector<std::string> const& names);

    [[nodiscard]] auto GetValues(const std::string& name) const noexcept -> Operon::Span<const Operon::Scalar>;
    [[nodiscard]] auto GetValues(Operon::Hash hashValue) const noexcept -> Operon::Span<const Operon::Scalar>;
    [[nodiscard]] auto GetValues(int index) const noexcept -> Operon::Span<const Operon::Scalar>;
    [[nodiscard]] auto GetValues(Variable const& variable) const noexcept -> Operon::Span<const Operon::Scalar> { return GetValues(variable.Hash); }

    [[nodiscard]] auto GetVariable(const std::string& name) const noexcept -> std::optional<Variable>;
    [[nodiscard]] auto GetVariable(Operon::Hash hashValue) const noexcept -> std::optional<Variable>;

    [[nodiscard]] auto Variables() const noexcept -> Operon::Span<const Variable> { return {variables_.data(), variables_.size()}; }

    void Shuffle(Operon::RandomGenerator& random);

    void Normalize(size_t i, Range range);

    // standardize column i using mean and stddev calculated over the specified range
    void Standardize(size_t i, Range range);
};
} // namespace Operon

#endif
