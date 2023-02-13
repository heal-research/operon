// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

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
    using Variables = Operon::Map<Operon::Hash, Operon::Variable>;
    using Matrix = Eigen::Array<Operon::Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    using Map = Eigen::Map<Matrix const>;

private:
    Variables variables_;
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

    Dataset(std::vector<std::string> const& vars, std::vector<std::vector<Operon::Scalar>> const& vals);

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

    template<std::integral T = Eigen::Index>
    [[nodiscard]] auto Rows() const -> T { return static_cast<T>(map_.rows()); }

    template<std::integral T = Eigen::Index>
    [[nodiscard]] auto Cols() const -> T { return static_cast<T>(map_.cols()); }

    template<std::integral T = Eigen::Index>
    [[nodiscard]] auto Dimensions() const -> std::pair<T, T> { return { Rows<T>(), Cols<T>() }; }

    [[nodiscard]] auto Values() const -> Eigen::Ref<Matrix const> { return map_; }

    [[nodiscard]] auto VariableNames() const -> std::vector<std::string>;
    void SetVariableNames(std::vector<std::string> const& names);

    [[nodiscard]] auto VariableHashes() const -> std::vector<Operon::Hash>;
    [[nodiscard]] auto VariableIndices() const -> std::vector<std::size_t>;

    [[nodiscard]] auto GetValues(std::string const& name) const noexcept -> Operon::Span<const Operon::Scalar>;
    [[nodiscard]] auto GetValues(Operon::Hash hash) const noexcept -> Operon::Span<const Operon::Scalar>;
    [[nodiscard]] auto GetValues(int64_t index) const noexcept -> Operon::Span<const Operon::Scalar>;
    [[nodiscard]] auto GetValues(Variable const& variable) const noexcept -> Operon::Span<const Operon::Scalar> { return GetValues(variable.Hash); }

    [[nodiscard]] auto GetVariable(const std::string& name) const noexcept -> std::optional<Variable>;
    [[nodiscard]] auto GetVariable(Operon::Hash hash) const noexcept -> std::optional<Variable>;

    [[nodiscard]] auto GetVariables() const noexcept -> std::vector<Operon::Variable> {
        std::vector<Operon::Variable> variables; variables.reserve(variables_.size());
        auto const& values = variables_.values();
        std::transform(values.begin(), values.end(), std::back_inserter(variables), [](auto const& p) { return p.second; });
        return variables;
    }

    void Shuffle(Operon::RandomGenerator& random);

    void Normalize(size_t i, Range range);

    void PermuteRows(std::vector<Eigen::Index> const& indices);

    // standardize column i using mean and stddev calculated over the specified range
    void Standardize(size_t i, Range range);
};
} // namespace Operon

#endif
