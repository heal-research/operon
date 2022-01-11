// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_INDIVIDUAL_HPP
#define OPERON_INDIVIDUAL_HPP

#include "tree.hpp" 
#include "types.hpp" 
#include <cstddef>

namespace Operon {

enum class Dominance : int { Left = 0,
    Equal = 1,
    Right = 2,
    None = 3 };

namespace detail {
    template <typename T>
    inline auto Compare(T const* lhs, T const* rhs, size_t n) noexcept -> Dominance
    {
        Eigen::Map<Eigen::Array<T, -1, 1> const> x(lhs, n);
        Eigen::Map<Eigen::Array<T, -1, 1> const> y(rhs, n);
        auto better = (x < y).any();
        auto worse = (x > y).any();
        if (better) { return worse ? Dominance::None : Dominance::Left; }
        if (worse) { return better ? Dominance::None : Dominance::Right; }
        return Dominance::Equal;
    }
} // namespace detail

struct LexicographicalComparison; // fwd def

struct Individual {
    Tree Genotype;
    Operon::Vector<Operon::Scalar> Fitness;
    size_t Rank{}; // domination rank; used by NSGA2
    Operon::Scalar Distance{}; // crowding distance; used by NSGA2

    inline auto operator[](size_t const i) noexcept -> Operon::Scalar& { return Fitness[i]; }
    inline auto operator[](size_t const i) const noexcept -> Operon::Scalar { return Fitness[i]; }

    [[nodiscard]] inline auto Size() const noexcept -> size_t { return Fitness.size(); }

    Individual()
        : Individual(1)
    {
    }
    explicit Individual(size_t nObj)
        : Fitness(nObj, 0.0)
    {
    }

    inline auto operator==(Individual const& other) const noexcept -> bool
    {
        return std::equal(Fitness.begin(), Fitness.end(), other.Fitness.begin());
    }

    inline auto operator!=(Individual const& other) const noexcept -> bool
    {
        return !(*this == other);
    }

    [[nodiscard]] inline auto LexicographicalCompare(Individual const& other) const noexcept -> bool;

    [[nodiscard]] inline auto ParetoCompare(Individual const& other) const noexcept -> Dominance
    {
        return detail::Compare(Fitness.data(), other.Fitness.data(), Fitness.size());
    }
};

struct Comparison {
    virtual auto operator()(Individual const&, Individual const&) const -> bool = 0;
    virtual ~Comparison() noexcept = default;
    Comparison() = default;
    Comparison(Comparison const&) = default;
    Comparison(Comparison&&) = default;
    auto operator=(Comparison const&) -> Comparison& = default;
    auto operator=(Comparison&&) -> Comparison& = default;
};

struct SingleObjectiveComparison final : public Comparison {
    explicit SingleObjectiveComparison(size_t idx)
        : objectiveIndex_(idx)
    {
    }
    SingleObjectiveComparison()
        : SingleObjectiveComparison(0)
    {
    }

    auto operator()(Individual const& lhs, Individual const& rhs) const -> bool override
    {
        return lhs[objectiveIndex_] < rhs[objectiveIndex_];
    }

    [[nodiscard]] auto GetObjectiveIndex() const -> size_t { return objectiveIndex_; }
    void SetObjectiveIndex(size_t objIdx) { objectiveIndex_ = objIdx; }

private:
    size_t objectiveIndex_;
};

struct LexicographicalComparison : public Comparison {
    auto operator()(Individual const& lhs, Individual const& rhs) const -> bool override
    {
        EXPECT(std::size(lhs.Fitness) == std::size(rhs.Fitness));
        return std::lexicographical_compare(lhs.Fitness.cbegin(), lhs.Fitness.cend(), rhs.Fitness.cbegin(), rhs.Fitness.cend());
    }
};

// TODO: use a collection of SingleObjectiveComparison functors
// returns true if lhs dominates rhs
struct ParetoComparison : public Comparison {
    // assumes minimization in every dimension
    auto operator()(Individual const& lhs, Individual const& rhs) const -> bool override
    {
        EXPECT(std::size(lhs.Fitness) == std::size(rhs.Fitness));
        bool better { false };
        bool worse { false };

        for (size_t i = 0; i < std::size(lhs.Fitness); ++i) {
            better |= lhs[i] < rhs[i];
            worse |= lhs[i] > rhs[i];
        }

        return better && !worse;
    }
};

struct CrowdedComparison : public Comparison {

    auto operator()(Individual const& lhs, Individual const& rhs) const -> bool override
    {
        EXPECT(std::size(lhs.Fitness) == std::size(rhs.Fitness));
        return std::tie(lhs.Rank, rhs.Distance) < std::tie(rhs.Rank, lhs.Distance);
    }
};

auto Individual::LexicographicalCompare(Individual const& other) const noexcept -> bool
{
    return LexicographicalComparison{}(*this, other);
}

using ComparisonCallback = std::function<bool(Individual const&, Individual const&)>;

} // namespace Operon

#endif
