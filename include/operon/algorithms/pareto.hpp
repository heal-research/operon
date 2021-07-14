#ifndef OPERON_PARETO_HPP
#define OPERON_PARETO_HPP

#include "core/individual.hpp"

namespace Operon {
enum class DominanceResult : int { LeftDominates = -1, NoDomination = 0, RightDominates = 1, Equality = 2 };

struct DominanceCalculator {
    inline DominanceResult operator()(Individual const& lhs, Individual const& rhs) noexcept {
        EXPECT(lhs.Size() == rhs.Size());
        auto const& a = lhs.Fitness;
        auto const& b = rhs.Fitness;

        bool better{false}, worse{false};
        for (size_t i = 0; i < a.size(); ++i) {
            better |= a[i] < b[i];
            worse  |= a[i] > b[i];
        }

        if (better && worse) return DominanceResult::NoDomination;
        if (!(better || worse)) return DominanceResult::Equality;

        return better
            ? DominanceResult::LeftDominates
            : DominanceResult::RightDominates;
    }

    static DominanceResult Compare(Individual const& lhs, Individual const& rhs) noexcept {
        return DominanceCalculator{}(lhs, rhs);
    }
};
} // namespace

#endif
