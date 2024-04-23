// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <cstdint>
#include <algorithm>
#include <iterator>
#include <span>
#include <vector>

#include "operon/algorithms/solution_archive.hpp"
#include "operon/core/comparison.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/types.hpp"

namespace Operon {

auto SolutionArchive::Insert(Operon::Individual const& individual) -> bool
{
    auto const& y = individual;
    Operon::ParetoDominance dom{};

    // check if the current individual is dominated by any solution in the archive_
    // or if it is equal to any solution in the archive_
    if (std::any_of(archive_.begin(), archive_.end(), [&](auto const& x) {
        auto res = dom(x.Fitness, y.Fitness, eps_);
        return res == Dominance::Left || res == Dominance::Equal;
    })) { return false ; } // individual is dominated by or equal to an existing solution

    // remove solutions that are dominated by the current individual
    std::erase_if(archive_, [&](auto const& x) { return dom(x.Fitness, y.Fitness, eps_) == Dominance::Right; });
    archive_.push_back(y);
    return true;
}

auto SolutionArchive::Insert(Operon::Span<Operon::Individual const> individuals) -> int64_t {
    auto const s { std::ssize(archive_) };
    for (auto const& x : individuals) { Insert(x); }
    if (!archive_.empty()) {
        auto const m{ archive_.front().Size() };
        for (auto k = 0UL; k < m; ++k) {
            auto cmp = [k](auto const& a, auto const& b) { return a[k] < b[k]; };
            std::stable_sort(archive_.begin(), archive_.end(), cmp);
        }
    }
    return std::ssize(archive_) - s;
}
} // namespace Operon
