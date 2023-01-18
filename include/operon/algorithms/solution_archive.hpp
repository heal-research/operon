// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_SOLUTION_ARCHIVE_HPP
#define OPERON_SOLUTION_ARCHIVE_HPP

#include "operon/core/individual.hpp"
#include "operon/operators/non_dominated_sorter.hpp"

namespace Operon {
class SolutionArchive {
public:
    auto Insert(Operon::Individual const& individual) -> bool;
    auto Insert(Operon::Span<Operon::Individual const> individuals) -> int64_t;

    [[nodiscard]] auto Solutions() const { return Operon::Span<Operon::Individual const> { archive_ }; }
    auto Clear() { archive_.clear(); }
private:
    Operon::Scalar eps_{};
    std::vector<Operon::Individual> archive_;
};
} // namespace Operon
#endif
