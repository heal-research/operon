// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_DISTANCE_HPP
#define OPERON_DISTANCE_HPP

#include "types.hpp"
#include "operon/operon_export.hpp"

namespace Operon::Distance {
    auto OPERON_EXPORT Jaccard(Operon::Vector<Operon::Hash> const& lhs, Operon::Vector<Operon::Hash> const& rhs) noexcept -> double;
    auto OPERON_EXPORT SorensenDice(Operon::Vector<Operon::Hash> const& lhs, Operon::Vector<Operon::Hash> const& rhs) noexcept -> double;
} // namespace Operon::Distance

#endif
