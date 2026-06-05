// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_NSGP_SMS_HPP
#define OPERON_NSGP_SMS_HPP

#include "operon/algorithms/mo_base.hpp"

namespace Operon {

// NSGA-II generational structure with hypervolume contribution as the
// secondary diversity criterion (instead of crowding distance).
// Currently restricted to 2 objectives.
class OPERON_EXPORT NSGPSMS : public MultiObjectiveGABase {
protected:
    auto UpdateDistance(Operon::Span<Individual> pop) -> void override;

public:
    using MultiObjectiveGABase::MultiObjectiveGABase;
};

} // namespace Operon

#endif
