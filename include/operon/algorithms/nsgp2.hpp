// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_NSGP2_HPP
#define OPERON_NSGP2_HPP

#include "operon/algorithms/mo_base.hpp"

namespace Operon {

class OPERON_EXPORT NSGP2 : public MultiObjectiveGABase {
protected:
    auto UpdateDistance(Operon::Span<Individual> pop) -> void override;

public:
    using MultiObjectiveGABase::MultiObjectiveGABase;
};

} // namespace Operon

#endif
