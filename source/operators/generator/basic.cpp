// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include "operon/operators/generator.hpp"

namespace Operon {
    auto BasicOffspringGenerator::operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, double pLocal, Operon::Span<Operon::Scalar> buf) const -> std::optional<Individual>
    {
        auto res = OffspringGeneratorBase::Generate(random, pCrossover, pMutation, pLocal, buf);
        return res.Child;
    }
} // namespace Operon
