// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/operators/generator.hpp"

namespace Operon {
    auto BasicOffspringGenerator::operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, double pLocal, double pLamarck, Operon::Span<Operon::Scalar> buf) const -> std::optional<Individual>
    {
        auto res = OffspringGeneratorBase::Generate(random, pCrossover, pMutation, pLocal, pLamarck, buf);
        return res.Child;
    }
} // namespace Operon
