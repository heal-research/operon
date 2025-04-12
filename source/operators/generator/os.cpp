// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include "operon/operators/generator.hpp"
#include "operon/core/comparison.hpp"

namespace Operon {
    auto OffspringSelectionGenerator::operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, double pLocal, double pLamarck, Operon::Span<Operon::Scalar> buf) const -> std::optional<Individual>
    {
        auto res = OffspringGeneratorBase::Generate(random, pCrossover, pMutation, pLocal, pLamarck, buf);
        bool accept{false};
        std::vector<Operon::Scalar> q = res.Parent1->Fitness;
        if (res.Parent2) {
            for (size_t i = 0; i < q.size(); ++i) {
                auto f1 = (*res.Parent1)[i];
                auto f2 = (*res.Parent2)[i];
                q[i] = std::max(f1, f2) - static_cast<Operon::Scalar>(comparisonFactor_) * std::abs(f1 - f2);
            }
        }
        accept = Operon::ParetoDominance{}(res.Child->Fitness, q) != Dominance::Right;
        return accept ? res.Child : std::nullopt;
    }

} // namespace Operon
