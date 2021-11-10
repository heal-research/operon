// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef POLYGENIC_GENERATOR_HPP
#define POLYGENIC_GENERATOR_HPP

#include "core/operator.hpp"

namespace Operon {

class PolygenicOffspringGenerator : public OffspringGeneratorBase {
public:
    explicit PolygenicOffspringGenerator(EvaluatorBase& eval, CrossoverBase& cx, MutatorBase& mut, SelectorBase& femSel, SelectorBase& maleSel)
        : OffspringGeneratorBase(eval, cx, mut, femSel, maleSel)
    {
    }

    std::optional<Individual> operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, Operon::Span<Operon::Scalar> buf = Operon::Span<Operon::Scalar>{}) const override;

    void PolygenicSize(size_t value) { broodSize = value; }
    size_t PolygenicSize() const { return broodSize; }

private:
    size_t broodSize;
};

}
#endif
