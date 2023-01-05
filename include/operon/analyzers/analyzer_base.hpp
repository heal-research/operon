// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_ANALYZER_BASE_HPP
#define OPERON_ANALYZER_BASE_HPP

#include "operon/core/operator.hpp"

namespace Operon {
template <typename T>
class PopulationAnalyzerBase : public OperatorBase<double> {
public:
    virtual void Prepare(Operon::Span<T> pop) = 0;
};
} // namespace Operon

#endif
