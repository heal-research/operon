// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_ANALYZER_BASE_HPP
#define OPERON_ANALYZER_BASE_HPP

#include "operon/core/operators.hpp"

namespace Operon {
template <typename T>
class PopulationAnalyzerBase : public OperatorBase<double> {
public:
    virtual void Prepare(Operon::Span<const T> pop) = 0;
};
} // namespace Operon

#endif
