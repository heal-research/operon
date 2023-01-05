// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_OPERATOR_HPP
#define OPERON_OPERATOR_HPP

#include "types.hpp"

namespace Operon {
template <typename Ret, typename... Args>
struct OperatorBase {
    using ReturnType = Ret;
    using ArgumentType = std::tuple<Args...>;
    // all operators take a random device (source of randomness) as the first parameter
    virtual auto operator()(Operon::RandomGenerator& /*random*/, Args... /*args*/) const -> Ret = 0;
    virtual ~OperatorBase() = default;

    OperatorBase() = default;
    OperatorBase(OperatorBase const& other) = default;
    OperatorBase(OperatorBase&& other) noexcept = default;

    auto operator=(OperatorBase const& other) -> OperatorBase& = default;
    auto operator=(OperatorBase&& other) noexcept -> OperatorBase& = default;
};
} // namespace Operon

#endif
