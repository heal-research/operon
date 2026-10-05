// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_STATISTICAL_CONCEPTS_HPP
#define OPERON_STATISTICAL_CONCEPTS_HPP

#include <Eigen/Core>
#include <concepts>

#include "operon/core/memory_view.hpp"
#include "operon/optimizer/fisher_information.hpp"

namespace Operon {

namespace Concepts {
    // Statistical-only contract. MDL callers supply arbitrary-stride Jacobian
    // storage and receive only the Fisher diagonal they consume.
    template <typename T>
    concept HasFisherDiagonal
        = requires(Operon::Span<Operon::Scalar const> prediction, Operon::ConstScalarMatrixView jacobian,
            Operon::Span<Operon::Scalar const> auxiliary, Operon::ScalarSpan diagonal) {
              {
                  T::ComputeFisherDiagonal(prediction, jacobian, auxiliary, diagonal)
              } -> std::same_as<tl::expected<void, Operon::FisherError>>;
          };
} // namespace Concepts

} // namespace Operon

#endif
