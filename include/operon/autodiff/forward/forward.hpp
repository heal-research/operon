// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_AUTODIFF_FORWARD_HPP
#define OPERON_AUTODIFF_FORWARD_HPP

#include "operon/core/dataset.hpp"
#include "operon/core/tree.hpp"
#include "dual.hpp"

namespace Operon::Autodiff::Forward {
template<typename Interpreter>
class DerivativeCalculator {
    Interpreter const& interpreter_;

public:
    explicit DerivativeCalculator(Interpreter const& interpreter)
        : interpreter_(interpreter) { }

    template<int StorageOrder = Eigen::ColMajor>
    auto operator()(Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Range const range, Operon::Span<Operon::Scalar const> coeff) const {
        Eigen::Array<Operon::Scalar, -1, -1, StorageOrder> jacobian(static_cast<int>(range.Size()), std::ssize(coeff));
        this->operator()<StorageOrder>(tree, dataset, range, coeff, {/* empty */}, { jacobian.data(), static_cast<std::size_t>(jacobian.size()) });
        return jacobian;
    }

    template<int StorageOrder = Eigen::ColMajor>
    auto operator()(Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Range const range, Operon::Span<Operon::Scalar const> coeff, Operon::Span<Operon::Scalar> jacobian) const{
        this->operator()<StorageOrder>(tree, dataset, range, coeff, {/* empty */}, jacobian);
    }

    template<int StorageOrder = Eigen::ColMajor>
    auto operator()(Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Range const range, Operon::Span<Operon::Scalar const> coeff, Operon::Span<Operon::Scalar> residual, Operon::Span<Operon::Scalar> jacobian) const
    {
        auto const& nodes = tree.Nodes();
        std::vector<Dual> inputs(coeff.size());
        std::vector<Dual> outputs(range.Size());
        ENSURE(jacobian.size() == inputs.size() * outputs.size());
        Eigen::Map<Eigen::Array<Operon::Scalar, -1, -1, StorageOrder>> jac(jacobian.data(), std::ssize(outputs), std::ssize(inputs));
        jac.setConstant(0);

        for (auto i{0UL}; i < inputs.size(); ++i) {
            inputs[i].a = coeff[i];
            inputs[i].v.setZero();
        }

        auto constexpr d{ Dual::DIMENSION };
        for (auto s = 0U; s < inputs.size(); s += d) {
            auto r = std::min(static_cast<uint32_t>(inputs.size()), s + d); // remaining parameters

            for (auto i = s; i < r; ++i) {
                inputs[i].v[i - s] = 1.0;
            }

            interpreter_.template operator()<Dual>(tree, dataset, range, outputs, inputs);

            for (auto i = s; i < r; ++i) {
                inputs[i].v[i - s] = 0.0;
            }

            // fill in the jacobian trying to exploit its layout for efficiency
            if constexpr (StorageOrder == Eigen::ColMajor) {
                for (auto i = s; i < r; ++i) {
                    std::transform(outputs.cbegin(), outputs.cend(), jac.col(i).data(), [&](auto const& jet) { return jet.v[i - s]; });
                }
            } else {
                for (auto i = 0; i < outputs.size(); ++i) {
                    std::copy_n(outputs[i].v.data(), r - s, jac.row(i).data() + s);
                }
            }
        }

        // copy the residual over
        if (residual.size() == outputs.size()) {
            std::transform(outputs.cbegin(), outputs.cend(), residual.begin(), [&](auto const& jet) { return jet.a; });
        }
    }

    [[nodiscard]] auto GetInterpreter() const -> Interpreter const& { return interpreter_; }
};
} // namespace Operon::Autodiff::Forward

#endif
