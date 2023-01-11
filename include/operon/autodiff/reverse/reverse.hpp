// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_AUTODIFF_REVERSE_HPP
#define OPERON_AUTODIFF_REVERSE_HPP

#include "operon/core/dataset.hpp"
#include "operon/core/tree.hpp"
#include "operon/interpreter/dispatch_table.hpp"
#include "derivatives.hpp"

namespace Operon::Autodiff::Reverse {
template<typename Interpreter>
class DerivativeCalculator {
    Interpreter const& interpreter_;

    auto ComputeDerivative(auto const& nodes, auto const& values, auto& rnodes, auto i) const -> void {
        switch(nodes[i].Type) {
        case NodeType::Add: { Derivative<NodeType::Add>{}(nodes, values, rnodes, i); return; }
        case NodeType::Sub: { Derivative<NodeType::Sub>{}(nodes, values, rnodes, i); return; }
        case NodeType::Mul: { Derivative<NodeType::Mul>{}(nodes, values, rnodes, i); return; }
        case NodeType::Div: { Derivative<NodeType::Div>{}(nodes, values, rnodes, i); return; }
        case NodeType::Pow: { Derivative<NodeType::Pow>{}(nodes, values, rnodes, i); return; }
        case NodeType::Aq: { Derivative<NodeType::Aq>{}(nodes, values, rnodes, i); return; }
        case NodeType::Exp: { Derivative<NodeType::Exp>{}(nodes, values, rnodes, i); return; }
        case NodeType::Log: { Derivative<NodeType::Log>{}(nodes, values, rnodes, i); return; }
        case NodeType::Logabs: { Derivative<NodeType::Logabs>{}(nodes, values, rnodes, i); return; }
        case NodeType::Log1p: { Derivative<NodeType::Log1p>{}(nodes, values, rnodes, i); return; }
        case NodeType::Sin: { Derivative<NodeType::Sin>{}(nodes, values, rnodes, i); return; }
        case NodeType::Cos: { Derivative<NodeType::Cos>{}(nodes, values, rnodes, i); return; }
        case NodeType::Tan: { Derivative<NodeType::Tan>{}(nodes, values, rnodes, i); return; }
        case NodeType::Tanh: { Derivative<NodeType::Tanh>{}(nodes, values, rnodes, i); return; }
        case NodeType::Asin: { Derivative<NodeType::Asin>{}(nodes, values, rnodes, i); return; }
        case NodeType::Acos: { Derivative<NodeType::Acos>{}(nodes, values, rnodes, i); return; }
        case NodeType::Atan: { Derivative<NodeType::Atan>{}(nodes, values, rnodes, i); return; }
        case NodeType::Sqrt: { Derivative<NodeType::Sqrt>{}(nodes, values, rnodes, i); return; }
        case NodeType::Sqrtabs: { Derivative<NodeType::Sqrtabs>{}(nodes, values, rnodes, i); return; }
        default: { throw std::runtime_error("unsupported node type"); }
        }
    }

    struct RNode {
        using Array = Dispatch::Array<Operon::Scalar>;
        Array P;              // primal
        std::vector<Array> D; // derivatives
    };

public:
    explicit DerivativeCalculator(Interpreter const& interpreter)
        : interpreter_(interpreter) { }

    template<int StorageOrder = Eigen::ColMajor>
    auto operator()(Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Span<Operon::Scalar const> coeff, Operon::Range const range) const {
        Eigen::Array<Operon::Scalar, -1, -1, StorageOrder> jacobian(static_cast<Eigen::Index>(range.Size()), std::ssize(coeff));
        this->operator()<StorageOrder>(tree, dataset, coeff, range, jacobian.data());
        return jacobian;
    }

    template<int StorageOrder = Eigen::ColMajor>
    auto operator()(Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Span<Operon::Scalar const> coeff, Operon::Range const range, Operon::Scalar* jacobian) const
    {
        auto const& nodes = tree.Nodes();
        auto const np{ static_cast<int>(coeff.size()) }; // number of parameters
        auto const nr{ static_cast<int>(range.Size()) }; // number of residuals
        Eigen::Map<Eigen::Array<Operon::Scalar, -1, -1, StorageOrder>> jac(jacobian, nr, np);
        jac.setConstant(0);

        std::vector<RNode> rnodes(nodes.size());
        std::vector<typename RNode::Array> weights(nodes.size());

        auto constexpr S{ static_cast<int>(Dispatch::BatchSize<Operon::Scalar>) };

        for (auto i{0UL}; i < nodes.size(); ++i) {
            rnodes[i].P.setConstant(1);
            if (nodes[i].Arity > 0) {
                rnodes[i].D.resize(nodes[i].Arity);
            }
        }

        std::vector<Operon::Scalar> result(nr);

        auto callback = [&](auto const& values, auto row) {
            auto const len = std::min(S, nr - row);

            // compute derivatives
            for (auto i = 0UL, j = 0UL; i < nodes.size(); ++i) {
                weights[i].setConstant(0);
                auto const& n{ nodes[i] };
                if (n.IsVariable()) {
                    auto s { dataset.GetValues(n.HashValue).subspan(row, len) };
                    rnodes[i].P.segment(0, len) = Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const>(s.data(), std::ssize(s)); 
                } else if (n.Arity > 0) {
                    ComputeDerivative(nodes, values, rnodes, i);
                }
            }

            // update weights
            weights.back() = rnodes.back().P;
            for (auto i = std::ssize(nodes)-1, j = jac.cols(); i >= 0; --i) {
                if (nodes[i].IsLeaf()) { continue; }
                for (auto [k, j] : Enumerate(nodes, static_cast<std::size_t>(i))) {
                    weights[j].segment(0, len) += weights[i].segment(0, len) * rnodes[i].D[k].segment(0, len);
                }
            }

            for (auto i = 0, j = 0; i < std::ssize(nodes); ++i) {
                if (!nodes[i].Optimize) { continue; }
                jac.col(j++).segment(row, len) = weights[i].segment(0, len);
            }
        };

        interpreter_.template operator()<Operon::Scalar>(tree, dataset, range, result, coeff, callback);
    }

    [[nodiscard]] auto GetInterpreter() const -> Interpreter const& { return interpreter_; }
};
} // namespace Operon::Autodiff::Reverse

#endif
