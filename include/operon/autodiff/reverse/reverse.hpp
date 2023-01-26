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

    auto ComputeDerivative(auto const& nodes, auto const& primal, auto& rnodes, auto i) const -> void {
        switch(nodes[i].Type) {
        case NodeType::Add: { Derivative<NodeType::Add>{}(nodes, primal, rnodes, i); return; }
        case NodeType::Sub: { Derivative<NodeType::Sub>{}(nodes, primal, rnodes, i); return; }
        case NodeType::Mul: { Derivative<NodeType::Mul>{}(nodes, primal, rnodes, i); return; }
        case NodeType::Div: { Derivative<NodeType::Div>{}(nodes, primal, rnodes, i); return; }
        case NodeType::Pow: { Derivative<NodeType::Pow>{}(nodes, primal, rnodes, i); return; }
        case NodeType::Aq: { Derivative<NodeType::Aq>{}(nodes, primal, rnodes, i); return; }
        case NodeType::Exp: { Derivative<NodeType::Exp>{}(nodes, primal, rnodes, i); return; }
        case NodeType::Log: { Derivative<NodeType::Log>{}(nodes, primal, rnodes, i); return; }
        case NodeType::Logabs: { Derivative<NodeType::Logabs>{}(nodes, primal, rnodes, i); return; }
        case NodeType::Log1p: { Derivative<NodeType::Log1p>{}(nodes, primal, rnodes, i); return; }
        case NodeType::Sin: { Derivative<NodeType::Sin>{}(nodes, primal, rnodes, i); return; }
        case NodeType::Cos: { Derivative<NodeType::Cos>{}(nodes, primal, rnodes, i); return; }
        case NodeType::Tan: { Derivative<NodeType::Tan>{}(nodes, primal, rnodes, i); return; }
        case NodeType::Tanh: { Derivative<NodeType::Tanh>{}(nodes, primal, rnodes, i); return; }
        case NodeType::Asin: { Derivative<NodeType::Asin>{}(nodes, primal, rnodes, i); return; }
        case NodeType::Acos: { Derivative<NodeType::Acos>{}(nodes, primal, rnodes, i); return; }
        case NodeType::Atan: { Derivative<NodeType::Atan>{}(nodes, primal, rnodes, i); return; }
        case NodeType::Sqrt: { Derivative<NodeType::Sqrt>{}(nodes, primal, rnodes, i); return; }
        case NodeType::Sqrtabs: { Derivative<NodeType::Sqrtabs>{}(nodes, primal, rnodes, i); return; }
        case NodeType::Cbrt: { Derivative<NodeType::Cbrt>{}(nodes, primal, rnodes, i); return; }
        default: { throw std::runtime_error("unsupported node type"); }
        }
    }

    static auto constexpr BatchSize{ Dispatch::BatchSize<Operon::Scalar> };

    // a node of the reverse computational graph
    class RNode {
        Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> w_{nullptr, BatchSize, 1};
        Eigen::Array<Operon::Scalar, BatchSize, -1> d_;
        int64_t len_{BatchSize};

    public:
        auto Length() const { return len_; }
        auto SetLength(auto len) { len_ = len; }
        auto SetWeight(auto const* ptr) {
            new (&w_) decltype(w_)(ptr, len_, 1);
        }
        auto Resize(auto ncol) { d_.resize(BatchSize, ncol); }

        auto W() { return w_.head(len_); }
        auto D(auto i) { return d_.col(i).head(len_); }
    };

public:
    explicit DerivativeCalculator(Interpreter const& interpreter)
        : interpreter_(interpreter) { }

    template<int StorageOrder = Eigen::ColMajor>
    auto operator()(Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Range const range, Operon::Span<Operon::Scalar const> coeff) const {
        Eigen::Array<Operon::Scalar, -1, -1, StorageOrder> jacobian(static_cast<Eigen::Index>(range.Size()), std::ssize(coeff));
        this->operator()<StorageOrder>(tree, dataset, range, coeff, {/* empty */}, { jacobian.data(), static_cast<std::size_t>(jacobian.size()) });
        return jacobian;
    }

    template<int StorageOrder = Eigen::ColMajor>
    auto operator()(Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Range const range, Operon::Span<Operon::Scalar const> coeff, Operon::Span<Operon::Scalar> jacobian) const {
        this->operator()<StorageOrder>(tree, dataset, range, coeff, {/* empty */}, jacobian);
    }

    template<int StorageOrder = Eigen::ColMajor>
    auto operator()(Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Range const range, Operon::Span<Operon::Scalar const> coeff, Operon::Span<Operon::Scalar> residual, Operon::Span<Operon::Scalar> jacobian) const
    {
        auto const& nodes = tree.Nodes();
        auto const nn{ std::ssize(nodes) };
        auto const np{ static_cast<int>(coeff.size()) }; // number of parameters
        auto const nr{ static_cast<int>(range.Size()) }; // number of residuals
        Eigen::Map<Eigen::Array<Operon::Scalar, -1, -1, StorageOrder>> jac(jacobian.data(), nr, np);
        jac.setConstant(0);

        auto constexpr S{ static_cast<int>(Dispatch::BatchSize<Operon::Scalar>) };
        Eigen::Array<Operon::Scalar, S, 1> one = decltype(one)::Ones(S, 1);

        std::vector<RNode> rnodes(nn);
        for (auto i = 0; i < nn; ++i) {
            rnodes[i].SetWeight(one.data());
            if (nodes[i].Arity > 0) {
                rnodes[i].Resize(nodes[i].Arity);
            }
        }

        Dispatch::Matrix<Operon::Scalar> adj(S, nn);

        auto reverse = [&](auto const& primal, auto row) {
            auto const len = std::min(S, nr - row);

            // forward pass - populate intermediate primal (weights and partial derivatives)
            for (auto i = 0; i < nn; ++i) {
                rnodes[i].SetLength(len);
                adj.col(i).setConstant(0);
                auto const& n = nodes[i];
                if (n.IsVariable()) {
                    auto s { dataset.GetValues(n.HashValue).subspan(row, len) };
                    rnodes[i].SetWeight(s.data());
                } else if (n.Arity > 0) {
                    ComputeDerivative(nodes, primal.topRows(len), rnodes, static_cast<std::size_t>(i));
                }
            }

            // backward pass - propagate adjoints and fill the jacobian
            adj.col(nn-1) = rnodes.back().W();
            auto c{np};
            for (auto i = nn-1; i >= 0; --i) {
                if (nodes[i].Optimize) { jac.col(--c).segment(row, len) = adj.col(i).head(len); }
                if (nodes[i].IsLeaf()) { continue; }
                for (auto [k, j] : Enumerate(nodes, static_cast<std::size_t>(i))) {
                    adj.col(static_cast<int>(j)).head(len) += adj.col(i).head(len) * rnodes[i].D(k);
                }
            }
        };

        interpreter_.template operator()<Operon::Scalar>(tree, dataset, range, residual, coeff, reverse);
    }

    [[nodiscard]] auto GetInterpreter() const -> Interpreter const& { return interpreter_; }
};
} // namespace Operon::Autodiff::Reverse

#endif
