// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_DERIVATIVE_CALCULATOR
#define OPERON_DERIVATIVE_CALCULATOR

#include "interpreter.hpp"
#include "interpreter_new.hpp"

namespace Operon {

namespace detail {
    using T = detail::Array<Operon::Scalar>;

    struct RNode {
        T A; // adjoint
        T W; // weight
        std::vector<T> D;
    };

    auto Indices(auto const& nodes, auto i) {
        std::vector<decltype(i)> indices;
        indices.reserve(nodes[i].Arity);
        for (decltype(i) j = i-1, k = 0; k < nodes[i].Arity; ++k, j -= nodes[j].Length+1) {
            indices.push_back(j);
        }
        return indices;
    };

    // derivatives
    template <Operon::NodeType N = Operon::NodeType::Add>
    struct Derivative {
        inline auto operator()(auto const& nodes, auto const& /*unused*/, auto& rnodes, auto i) {
            auto const& n = nodes[i];
            rnodes[i].A.setConstant(0);

            for (decltype(i) j = i-1, k = 0; k < n.Arity; ++k, j -= (nodes[j].Length+1)) {
                auto& d = rnodes[i].D[k];
                auto const x = nodes[j].IsLeaf();
                if (x) { d = rnodes[j].A; }
                else { d.setConstant(1); }
                rnodes[i].A += rnodes[j].A;
            }
        }
    };

    template<>
    struct Derivative<Operon::NodeType::Sub> {
        inline auto operator()(auto const& nodes, auto const& /*unused*/, auto& rnodes, auto i) {
            auto const& n = nodes[i];
            rnodes[i].A.setConstant(0);

            for (decltype(i) j = i-1, k = 0; k < n.Arity; ++k, j -= (nodes[j].Length+1)) {
                auto sign = k == 0 ? +1 : -1;
                auto& d = rnodes[i].D[k];
                auto const x = nodes[j].IsLeaf();
                if (x) { d = rnodes[j].A * sign; }
                else { d.setConstant(sign); }
                rnodes[i].A += rnodes[j].A;
            }
        }
    };

    template<>
    struct Derivative<Operon::NodeType::Mul> {
        inline auto operator()(auto const& nodes, auto const& values, auto& rnodes, auto i) {
            auto const& n = nodes[i];
            auto indices{Indices(nodes, i)};
            auto idx{0};
            for (auto j : indices) {
                auto& d = rnodes[i].D[idx++];
                auto const x = nodes[j].IsLeaf();
                if (x) { d = rnodes[j].A; }
                else { d.setConstant(1); }
                for (auto k : indices) {
                    if (j == k) { continue; }
                    d *= values[k];
                }
                rnodes[i].A += d;
            }
        }
    };

    template<>
    struct Derivative<Operon::NodeType::Div> {
        inline auto operator()(auto const& nodes, auto const& values, auto& rnodes, auto i) {
            auto const& n = nodes[i];

            if (n.Arity > 2) {
                throw std::runtime_error("derivative of division with more than 2 children is not supported");
            }

            auto& d = rnodes[i].D;

            if (n.Arity == 1) {
                auto j = i-1;
                if (nodes[j].IsLeaf()) { d[0] = -rnodes[j].A / values[j].square(); }
                else { d[0] = -values[j].square().inverse(); }
                rnodes[i].A = d[0];
            } else if (n.Arity == 2) {
                auto j = i-1;
                auto k = j-(nodes[j].Length+1);
                if (nodes[j].IsLeaf()) { d[0] = rnodes[j].A / values[k]; }
                else { d[0] = values[k].inverse(); }

                if (nodes[k].IsLeaf()) { d[1] = -rnodes[k].A * values[j] / values[k].square(); }
                else { d[1] = -values[j] / values[k].square(); }
                rnodes[i].A = d[0] + d[1];
            } else {
                // TODO
                // f = g * (h * i * ...)
                // u = (h * i * ...)
                // f' = gu' + ug' / u^2
            }
        }
    };

    template<>
    struct Derivative<Operon::NodeType::Exp> {
        inline auto operator()(auto const& nodes, auto const& values, auto& rnodes, auto i) {
            auto const& n = nodes[i];
            auto j = i-1;
            auto const x = nodes[j].IsLeaf();
            auto& d = rnodes[i].D[0];
            if (x) { d = rnodes[j].A * values[i]; }
            else { d = values[i]; }
            rnodes[i].A = d;
        }
    };

    template<>
    struct Derivative<Operon::NodeType::Log> {
        inline auto operator()(auto const& nodes, auto const& values, auto& rnodes, auto i) {
            auto const& n = nodes[i];
            auto j = i-1;
            auto const x = nodes[j].IsLeaf();
            auto& d = rnodes[i].D[0];
            if (x) { d = rnodes[j].A * values[j].inverse(); }
            else { d = values[j].inverse(); }
            rnodes[i].A = d;
        }
    };

    template<>
    struct Derivative<Operon::NodeType::Sin> {
        inline auto operator()(auto const& nodes, auto const& values, auto& rnodes, auto i) {
            auto const x = nodes[i-1].IsLeaf();
            if (x) { rnodes[i].D[0] = rnodes[i-1].A * values[i-1].cos(); }
            else { rnodes[i].D[0] = values[i-1].cos(); }
            rnodes[i].A = rnodes[i].D[0];
        }
    };

    template<>
    struct Derivative<Operon::NodeType::Cos> {
        inline auto operator()(auto const& nodes, auto const& values, auto& rnodes, auto i) {
            auto const x = nodes[i-1].IsLeaf();
            if (x) { rnodes[i].D[0] = rnodes[i-1].A * -values[i-1].sin(); }
            else { rnodes[i].D[0] = -values[i-1].sin(); }
            rnodes[i].A = rnodes[i].D[0];
        }
    };

    template<>
    struct Derivative<Operon::NodeType::Variable> {
        inline auto operator()(auto const& nodes, auto const& values, auto& rnodes, auto i) {
            // the values corresponding to this node already include the multiplication
            // with the variable weight, this must be reversed here, hence the division
            rnodes[i].A = values[i] / nodes[i].Value;
        }
    };

    template<>
    struct Derivative<Operon::NodeType::Constant> {
        inline auto operator()(auto const& /*unused*/, auto const& /*unused*/, auto& rnodes, auto i) {
            rnodes[i].A.setConstant(1);
        }
    };

    auto ComputeDerivative(auto const& nodes, auto const& values, auto& rnodes, auto i) -> void {
        switch(nodes[i].Type) {
            case NodeType::Add: { Derivative<NodeType::Add>{}(nodes, values, rnodes, i); return; }
            case NodeType::Sub: { Derivative<NodeType::Sub>{}(nodes, values, rnodes, i); return; }
            case NodeType::Mul: { Derivative<NodeType::Mul>{}(nodes, values, rnodes, i); return; }
            case NodeType::Div: { Derivative<NodeType::Div>{}(nodes, values, rnodes, i); return; }
            case NodeType::Exp: { Derivative<NodeType::Exp>{}(nodes, values, rnodes, i); return; }
            case NodeType::Log: { Derivative<NodeType::Log>{}(nodes, values, rnodes, i); return; }
            case NodeType::Sin: { Derivative<NodeType::Sin>{}(nodes, values, rnodes, i); return; }
            case NodeType::Cos: { Derivative<NodeType::Cos>{}(nodes, values, rnodes, i); return; }
            case NodeType::Constant: { Derivative<NodeType::Constant>{}(nodes, values, rnodes, i); return; }
            case NodeType::Variable: { Derivative<NodeType::Variable>{}(nodes, values, rnodes, i); return; }
            default: { throw std::runtime_error("unsupported node type"); }
        }
    }
} // namespace detail

template <typename Interpreter>
struct DerivativeCalculator {
    explicit DerivativeCalculator(Interpreter const& interpreter)
        : interpreter_(interpreter)
    {
    }

    auto operator()(Operon::Span<Operon::Scalar const> parameters)
    {
        auto const& tree = interpreter_.Tree;
        auto const& nodes = tree.Nodes();
        auto const range = interpreter_.Range;

        jacobian_.resize(
            static_cast<Eigen::Index>(range.Size()),
            static_cast<Eigen::Index>(parameters.size()));

        Operon::Vector<detail::RNode> rnodes(nodes.size());

        auto row = 0;
        auto callback = [&](auto const& values) {
            // compute derivatives
            for (auto i = 0; i < std::ssize(nodes); ++i) {
                if (!nodes[i].IsLeaf()) { rnodes[i].D.resize(nodes[i].Arity); }
                ComputeDerivative(nodes, values, rnodes, i);

                //fmt::print("{}: d_{} = {}\n", nodes[i].Name(), i, rnodes[i].A[0]);
                //for (auto j = 0; j < rnodes[i].D.size(); ++j) {
                //    fmt::print("{}: d_{}{} = {}\n", nodes[i].Name(), i, j, rnodes[i].D[j][0]);
                //}
            }

            // update weights
            auto const& root = nodes.back();
            if (root.IsLeaf()) {
                rnodes.back().W = rnodes.back().A;
            } else {
                rnodes.back().W.setConstant(1);
                for (auto i = std::ssize(nodes)-1; i >= 0; --i) {
                    if (nodes[i].IsLeaf()) { continue; }
                    auto const& n = nodes[i];
                    for (decltype(i) j = i-1, k = 0; k < n.Arity; ++k, j -= (nodes[j].Length+1)) {
                        rnodes[j].W += rnodes[i].W * rnodes[i].D[k];
                    }
                }
            }

            for (auto i = 0, j = 0; i < std::ssize(nodes); ++i) {
                if (!nodes[i].Optimize) { continue; }
                auto const sz = std::min(values.size(), range.Size()-row);
                //ENSURE(j < jacobian_.cols());
                //ENSURE(row+sz < jacobian_.rows());
                jacobian_.col(j++).segment(row, sz) = rnodes[i].W.segment(0, sz);
            }
            row += values.front().size();
        };

        interpreter_(parameters, callback);
    }

    auto Jacobian() const { return jacobian_; }

private:
    // internal
    Interpreter const& interpreter_;
    Eigen::Array<Operon::Scalar, -1, -1> jacobian_;
};
} // namespace Operon

#endif
