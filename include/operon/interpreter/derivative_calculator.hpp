// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_DERIVATIVE_CALCULATOR
#define OPERON_DERIVATIVE_CALCULATOR

#include "interpreter.hpp"
#include "interpreter_new.hpp"

#include <ranges>

namespace Operon {

namespace detail {
    using T = detail::Array<Operon::Scalar>;

    struct RNode {
        T P;              // primal
        std::vector<T> D; // derivatives
    };

    inline auto Enumerate(auto const& nodes, auto i) {
        return Subtree<Node const>{nodes, i}.EnumerateIndices();
    }

    inline auto Indices(auto const& nodes, auto i) {
        return Subtree<Node const>{nodes, i}.Indices();
    }

    // derivatives
    template <Operon::NodeType N = Operon::NodeType::Add>
    struct Derivative {
        inline auto operator()(auto const& nodes, auto const& /*unused*/, auto& rnodes, auto i) {
            for (auto [k, j] : Enumerate(nodes, i)) {
                rnodes[i].D[k] = rnodes[j].P;
            }
        }
    };

    template<>
    struct Derivative<Operon::NodeType::Sub> {
        inline auto operator()(auto const& nodes, auto const& /*unused*/, auto& rnodes, auto i) {
            for (auto [k, j] : Enumerate(nodes, i)) {
                rnodes[i].D[k] = (k == 0 ? +1 : -1) * rnodes[j].P;
            }
        }
    };

    template<>
    struct Derivative<Operon::NodeType::Mul> {
        inline auto operator()(auto const& nodes, auto const& values, auto& rnodes, auto i) {
            auto const& n = nodes[i];
            auto idx{0};
            for (auto j : Indices(nodes, i)) {
                auto& d = rnodes[i].D[idx++];
                d = rnodes[j].P;
                for (auto k : Indices(nodes, i)) {
                    if (j == k) { continue; }
                    d *= values[k];
                }
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
                d[0] = -rnodes[j].P / values[j].square();
            } else if (n.Arity == 2) {
                auto j = i-1;
                auto k = j-(nodes[j].Length+1);
                d[0] = rnodes[j].P / values[k];
                d[1] = -rnodes[k].P * values[j] / values[k].square();
            } else {
                // TODO
                // f = g * (h * i * ...)
                // u = (h * i * ...)
                // f' = gu' + ug' / u^2
            }
        }
    };

    // binary symbols
    template<>
    struct Derivative<Operon::NodeType::Aq> {
        inline auto operator()(auto const& nodes, auto const& values, auto& rnodes, auto i) {
            auto j = i-1;
            auto k = j - (nodes[j].Length+1);
            rnodes[i].D[0] = rnodes[j].P * values[i] / values[j];
            //rnodes[i].D[1] = -rnodes[k].P * values[k] * values[j] * (values[i] / values[j]).pow(3);
            rnodes[i].D[1] = -rnodes[k].P * values[k] * values[i].pow(3) / values[j].square();
        }
    };

    template<>
    struct Derivative<Operon::NodeType::Pow> {
        inline auto operator()(auto const& nodes, auto const& values, auto& rnodes, auto i) {
            auto j = i-1;
            auto k = j - (nodes[j].Length+1);
            rnodes[i].D[0] = rnodes[j].P * values[j].pow(values[k]-1) * values[k];
            rnodes[i].D[1] = rnodes[k].P * values[i] * values[j].log();
        }
    };

    // unary symbols
    template<>
    struct Derivative<Operon::NodeType::Exp> {
        inline auto operator()(auto const& /*nodes*/, auto const& values, auto& rnodes, auto i) {
            rnodes[i].D[0] = rnodes[i-1].P * values[i];
        }
    };

    template<>
    struct Derivative<Operon::NodeType::Log> {
        inline auto operator()(auto const& /*nodes*/, auto const& values, auto& rnodes, auto i) {
            rnodes[i].D[0] = rnodes[i-1].P / values[i-1];
        }
    };

    template<>
    struct Derivative<Operon::NodeType::Logabs> {
        inline auto operator()(auto const& /*nodes*/, auto const& values, auto& rnodes, auto i) {
            rnodes[i].D[0] = rnodes[i-1].P * values[i-1].sign() / values[i-1].abs();
        }
    };

    template<>
    struct Derivative<Operon::NodeType::Log1p> {
        inline auto operator()(auto const& /*nodes*/, auto const& values, auto& rnodes, auto i) {
            rnodes[i].D[0] = rnodes[i-1].P / (values[i-1] + 1);
        }
    };

    template<>
    struct Derivative<Operon::NodeType::Sin> {
        inline auto operator()(auto const& /*nodes*/, auto const& values, auto& rnodes, auto i) {
            rnodes[i].D[0] = rnodes[i-1].P * values[i-1].cos();
        }
    };

    template<>
    struct Derivative<Operon::NodeType::Cos> {
        inline auto operator()(auto const& /*nodes*/, auto const& values, auto& rnodes, auto i) {
            rnodes[i].D[0] = -rnodes[i-1].P * values[i-1].sin();
        }
    };

    template<>
    struct Derivative<Operon::NodeType::Tan> {
        inline auto operator()(auto const& /*nodes*/, auto const& values, auto& rnodes, auto i) {
            rnodes[i].D[0] = rnodes[i-1].P * (values[i].square() + 1);
        }
    };

    template<>
    struct Derivative<Operon::NodeType::Tanh> {
        inline auto operator()(auto const& /*nodes*/, auto const& values, auto& rnodes, auto i) {
            rnodes[i].D[0] = rnodes[i-1].P * (1 - values[i].square());
        }
    };

    template<>
    struct Derivative<Operon::NodeType::Asin> {
        inline auto operator()(auto const& /*nodes*/, auto const& values, auto& rnodes, auto i) {
            rnodes[i].D[0] = rnodes[i-1].P / (1 - values[i-1].square()).sqrt();
        }
    };

    template<>
    struct Derivative<Operon::NodeType::Acos> {
        inline auto operator()(auto const& /*nodes*/, auto const& values, auto& rnodes, auto i) {
            rnodes[i].D[0] = -rnodes[i-1].P / (1 - values[i-1].square()).sqrt();
        }
    };

    template<>
    struct Derivative<Operon::NodeType::Atan> {
        inline auto operator()(auto const& /*nodes*/, auto const& values, auto& rnodes, auto i) {
            rnodes[i].D[0] = rnodes[i-1].P / (1 + values[i-1].square());
        }
    };

    template<>
    struct Derivative<Operon::NodeType::Sqrt> {
        inline auto operator()(auto const& /*nodes*/, auto const& values, auto& rnodes, auto i) {
            rnodes[i].D[0] = rnodes[i-1].P / (2 * values[i]);
        }
    };

    template<>
    struct Derivative<Operon::NodeType::Sqrtabs> {
        inline auto operator()(auto const& /*nodes*/, auto const& values, auto& rnodes, auto i) {
            rnodes[i].D[0] = rnodes[i-1].P * values[i-1].sign() / (2 * values[i]);
        }
    };

    auto ComputeDerivative(auto const& nodes, auto const& values, auto& rnodes, auto i) -> void {
        switch(nodes[i].Type) {
            case NodeType::Constant: { rnodes[i].P.setConstant(1); return; }
            case NodeType::Variable: { rnodes[i].P = values[i] / nodes[i].Value; return; }
            default: {
                rnodes[i].P.setConstant(nodes[i].Value);
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
            static_cast<Eigen::Index>(parameters.size())
        );

        std::vector<detail::RNode> rnodes(nodes.size());
        std::vector<detail::T> weights(nodes.size(), detail::T::Zero());

        auto row = 0;
        auto callback = [&](auto const& values) {
            // compute derivatives
            for (auto i = 0UL; i < nodes.size(); ++i) {
                if (!nodes[i].IsLeaf()) { rnodes[i].D.resize(nodes[i].Arity); }
                ComputeDerivative(nodes, values, rnodes, i);
            }

            // update weights
            auto const& root = nodes.back();
            if (root.IsLeaf()) {
                weights.back() = rnodes.back().P;
            } else {
                weights.back().setConstant(1);
                for (auto i = std::ssize(nodes)-1; i >= 0; --i) {
                    if (nodes[i].IsLeaf()) { continue; }
                    auto const& n = nodes[i];
                    for (decltype(i) j = i-1, k = 0; k < n.Arity; ++k, j -= (nodes[j].Length+1)) {
                        weights[j] += weights[i] * rnodes[i].D[k];
                    }
                }
            }

            for (auto i = 0, j = 0; i < std::ssize(nodes); ++i) {
                if (!nodes[i].Optimize) { continue; }
                auto const sz = std::min(values.size(), range.Size()-row);
                jacobian_.col(j++).segment(row, sz) = weights[i].segment(0, sz);
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
