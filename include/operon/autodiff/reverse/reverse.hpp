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
    bool print_{false};

    auto ComputeDerivative(auto const& nodes, auto const primal, auto trace, auto weights, auto i, auto j) const {
        switch(nodes[i].Type) {
        case NodeType::Add:     { Derivative<NodeType::Add>{}(nodes, primal, trace, weights, i, j); return; }
        case NodeType::Sub:     { Derivative<NodeType::Sub>{}(nodes, primal, trace, weights, i, j); return; }
        case NodeType::Mul:     { Derivative<NodeType::Mul>{}(nodes, primal, trace, weights, i, j); return; }
        case NodeType::Div:     { Derivative<NodeType::Div>{}(nodes, primal, trace, weights, i, j); return; }
        case NodeType::Pow:     { Derivative<NodeType::Pow>{}(nodes, primal, trace, weights, i, j); return; }
        case NodeType::Aq:      { Derivative<NodeType::Aq>{}(nodes, primal, trace, weights, i, j); return; }
        case NodeType::Square:  { Derivative<NodeType::Square>{}(nodes, primal, trace, weights, i, j); return; }
        case NodeType::Abs:     { Derivative<NodeType::Abs>{}(nodes, primal, trace, weights, i, j); return; }
        case NodeType::Exp:     { Derivative<NodeType::Exp>{}(nodes, primal, trace, weights, i, j); return; }
        case NodeType::Log:     { Derivative<NodeType::Log>{}(nodes, primal, trace, weights, i, j); return; }
        case NodeType::Logabs:  { Derivative<NodeType::Logabs>{}(nodes, primal, trace, weights, i, j); return; }
        case NodeType::Log1p:   { Derivative<NodeType::Log1p>{}(nodes, primal, trace, weights, i, j); return; }
        case NodeType::Sin:     { Derivative<NodeType::Sin>{}(nodes, primal, trace, weights, i, j); return; }
        case NodeType::Cos:     { Derivative<NodeType::Cos>{}(nodes, primal, trace, weights, i, j); return; }
        case NodeType::Tan:     { Derivative<NodeType::Tan>{}(nodes, primal, trace, weights, i, j); return; }
        case NodeType::Tanh:    { Derivative<NodeType::Tanh>{}(nodes, primal, trace, weights, i, j); return; }
        case NodeType::Asin:    { Derivative<NodeType::Asin>{}(nodes, primal, trace, weights, i, j); return; }
        case NodeType::Acos:    { Derivative<NodeType::Acos>{}(nodes, primal, trace, weights, i, j); return; }
        case NodeType::Atan:    { Derivative<NodeType::Atan>{}(nodes, primal, trace, weights, i, j); return; }
        case NodeType::Sqrt:    { Derivative<NodeType::Sqrt>{}(nodes, primal, trace, weights, i, j); return; }
        case NodeType::Sqrtabs: { Derivative<NodeType::Sqrtabs>{}(nodes, primal, trace, weights, i, j); return; }
        case NodeType::Cbrt:    { Derivative<NodeType::Cbrt>{}(nodes, primal, trace, weights, i, j); return; }
        default:                { throw std::runtime_error("unsupported node type"); }
        }
    }

    auto WriteTrace(auto const& nodes, auto const& dataset, auto const& trace, auto const& adjoint) const // NOLINT
    {
        auto const nn{ std::ssize(nodes) };
        std::string str;
        auto out{ std::back_inserter(str) };
        fmt::format_to(out, "{}", "strict digraph reverse_graph {\n");

        for (auto i = 0; i < nn; ++i) {
            auto const& n = nodes[i];
            auto name = n.Name();
            if (n.IsConstant()) {
                name = fmt::format("{:.3f}", n.Value);
            } else if (n.IsVariable()) {
                name = dataset.GetVariable(n.HashValue).value().Name;
            }
            fmt::format_to(out, "\tn{} [label=\"{}\"];\n", i, name);
        }

        for (auto i = 0; i < nn; ++i) {
            auto const& n = nodes[i];
            if (n.IsLeaf()) { continue; }

            for(auto [k, j] : Enumerate(nodes, i)) {
                fmt::format_to(out, "\tn{} -> n{} [label=\"{:.3f}\"]\n", j, i, adjoint.row(0)(j));
            }
        }

        fmt::format_to(out, "{}", "}");
        return str;
    }

public:
    explicit DerivativeCalculator(Interpreter const& interpreter, bool print = false)
        : interpreter_(interpreter), print_(print) { }

    template<int StorageOrder = Eigen::ColMajor>
    auto operator()(Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Range const range, Operon::Span<Operon::Scalar const> coeff) const -> Eigen::Array<Operon::Scalar, -1, -1> {
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
        Dispatch::Matrix<Operon::Scalar> adj(S, nn);
        auto ncol = std::transform_reduce(nodes.begin(), nodes.end(), 0L, std::plus{}, [](auto const& n) { return n.Arity; });
        Dispatch::Matrix<Operon::Scalar> partial(S, ncol);

        struct Indexer {
            int T; // trace column index
            int J; // jacobian column index
        };

        std::vector<Operon::Scalar> cof(nn);
        std::vector<Indexer> idx(nn);
        for (auto i = 0, j = 0, k = 0; i < nn; ++i) {
            idx[i] = { j, k };
            cof[i] = nodes[i].Optimize ? coeff[k] : nodes[i].Value;
            j += nodes[i].Arity;
            k += nodes[i].Optimize;
        }
        std::span<Operon::Scalar const> w{cof.data(), cof.size()};

        auto reverse = [&](auto const& primal, auto row) {
            auto const len = std::min(S, nr - row);

            // forward pass - compute partial derivatives
            for (auto i = 0; i < nn; ++i) {
                adj.col(i).setConstant(0);
                if (nodes[i].Arity > 0) {
                    ComputeDerivative(nodes, primal.topRows(len), partial.topRows(len), w, i, idx[i].T);
                }
            }

            // backward pass - propagate adjoints
            adj.col(nn-1).setConstant(1);
            for (auto i = nn-1; i >= 0; --i) {
                if (nodes[i].Optimize) {
                    jac.col(idx[i].J).segment(row, len) = adj.col(i).head(len) * primal.col(i).head(len) / cof[i];
                }
                if (nodes[i].IsLeaf()) { continue; }
                for (auto [k, j] : Enumerate(nodes, i)) {
                    auto const l { static_cast<int64_t>(k) + idx[i].T };
                    adj.col(static_cast<int64_t>(j)).head(len) += adj.col(i).head(len) * partial.col(l).head(len) * cof[i];
                }
            }
        };

        interpreter_(tree, dataset, range, residual, coeff, reverse);

        if (print_) {
            auto str = WriteTrace(nodes, dataset, partial, adj);
            fmt::print("{}\n", str);
        }
    }

    [[nodiscard]] auto GetInterpreter() const -> Interpreter const& { return interpreter_; }
};
} // namespace Operon::Autodiff::Reverse

#endif
