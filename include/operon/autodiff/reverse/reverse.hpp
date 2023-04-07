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
    std::reference_wrapper<Interpreter const> interpreter_;
    bool print_{false};

    auto ComputeDerivative(auto const& nodes, auto primal, auto trace, auto weights, auto i, auto j) const {
        auto df = [&]<auto N>() { Derivative<N>{}(nodes, primal, trace, weights, i, j); };

        switch(nodes[i].Type) {
        case NodeType::Add:     { df.template operator()<NodeType::Add>(); return; }
        case NodeType::Sub:     { df.template operator()<NodeType::Sub>(); return; }
        case NodeType::Mul:     { df.template operator()<NodeType::Mul>(); return; }
        case NodeType::Div:     { df.template operator()<NodeType::Div>(); return; }
        case NodeType::Pow:     { df.template operator()<NodeType::Pow>(); return; }
        case NodeType::Aq:      { df.template operator()<NodeType::Aq>(); return; }
        case NodeType::Square:  { df.template operator()<NodeType::Square>(); return; }
        case NodeType::Fmin:    { df.template operator()<NodeType::Fmin>(); return; }
        case NodeType::Fmax:    { df.template operator()<NodeType::Fmax>(); return; }
        case NodeType::Abs:     { df.template operator()<NodeType::Abs>(); return; }
        case NodeType::Exp:     { df.template operator()<NodeType::Exp>(); return; }
        case NodeType::Log:     { df.template operator()<NodeType::Log>(); return; }
        case NodeType::Logabs:  { df.template operator()<NodeType::Logabs>(); return; }
        case NodeType::Log1p:   { df.template operator()<NodeType::Log1p>(); return; }
        case NodeType::Sin:     { df.template operator()<NodeType::Sin>(); return; }
        case NodeType::Cos:     { df.template operator()<NodeType::Cos>(); return; }
        case NodeType::Tan:     { df.template operator()<NodeType::Tan>(); return; }
        case NodeType::Tanh:    { df.template operator()<NodeType::Tanh>(); return; }
        case NodeType::Asin:    { df.template operator()<NodeType::Asin>(); return; }
        case NodeType::Acos:    { df.template operator()<NodeType::Acos>(); return; }
        case NodeType::Atan:    { df.template operator()<NodeType::Atan>(); return; }
        case NodeType::Sqrt:    { df.template operator()<NodeType::Sqrt>(); return; }
        case NodeType::Sqrtabs: { df.template operator()<NodeType::Sqrtabs>(); return; }
        case NodeType::Cbrt:    { df.template operator()<NodeType::Cbrt>(); return; }
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

        interpreter_.get().template operator()<Operon::Scalar>(tree, dataset, range, residual, coeff, reverse);

        if (print_) {
            auto str = WriteTrace(nodes, dataset, partial, adj);
            fmt::print("{}\n", str);
        }
    }

    [[nodiscard]] auto GetInterpreter() const -> Interpreter const& { return interpreter_; }
};
} // namespace Operon::Autodiff::Reverse

#endif
