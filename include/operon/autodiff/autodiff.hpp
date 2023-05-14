// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_AUTODIFF_REVERSE_HPP
#define OPERON_AUTODIFF_REVERSE_HPP

#include "operon/core/dataset.hpp"
#include "operon/core/tree.hpp"
#include "operon/interpreter/dispatch_table.hpp"
#include "derivatives.hpp"
#include "dual.hpp"

namespace Operon::Autodiff {

enum class AutodiffMode { Forward, ForwardJet, Reverse };

template<typename Interpreter, AutodiffMode Mode = AutodiffMode::Reverse>
class DerivativeCalculator {
    std::reference_wrapper<Interpreter const> interpreter_;
    bool print_{false};

    auto ComputeTrace(auto const& nodes, auto primal, auto trace, auto weights, auto primalColIdx) const {
        auto df = [&]<auto N>() { Derivative<N>{}(nodes, primal, trace, weights, primalColIdx); };

        switch(nodes[primalColIdx].Type) {
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
        case NodeType::Sinh:    { df.template operator()<NodeType::Sinh>(); return; }
        case NodeType::Cosh:    { df.template operator()<NodeType::Cosh>(); return; }
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

    auto WriteTrace(auto const& nodes, auto const& dataset, auto const& trace) const // NOLINT
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
                fmt::format_to(out, "\tn{} -> n{} [label=\"{:.3f}\"]\n", j, i, trace.row(0)(j));
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
        if constexpr (Mode == AutodiffMode::Reverse) {
            ReverseMode(tree, dataset, range, coeff, residual, jacobian);
        } else if constexpr (Mode == AutodiffMode::Forward) {
            ForwardMode(tree, dataset, range, coeff, residual, jacobian);
        } else if constexpr (Mode == AutodiffMode::ForwardJet) {
            ForwardModeJet(tree, dataset, range, coeff, residual, jacobian);
        }
    }

    template<int StorageOrder = Eigen::ColMajor>
    auto ForwardModeJet(Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Range const range, Operon::Span<Operon::Scalar const> coeff, Operon::Span<Operon::Scalar> residual, Operon::Span<Operon::Scalar> jacobian) const
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

            interpreter_.get().template operator()<Dual>(tree, dataset, range, outputs, inputs);

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

    template<int StorageOrder = Eigen::ColMajor>
    auto ForwardMode(Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Range const range, Operon::Span<Operon::Scalar const> coeff, Operon::Span<Operon::Scalar> residual, Operon::Span<Operon::Scalar> jacobian) const
    {
        auto const& nodes = tree.Nodes();
        auto const nn{ std::ssize(nodes) };
        auto const np{ static_cast<int>(coeff.size()) }; // number of parameters
        auto const nr{ static_cast<int>(range.Size()) }; // number of residuals
        Eigen::Map<Eigen::Array<Operon::Scalar, -1, -1, StorageOrder>> jac(jacobian.data(), nr, np);
        jac.setConstant(0);

        auto constexpr S{ static_cast<int>(Dispatch::BatchSize<Operon::Scalar>) };
        Dispatch::Matrix<Operon::Scalar> dot(S, nn);
        Dispatch::Matrix<Operon::Scalar> trace(S, nn);

        std::vector<Operon::Scalar> param(nn);
        std::vector<int> idx(nn);
        std::vector<int> cidx; cidx.reserve(coeff.size());
        for (auto i = 0, j = 0, k = 0; i < nn; ++i) {
            param[i] = nodes[i].Optimize ? coeff[k] : nodes[i].Value;
            idx[i] = k;
            k += nodes[i].Optimize;
            if (nodes[i].Optimize) { cidx.push_back(i); }
        }

        auto forward = [&](auto const& primal, auto row) {
            auto const len = std::min(S, nr - row);

            for (auto i = 0; i < nn; ++i) {
                if (nodes[i].IsLeaf()) { continue; }
                ComputeTrace(nodes, primal.topRows(len), trace.topRows(len), Operon::Span<Operon::Scalar>{ param }, i);
            }

            for (auto c : cidx) {
                dot.topRows(len).setConstant(0);
                dot.col(c).head(len).setConstant(1);

                for (auto i = 0; i < nn; ++i) {
                    if (nodes[i].IsLeaf()) { continue; }
                    for (auto x : Indices(nodes, i)) {
                        auto j{ static_cast<int64_t>(x) };
                        if (nodes[j].IsLeaf() && j != c) { continue; }
                        dot.col(i).head(len) += dot.col(j).head(len) * trace.col(j).head(len) * param[i];
                    }
                }

                jac.col(idx[c]).segment(row, len) = dot.col(nn-1).head(len) * primal.col(c).head(len) / param[c];
            }
        };

        interpreter_.get().template operator()<Operon::Scalar>(tree, dataset, range, residual, coeff, forward);

        if (print_) {
            auto str = WriteTrace(nodes, dataset, trace);
            fmt::print("{}\n", str);
        }
    }


    template<int StorageOrder = Eigen::ColMajor>
    auto ReverseMode(Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Range const range, Operon::Span<Operon::Scalar const> coeff, Operon::Span<Operon::Scalar> residual, Operon::Span<Operon::Scalar> jacobian) const
    {
        auto const& nodes = tree.Nodes();
        auto const nn{ std::ssize(nodes) };
        auto const np{ static_cast<int>(coeff.size()) }; // number of parameters
        auto const nr{ static_cast<int>(range.Size()) }; // number of residuals
        Eigen::Map<Eigen::Array<Operon::Scalar, -1, -1, StorageOrder>> jac(jacobian.data(), nr, np);
        jac.setConstant(0);

        auto constexpr S{ static_cast<int>(Dispatch::BatchSize<Operon::Scalar>) };

        Dispatch::Matrix<Operon::Scalar> trace(S, nn);

        std::vector<Operon::Scalar> param(nn);
        std::vector<int> idx(nn);
        for (auto i = 0, j = 0, k = 0; i < nn; ++i) {
            param[i] = nodes[i].Optimize ? coeff[k] : nodes[i].Value;
            idx[i] = k;
            k += nodes[i].Optimize;
        }

        trace.col(nn-1).setConstant(1);

        auto reverse = [&](auto const& primal, auto row) {
            auto const len = std::min(S, nr - row);

            // forward pass - compute trace
            for (auto i = 0; i < nn; ++i) {
                if (nodes[i].Arity > 0) {
                    ComputeTrace(nodes, primal.topRows(len), trace.topRows(len), Operon::Span<Operon::Scalar>{ param }, i);
                }
            }

            // backward pass - propagate adjoints
            for (auto i = nn-1; i >= 0; --i) {
                if (nodes[i].Optimize) {
                    jac.col(idx[i]).segment(row, len) = trace.col(i).head(len) * primal.col(i).head(len) / param[i];
                }

                if (nodes[i].IsLeaf()) { continue; }

                for (auto j : Indices(nodes, i)) {
                    auto const x { static_cast<int64_t>(j) };
                    trace.col(x).head(len) *= trace.col(i).head(len) * param[i];
                }
            }
        };

        interpreter_.get().template operator()<Operon::Scalar>(tree, dataset, range, residual, coeff, reverse);

        if (print_) {
            auto str = WriteTrace(nodes, dataset, trace);
            fmt::print("{}\n", str);
        }
    }

    [[nodiscard]] auto GetInterpreter() const -> Interpreter const& { return interpreter_; }
};
} // namespace Operon::Autodiff

#endif
