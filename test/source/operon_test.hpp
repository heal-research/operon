// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_TEST_HPP
#define OPERON_TEST_HPP

#include "thirdparty/elki_stats.hpp"
#include "thirdparty/nanobench.h"

#include <fmt/core.h>
#include <fmt/color.h>
#include <fmt/ranges.h>

#include "operon/core/dataset.hpp"
#include "operon/core/tree.hpp"
#include "operon/interpreter/backend/backend.hpp"
#include "operon/interpreter/dual.hpp"

namespace Operon::Test::Util {
    inline auto RandomDataset(Operon::RandomGenerator& rng, int rows, int cols) -> Operon::Dataset {
        std::uniform_real_distribution<Operon::Scalar> dist(-1.f, +1.f);
        Eigen::Matrix<decltype(dist)::result_type, -1, -1> data(rows, cols);
        for (auto& v : data.reshaped()) { v = dist(rng); }
        Operon::Dataset ds(data);
        return ds;
    }

    template<typename T, std::size_t S = Backend::BatchSize<T>>
    auto EvaluateTree(auto const& tree, auto const& dataset, auto const range, T const* coeff, T* out) {
        auto const nrows = range.Size();
        auto const& nodes = tree.Nodes();

        Eigen::Array<T, S, -1> buffer(S, nodes.size());
        buffer.setZero();
        Backend::View<T, S> view(buffer.data(), S, nodes.size());

        for (auto row = 0UL; row < nrows; row += S) {
            auto const rem = std::min(S, nrows - row);

            auto idx{0};
            for (auto i = 0UL; i < nodes.size(); ++i) {
                auto const& n = nodes[i];
                auto const v = n.Optimize ? coeff[idx++] : T{n.Value};
                switch(n.Type) {
                    case NodeType::Constant: {
                        buffer.col(i).segment(row, rem).setConstant(v);
                        break;
                    }
                    case NodeType::Variable: {
                        auto const* ptr = dataset.GetValues(n.HashValue).subspan(row, rem).data();
                        buffer.col(i).segment(row, rem) = v * Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const>(ptr, rem, 1).template cast<T>();
                        break;
                    }
                    case NodeType::Add: {
                        buffer.col(i).setConstant(T{0});
                        for (auto j : Tree::Indices(nodes, i)) {
                            buffer.col(i) += buffer.col(j);
                        }
                        break;
                    }
                    case NodeType::Mul: {
                        buffer.col(i).setConstant(T{1});
                        for (auto j : Tree::Indices(nodes, i)) {
                            buffer.col(i) *= buffer.col(j);
                        }
                        break;
                    }
                    case NodeType::Sub: {
                        if (n.Arity == 1) {
                            buffer.col(i) = -buffer.col(i-1);
                        } else {
                            buffer.col(i) = buffer.col(i-1);
                            for (auto j : Tree::Indices(nodes, i)) {
                                if (j == i-1) { continue; }
                                buffer.col(i) -= buffer.col(j);
                            }
                        }
                        break;
                    }
                    case NodeType::Div: {
                        if (n.Arity == 1) {
                            buffer.col(i) = buffer.col(i-1).inverse();
                        } else {
                            buffer.col(i) = buffer.col(i-1);
                            for (auto j : Tree::Indices(nodes, i)) {
                                if (j == i-1) { continue; }
                                buffer.col(i) /= buffer.col(j);
                            }
                        }
                        break;
                    }
                    case NodeType::Fmin: {
                        buffer.col(i) = buffer.col(i-1);
                        for (auto j : Tree::Indices(nodes, i)) {
                            if (j == i-1) { continue; }
                            buffer.col(i) = buffer.col(i).min(buffer.col(j));
                        }
                        break;
                    }
                    case NodeType::Fmax: {
                        buffer.col(i) = buffer.col(i-1);
                        for (auto j : Tree::Indices(nodes, i)) {
                            if (j == i-1) { continue; }
                            buffer.col(i) = buffer.col(i).max(buffer.col(j));
                        }
                        break;
                    }
                    case NodeType::Aq: {
                        auto j = i-1;
                        auto k = j - (nodes[j].Length + 1);
                        buffer.col(i) = buffer.col(j) / (T{1} + buffer.col(k).square()).sqrt();
                        break;
                    }
                    case NodeType::Pow: {
                        auto j = i-1;
                        auto k = j - (nodes[j].Length + 1);
                        buffer.col(i) = buffer.col(j).pow(buffer.col(k));
                        break;
                    }
                    case NodeType::Abs: {
                        buffer.col(i) = buffer.col(i-1).abs();
                        break;
                    }
                    case NodeType::Acos: {
                        buffer.col(i) = buffer.col(i-1).acos();
                        break;
                    }
                    case NodeType::Asin: {
                        buffer.col(i) = buffer.col(i-1).asin();
                        break;
                    }
                    case NodeType::Atan: {
                        buffer.col(i) = buffer.col(i-1).atan();
                        break;
                    }
                    case NodeType::Cbrt: {
                        buffer.col(i) = buffer.col(i-1).unaryExpr([](auto x) { return ceres::cbrt(x); });
                        break;
                    }
                    case NodeType::Ceil: {
                        buffer.col(i) = buffer.col(i-1).ceil();
                        break;
                    }
                    case NodeType::Cos: {
                        buffer.col(i) = buffer.col(i-1).cos();
                        break;
                    }
                    case NodeType::Cosh: {
                        buffer.col(i) = buffer.col(i-1).cosh();
                        break;
                    }
                    case NodeType::Exp: {
                        buffer.col(i) = buffer.col(i-1).exp();
                        break;
                    }
                    case NodeType::Floor: {
                        buffer.col(i) = buffer.col(i-1).floor();
                        break;
                    }
                    case NodeType::Log: {
                        buffer.col(i) = buffer.col(i-1).log();
                        break;
                    }
                    case NodeType::Logabs: {
                        buffer.col(i) = buffer.col(i-1).abs().log();
                        break;
                    }
                    case NodeType::Log1p: {
                        buffer.col(i) = buffer.col(i-1).log1p();
                        break;
                    }
                    case NodeType::Sin: {
                        buffer.col(i) = buffer.col(i-1).sin();
                        break;
                    }
                    case NodeType::Sinh: {
                        buffer.col(i) = buffer.col(i-1).sinh();
                        break;
                    }
                    case NodeType::Sqrt: {
                        buffer.col(i) = buffer.col(i-1).sqrt();
                        break;
                    }
                    case NodeType::Sqrtabs: {
                        buffer.col(i) = buffer.col(i-1).abs().sqrt();
                        break;
                    }
                    case NodeType::Square: {
                        buffer.col(i) = buffer.col(i-1).square();
                        break;
                    }
                    case NodeType::Tan: {
                        buffer.col(i) = buffer.col(i-1).tan();
                        break;
                    }
                    case NodeType::Tanh: {
                        buffer.col(i) = buffer.col(i-1).tanh();
                        break;
                    }
                    default: {
                        throw std::runtime_error(fmt::format("unknown node type: {}\n", n.Name()));
                    }
                }
                if (!n.IsLeaf()) { buffer.col(i) *= v; }
            }
            std::ranges::copy_n(buffer.col(nodes.size()-1).segment(row, rem).data(), rem, out + row);
        }
    }

    auto Autodiff(auto const& tree, auto const& dataset, auto const range) {
        static_assert(std::is_convertible_v<typename Dual::Scalar, Scalar>, "The chosen Jet and Scalar types are not compatible.");
        static_assert(std::is_convertible_v<Scalar, typename Dual::Scalar>, "The chosen Jet and Scalar types are not compatible.");

        auto coeff = tree.GetCoefficients();
        auto* parameters = coeff.data();

        std::vector<Operon::Scalar> resid(range.Size());
        auto* residuals = resid.data();

        std::vector<Operon::Scalar> jacob(range.Size() * coeff.size());
        auto* jacobian = jacob.data();

        if (parameters == nullptr) {
            return std::tuple{std::move(resid), std::move(jacob)};
        }

        EXPECT(parameters != nullptr);
        EXPECT(residuals != nullptr || jacobian != nullptr);

        std::vector<Dual> inputs(coeff.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            inputs[i].a = parameters[i];
            inputs[i].v.setZero();
        }
        std::vector<Dual> outputs(range.Size());

        static auto constexpr dim{Dual::DIMENSION};
        Eigen::Map<Eigen::Matrix<Scalar, -1, -1>> jmap(jacobian, outputs.size(), inputs.size());

        auto function = [&](auto const* inputs, auto* outputs) {
            EvaluateTree<Dual>(tree, dataset, range, inputs, outputs);
        };

        for (auto s = 0U; s < inputs.size(); s += dim) {
            auto r = std::min(static_cast<uint32_t>(inputs.size()), s + dim); // remaining parameters

            for (auto i = s; i < r; ++i) {
                inputs[i].v[i - s] = 1.0;
            }

            function(inputs.data(), outputs.data());

            for (auto i = s; i < r; ++i) {
                inputs[i].v[i - s] = 0.0;
            }

            for (auto i = s; i < r; ++i) {
                std::transform(outputs.cbegin(), outputs.cend(), jmap.col(i).data(), [&](auto const& jet) { return jet.v[i - s]; });
            }
        }
        if (residuals != nullptr) {
            std::transform(std::cbegin(outputs), std::cend(outputs), residuals, [](auto const& jet) { return jet.a; });
        }

        return std::tuple{std::move(resid), std::move(jacob)};
    }
} // namespace Operon::Test::Util

#endif
