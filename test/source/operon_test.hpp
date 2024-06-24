// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_TEST_HPP
#define OPERON_TEST_HPP

#include "thirdparty/elki_stats.hpp"
#include "thirdparty/nanobench.h"

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
                        auto *s = buffer.col(i).data() + row;

                        for (auto k = 0UL; k < rem; ++k) {
                            s[k] = T{v};
                        }

                        // std::span s{buffer.col(i).data() + row, rem};
                        // std::ranges::fill_n(s, rem, T{v});
                        //buffer.col(i).segment(row, rem).setConstant(v);
                        break;
                    }
                    case NodeType::Variable: {
                        auto const* ptr = dataset.GetValues(n.HashValue).subspan(row, rem).data();
                        auto* s = buffer.col(i).data() + row;
                        for (auto k = 0UL; k < rem; ++k) {
                            s[k] = v * T{ptr[k]};
                        }
                        // buffer.col(i).segment(row, rem) = v * Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const>(ptr, rem, 1).template cast<T>();
                        break;
                    }
                    case NodeType::Add: {
                        std::span s{buffer.col(i).data(), S};
                        for (auto k = 0UL; k < S; ++k) {
                            s[k] = T{0};
                        }
                        for (auto j : Tree::Indices(nodes, i)) {
                            std::span p{buffer.col(j).data(), S};
                            for (auto k = 0UL; k < S; ++k) {
                                s[k] += p[k];
                            }
                        }
                        break;
                    }
                    case NodeType::Mul: {
                        std::span s{buffer.col(i).data(), S};
                        for (auto k = 0UL; k < S; ++k) {
                            s[k] = T{1};
                        }
                        for (auto j : Tree::Indices(nodes, i)) {
                            std::span p{buffer.col(j).data(), S};
                            for (auto k = 0UL; k < S; ++k) {
                                s[k] *= p[k];
                            }
                        }
                        break;
                    }
                    case NodeType::Sub: {
                        auto* s = buffer.col(i).data();
                        auto const* p = buffer.col(i-1).data();
                        if (n.Arity == 1) {
                            for (auto k = 0UL; k < S; ++k) {
                                s[k] = -p[k];
                            }
                        } else {
                            for (auto k = 0UL; k < S; ++k) {
                                s[k] = p[k];
                            }
                            for (auto j : Tree::Indices(nodes, i)) {
                                if (j == i-1) { continue; }
                                auto const* q = buffer.col(j).data();
                                for (auto k = 0UL; k < S; ++k) {
                                    s[k] -= q[k];
                                }
                            }
                        }
                        break;
                    }
                    case NodeType::Div: {
                        auto* s = buffer.col(i).data();
                        auto const* p = buffer.col(i-1).data();
                        if (n.Arity == 1) {
                            for (auto k = 0UL; k < S; ++k) {
                                s[k] = T{1} / p[k];
                            }
                        } else {
                            for (auto k = 0UL; k < S; ++k) {
                                s[k] = p[k];
                            }
                            for (auto j : Tree::Indices(nodes, i)) {
                                if (j == i-1) { continue; }
                                auto const* q = buffer.col(j).data();
                                for (auto k = 0UL; k < S; ++k) {
                                    s[k] /= q[k];
                                }
                            }
                        }
                        break;
                    }
                    case NodeType::Fmin: {
                        auto* s = buffer.col(i).data();
                        auto const* p = buffer.col(i-1).data();
                        for (auto k = 0UL; k < S; ++k) {
                            s[k] = p[k];
                        }
                        for (auto j : Tree::Indices(nodes, i)) {
                            if (j == i-1) { continue; }
                            auto const* q = buffer.col(j).data();
                            for (auto k = 0UL; k < S; ++k) {
                                s[k] = std::min(s[k], q[k]);
                            }
                        }
                        break;
                    }
                    case NodeType::Fmax: {
                        auto* s = buffer.col(i).data();
                        auto const* p = buffer.col(i-1).data();
                        for (auto k = 0UL; k < S; ++k) {
                            s[k] = p[k];
                        }
                        for (auto j : Tree::Indices(nodes, i)) {
                            if (j == i-1) { continue; }
                            auto const* q = buffer.col(j).data();
                            for (auto k = 0UL; k < S; ++k) {
                                s[k] = std::max(s[k], q[k]);
                            }
                        }
                        break;
                    }
                    case NodeType::Aq: {
                        auto j = i-1;
                        auto k = j - (nodes[j].Length + 1);

                        auto* s = buffer.col(i).data();
                        auto const *p = buffer.col(j).data();
                        auto const *q = buffer.col(k).data();

                        for (auto k = 0UL; k < S; ++k) {
                            s[k] = p[k] / ceres::sqrt(T{1} + q[k] * q[k]);
                        }
                        break;
                    }
                    case NodeType::Pow: {
                        auto j = i-1;
                        auto k = j - (nodes[j].Length + 1);
                        auto* s = buffer.col(i).data();
                        auto const *p = buffer.col(j).data();
                        auto const *q = buffer.col(k).data();
                        // buffer.col(i) = buffer.col(j).pow(buffer.col(k));
                        for (auto k = 0UL; k < S; ++k) {
                            s[k] = ceres::pow(p[k], q[k]);
                        }
                        break;
                    }
                    case NodeType::Abs: {
                        auto* s = buffer.col(i).data();
                        auto const *p = buffer.col(i-1).data();
                        for (auto k = 0UL; k < S; ++k) {
                            s[k] = ceres::abs(p[k]);
                        }
                        break;
                    }
                    case NodeType::Acos: {
                        auto* s = buffer.col(i).data();
                        auto const *p = buffer.col(i-1).data();
                        for (auto k = 0UL; k < S; ++k) {
                            s[k] = ceres::acos(p[k]);
                        }
                        break;
                    }
                    case NodeType::Asin: {
                        auto* s = buffer.col(i).data();
                        auto const *p = buffer.col(i-1).data();
                        for (auto k = 0UL; k < S; ++k) {
                            s[k] = ceres::asin(p[k]);
                        }
                        break;
                    }
                    case NodeType::Atan: {
                        auto* s = buffer.col(i).data();
                        auto const *p = buffer.col(i-1).data();
                        for (auto k = 0UL; k < S; ++k) {
                            s[k] = ceres::atan(p[k]);
                        }
                        break;
                    }
                    case NodeType::Cbrt: {
                        auto* s = buffer.col(i).data();
                        auto const *p = buffer.col(i-1).data();
                        for (auto k = 0UL; k < S; ++k) {
                            s[k] = ceres::cbrt(p[k]);
                        }
                        break;
                    }
                    case NodeType::Ceil: {
                        auto* s = buffer.col(i).data();
                        auto const *p = buffer.col(i-1).data();
                        for (auto k = 0UL; k < S; ++k) {
                            s[k] = ceres::ceil(p[k]);
                        }
                        break;
                    }
                    case NodeType::Cos: {
                        auto* s = buffer.col(i).data();
                        auto const *p = buffer.col(i-1).data();
                        for (auto k = 0UL; k < S; ++k) {
                            s[k] = ceres::cos(p[k]);
                        }
                        break;
                    }
                    case NodeType::Cosh: {
                        auto* s = buffer.col(i).data();
                        auto const *p = buffer.col(i-1).data();
                        for (auto k = 0UL; k < S; ++k) {
                            s[k] = ceres::sin(p[k]);
                        }
                        break;
                    }
                    case NodeType::Exp: {
                        auto* s = buffer.col(i).data();
                        auto const *p = buffer.col(i-1).data();
                        for (auto k = 0UL; k < S; ++k) {
                            s[k] = ceres::exp(p[k]);
                        }
                        break;
                    }
                    case NodeType::Floor: {
                        auto* s = buffer.col(i).data();
                        auto const *p = buffer.col(i-1).data();
                        for (auto k = 0UL; k < S; ++k) {
                            s[k] = ceres::floor(p[k]);
                        }
                        break;
                    }
                    case NodeType::Log: {
                        auto* s = buffer.col(i).data();
                        auto const *p = buffer.col(i-1).data();
                        for (auto k = 0UL; k < S; ++k) {
                            s[k] = ceres::log(p[k]);
                        }
                        break;
                    }
                    case NodeType::Logabs: {
                        auto* s = buffer.col(i).data();
                        auto const *p = buffer.col(i-1).data();
                        for (auto k = 0UL; k < S; ++k) {
                            s[k] = ceres::log(ceres::abs(p[k]));
                        }
                        break;
                    }
                    case NodeType::Log1p: {
                        auto* s = buffer.col(i).data();
                        auto const *p = buffer.col(i-1).data();
                        for (auto k = 0UL; k < S; ++k) {
                            s[k] = ceres::log(p[k]+T{1});
                        }
                        break;
                    }
                    case NodeType::Sin: {
                        auto* s = buffer.col(i).data();
                        auto const *p = buffer.col(i-1).data();
                        for (auto k = 0UL; k < S; ++k) {
                            s[k] = ceres::sin(p[k]);
                        }
                        break;
                    }
                    case NodeType::Sinh: {
                        auto* s = buffer.col(i).data();
                        auto const *p = buffer.col(i-1).data();
                        for (auto k = 0UL; k < S; ++k) {
                            s[k] = ceres::sinh(p[k]);
                        }
                        break;
                    }
                    case NodeType::Sqrt: {
                        auto* s = buffer.col(i).data();
                        auto const *p = buffer.col(i-1).data();
                        for (auto k = 0UL; k < S; ++k) {
                            s[k] = ceres::sqrt(p[k]);
                        }
                        break;
                    }
                    case NodeType::Sqrtabs: {
                        auto* s = buffer.col(i).data();
                        auto const *p = buffer.col(i-1).data();
                        for (auto k = 0UL; k < S; ++k) {
                            s[k] = ceres::sqrt(ceres::abs(p[k]));
                        }
                        break;
                    }
                    case NodeType::Square: {
                        auto* s = buffer.col(i).data();
                        auto const *p = buffer.col(i-1).data();
                        for (auto k = 0UL; k < S; ++k) {
                            s[k] = p[k] * p[k];
                        }
                        break;
                    }
                    case NodeType::Tan: {
                        auto* s = buffer.col(i).data();
                        auto const *p = buffer.col(i-1).data();
                        for (auto k = 0UL; k < S; ++k) {
                            s[k] = ceres::tan(p[k]);
                        }
                        break;
                    }
                    case NodeType::Tanh: {
                        auto* s = buffer.col(i).data();
                        auto const *p = buffer.col(i-1).data();
                        for (auto k = 0UL; k < S; ++k) {
                            s[k] = ceres::tanh(p[k]);
                        }
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
