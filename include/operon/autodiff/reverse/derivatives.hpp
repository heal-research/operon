// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_AUTODIFF_REVERSE_DERIVATIVES_HPP
#define OPERON_AUTODIFF_REVERSE_DERIVATIVES_HPP

#include "operon/core/subtree.hpp"

namespace Operon::Autodiff::Reverse {
inline auto Enumerate(auto const& nodes, auto i)
{
    return Subtree<Node const> { nodes, static_cast<std::size_t>(i) }.EnumerateIndices();
}

inline auto Indices(auto const& nodes, auto i)
{
    return Subtree<Node const> { nodes, static_cast<std::size_t>(i) }.Indices();
}

// derivatives
template <Operon::NodeType N = Operon::NodeType::Add>
struct Derivative {
    inline auto operator()(auto const& nodes, auto const /*unused*/, auto trace, auto i, auto j)
    {
        trace.middleCols(j, nodes[i].Arity).setConstant(1);
    }
};

template <>
struct Derivative<Operon::NodeType::Sub> {
    inline auto operator()(auto const& nodes, auto const /*unused*/, auto trace, auto i, auto j)
    {
        auto const& n = nodes[i];
        if (n.Arity == 1) {
            trace.col(j).setConstant(-1);
        } else {
            for (auto k = 0; k < n.Arity; ++k) {
                trace.col(j+k).setConstant(k == 0 ? +1 : -1);
            }
        }
    }
};

template <>
struct Derivative<Operon::NodeType::Mul> {
    inline auto operator()(auto const& nodes, auto const primal, auto trace, auto i, auto j)
    {
        auto const& n = nodes[i];
        for (auto [k, a] : Enumerate(nodes, i)) {
            trace.col(j+k) = primal.col(i) / primal.col(a);
        }
    }
};

template <>
struct Derivative<Operon::NodeType::Div> {
    inline auto operator()(auto const& nodes, auto const primal, auto trace, auto i, auto j)
    {
        auto const& n = nodes[i];
        if (n.Arity == 1) {
            trace.col(j) = -primal.col(i-1).square().inverse();
        } else {
            for (auto [k, a] : Enumerate(nodes, i)) {
                trace.col(j+k) = (k == 0 ? +1 : -1) * primal.col(i) / primal.col(a);
            }
        }
    }
};

// binary symbols
template <>
struct Derivative<Operon::NodeType::Aq> {
    inline auto operator()(auto const& nodes, auto const primal, auto trace, auto i, auto j)
    {
        auto a = i - 1;
        auto b = a - (nodes[a].Length + 1);
        trace.col(j)   = primal.col(i) / primal.col(a);
        trace.col(j+1) = -primal.col(b) * primal.col(i).pow(3) / primal.col(a).square();
    }
};

template <>
struct Derivative<Operon::NodeType::Pow> {
    inline auto operator()(auto const& nodes, auto const primal, auto trace, auto i, auto j)
    {
        auto a = i - 1;
        auto b = a - (nodes[a].Length + 1);
        trace.col(j)   = primal.col(a).pow(primal.col(b) - 1) * primal.col(b);
        trace.col(j+1) = primal.col(i) * primal.col(a).log();
    }
};

// unary symbols
template <>
struct Derivative<Operon::NodeType::Exp> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto i, auto j)
    {
        trace.col(j) = primal.col(i);
    }
};

template <>
struct Derivative<Operon::NodeType::Log> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto i, auto j)
    {
        trace.col(j) = primal.col(i-1).inverse();
    }
};

template <>
struct Derivative<Operon::NodeType::Logabs> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto i, auto j)
    {
        trace.col(j) = primal.col(i-1).sign() / primal.col(i-1).abs();
    }
};

template <>
struct Derivative<Operon::NodeType::Log1p> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto i, auto j)
    {
        trace.col(j) = (primal.col(i-1) + 1).inverse();
    }
};

template <>
struct Derivative<Operon::NodeType::Sin> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto i, auto j)
    {
        trace.col(j) = primal.col(i-1).cos();
    }
};

template <>
struct Derivative<Operon::NodeType::Cos> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto i, auto j)
    {
        trace.col(j) = -primal.col(i-1).sin();
    }
};

template <>
struct Derivative<Operon::NodeType::Tan> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto i, auto j)
    {
        trace.col(j) = 1 + primal.col(i).square();
    }
};

template <>
struct Derivative<Operon::NodeType::Tanh> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto i, auto j)
    {
        trace.col(j) = 1 - primal.col(i).square();
    }
};

template <>
struct Derivative<Operon::NodeType::Asin> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto i, auto j)
    {
        trace.col(j) = (1 - primal.col(i-1).square()).sqrt().inverse();
    }
};

template <>
struct Derivative<Operon::NodeType::Acos> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto i, auto j)
    {
        trace.col(j) = -(1 - primal.col(i-1).square()).sqrt().inverse();
    }
};

template <>
struct Derivative<Operon::NodeType::Atan> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto i, auto j)
    {
        trace.col(j) = (1 + primal.col(i-1).square()).inverse();
    }
};

template <>
struct Derivative<Operon::NodeType::Sqrt> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto i, auto j)
    {
        trace.col(j) = (2 * primal.col(i)).inverse();
    }
};

template <>
struct Derivative<Operon::NodeType::Sqrtabs> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto i, auto j)
    {
        trace.col(j) = primal.col(i-1).sign() / (2 * primal.col(i));
    }
};

template <>
struct Derivative<Operon::NodeType::Cbrt> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto i, auto j)
    {
        trace.col(j) = (3 * primal.col(i).square()).inverse();
    }
};
} // namespace Operon::Autodiff::Reverse

#endif
