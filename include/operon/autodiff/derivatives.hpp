// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_AUTODIFF_REVERSE_DERIVATIVES_HPP
#define OPERON_AUTODIFF_REVERSE_DERIVATIVES_HPP

#include "operon/core/subtree.hpp"
#include <functional>

namespace Operon::Autodiff {
inline auto Enumerate(auto const& nodes, auto i)
{
    return Subtree<Node const> { nodes, static_cast<std::size_t>(i) }.EnumerateIndices();
}

inline auto Indices(auto const& nodes, auto i)
{
    return Subtree<Node const> { nodes, static_cast<std::size_t>(i) }.Indices();
}

namespace detail {
    template<typename Compare>
    struct FComp {
        auto operator()(auto x, auto y) const {
            using T = std::common_type_t<decltype(x), decltype(y)>;
            if ((std::isnan(x) && std::isnan(y)) || (x == y)) {
                return std::numeric_limits<T>::quiet_NaN();
            }
            if (std::isnan(x)) { return T{0}; }
            if (std::isnan(y)) { return T{1}; }
            return static_cast<Operon::Scalar>(Compare{}(x, y));
        }
    };
} // namespace detail

/* The methods below for computing derivatives have the following signature:
 * - nodes   : the list of notes in the tree
 * - primal  : the primal trace (i.e. intermediate node evaluation values)
 * - trace   : the derivatives computed so far
 * - weights : the weight coefficients for each node
 * - i       : index of the current tree node
 * - j       : index of the current trace column where the derivative values should be written to
 */

// derivatives
template <Operon::NodeType N = Operon::NodeType::Add>
struct Derivative {
    inline auto operator()(auto const& nodes, auto const /*unused*/, auto trace, auto /*weights*/, auto i)
    {
        for (auto j : Indices(nodes, i)) {
            trace.col(j).setConstant(1);
        }
    }
};

template <>
struct Derivative<Operon::NodeType::Sub> {
    inline auto operator()(auto const& nodes, auto const /*unused*/, auto trace, auto /*weights*/, auto i)
    {
        auto const& n = nodes[i];
        if (n.Arity == 1) {
            trace.col(i-1).setConstant(-1);
        } else {
            for (auto [k, j] : Enumerate(nodes, i)) {
                trace.col(j).setConstant(k == 0 ? +1 : -1);
            }
        }
    }
};

template <>
struct Derivative<Operon::NodeType::Mul> {
    inline auto operator()(auto const& nodes, auto const primal, auto trace, auto weights, auto i)
    {
        auto const& n = nodes[i];
        for (auto [k, j] : Enumerate(nodes, i)) {
            trace.col(j) = primal.col(i) / primal.col(j) / weights[i];
        }
    }
};

template <>
struct Derivative<Operon::NodeType::Div> {
    inline auto operator()(auto const& nodes, auto const primal, auto trace, auto weights, auto i)
    {
        auto const& n = nodes[i];
        if (n.Arity == 1) {
            trace.col(i-1) = -primal.col(i-1).square().inverse();
        } else {
            for (auto [k, j] : Enumerate(nodes, i)) {
                trace.col(j) = (k == 0 ? +1 : -1) * primal.col(i) / primal.col(j) / weights[i];
            }
        }
    }
};

// binary symbols
template <>
struct Derivative<Operon::NodeType::Aq> {
    inline auto operator()(auto const& nodes, auto const primal, auto trace, auto weights, auto i)
    {
        auto a = i - 1;
        auto b = a - (nodes[a].Length + 1);
        auto p = primal.col(i) / weights[i];
        trace.col(a)   = p / primal.col(a);
        trace.col(b) = -primal.col(b) * p.pow(3) / primal.col(a).square();
    }
};

template <>
struct Derivative<Operon::NodeType::Pow> {
    inline auto operator()(auto const& nodes, auto const primal, auto trace, auto weights, auto i)
    {
        auto a = i - 1;
        auto b = a - (nodes[a].Length + 1);
        trace.col(a)   = primal.col(i) * primal.col(b) / (primal.col(a) * weights[i]);
        trace.col(b) = primal.col(i) * primal.col(a).log() / weights[i];
    }
};

template <>
struct Derivative<Operon::NodeType::Fmin> {
    inline auto operator()(auto const& nodes, auto const primal, auto trace, auto /*weights*/, auto i)
    {
        auto a = i - 1;
        auto b = a - (nodes[a].Length + 1);
        trace.col(a) = primal.col(a).binaryExpr(primal.col(b), detail::FComp<std::less<>>{});
        trace.col(b) = (trace.col(a) - 1).abs();
    }
};

template <>
struct Derivative<Operon::NodeType::Fmax> {
    inline auto operator()(auto const& nodes, auto const primal, auto trace, auto /*weights*/, auto i)
    {
        auto a = i - 1;
        auto b = a - (nodes[a].Length + 1);
        trace.col(a) = primal.col(a).binaryExpr(primal.col(b), detail::FComp<std::greater<>>{});
        trace.col(b) = (trace.col(a) - 1).abs();
    }
};

template <>
struct Derivative<Operon::NodeType::Square> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto /*weights*/, auto i)
    {
        trace.col(i-1) = 2 * primal.col(i-1);
    }
};

template <>
struct Derivative<Operon::NodeType::Abs> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto /*weights*/, auto i)
    {
        trace.col(i-1) = primal.col(i-1).sign();
    }
};

// unary symbols
template <>
struct Derivative<Operon::NodeType::Exp> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto weights, auto i)
    {
        //trace.col(j) = primal.col(i-1).exp();
        trace.col(i-1) = primal.col(i) / weights[i];
    }
};

template <>
struct Derivative<Operon::NodeType::Log> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto /*weights*/, auto i)
    {
        trace.col(i-1) = primal.col(i-1).inverse();
    }
};

template <>
struct Derivative<Operon::NodeType::Logabs> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto /*weights*/, auto i)
    {
        trace.col(i-1) = primal.col(i-1).sign() / primal.col(i-1).abs();
    }
};

template <>
struct Derivative<Operon::NodeType::Log1p> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto /*weights*/, auto i)
    {
        trace.col(i-1) = (primal.col(i-1) + 1).inverse();
    }
};

template <>
struct Derivative<Operon::NodeType::Sin> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto /*weights*/, auto i)
    {
        trace.col(i-1) = primal.col(i-1).cos();
    }
};

template <>
struct Derivative<Operon::NodeType::Cos> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto /*weights*/, auto i)
    {
        trace.col(i-1) = -primal.col(i-1).sin();
    }
};

template <>
struct Derivative<Operon::NodeType::Tan> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto /*weights*/, auto i)
    {
        trace.col(i-1) = 1 + primal.col(i-1).tan().square();
    }
};

template <>
struct Derivative<Operon::NodeType::Tanh> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto /*weights*/, auto i)
    {
        trace.col(i-1) = 1 - primal.col(i-1).tanh().square();
    }
};

template <>
struct Derivative<Operon::NodeType::Asin> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto /*weights*/, auto i)
    {
        trace.col(i-1) = (1 - primal.col(i-1).square()).sqrt().inverse();
    }
};

template <>
struct Derivative<Operon::NodeType::Acos> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto /*weights*/, auto i)
    {
        trace.col(i-1) = -(1 - primal.col(i-1).square()).sqrt().inverse();
    }
};

template <>
struct Derivative<Operon::NodeType::Atan> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto /*weights*/, auto i)
    {
        trace.col(i-1) = (1 + primal.col(i-1).square()).inverse();
    }
};

template <>
struct Derivative<Operon::NodeType::Sqrt> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto weights, auto i)
    {
        trace.col(i-1) = weights[i] * (2 * primal.col(i)).inverse();
    }
};

template <>
struct Derivative<Operon::NodeType::Sqrtabs> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto weights, auto i)
    {
        trace.col(i-1) = weights[i] * primal.col(i-1).sign() / (2 * primal.col(i));
    }
};

template <>
struct Derivative<Operon::NodeType::Cbrt> {
    inline auto operator()(auto const& /*nodes*/, auto const primal, auto trace, auto weights, auto i)
    {
        trace.col(i-1) = (3 * (primal.col(i) / weights[i]).square()).inverse();
    }
};
} // namespace Operon::Autodiff

#endif
