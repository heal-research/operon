/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2020 Bogdan Burlacu 
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE. 
 */

#ifndef OPERON_EVAL_DETAIL
#define OPERON_EVAL_DETAIL

#include "node.hpp"
#include <Eigen/Dense>

namespace Operon {
namespace detail {
    // addition up to 5 arguments
    template <typename T, size_t S, Operon::NodeType N = NodeType::Add>
    struct op {
        using Arg = Eigen::Ref<typename Eigen::DenseBase<Eigen::Array<T, S, Eigen::Dynamic, Eigen::ColMajor>>::ColXpr, Eigen::Unaligned, Eigen::Stride<S, 1>>;

        static inline void apply(Arg ret, Arg arg1) { ret = arg1; }

        template <typename... Args>
        static inline void apply(Arg ret, Arg arg1, Args... args) { ret = arg1 + (args + ...); }

        static inline void accumulate(Arg ret, Arg arg1) { ret += arg1; }

        template <typename... Args>
        static inline void accumulate(Arg ret, Arg arg1, Args... args) { ret += arg1 + (args + ...); }
    };

    template <typename T, size_t S>
    struct op<T, S, Operon::NodeType::Sub> {
        using Arg = Eigen::Ref<typename Eigen::DenseBase<Eigen::Array<T, S, Eigen::Dynamic, Eigen::ColMajor>>::ColXpr, Eigen::Unaligned, Eigen::Stride<S, 1>>;

        static inline void apply(Arg ret, Arg arg1) { ret = -arg1; }

        template <typename... Args>
        static inline void apply(Arg ret, Arg arg1, Args... args) { ret = arg1 - (args + ...); }

        static inline void accumulate(Arg ret, Arg arg1) { ret -= arg1; }

        template <typename... Args>
        static inline void accumulate(Arg ret, Arg arg1, Args... args) { ret -= arg1 + (args + ...); }
    };

    template <typename T, size_t S>
    struct op<T, S, Operon::NodeType::Mul> {
        using Arg = Eigen::Ref<typename Eigen::DenseBase<Eigen::Array<T, S, Eigen::Dynamic, Eigen::ColMajor>>::ColXpr, Eigen::Unaligned, Eigen::Stride<S, 1>>;

        static inline void apply(Arg ret, Arg arg1) { ret = arg1; }

        template <typename... Args>
        static inline void apply(Arg ret, Arg arg1, Args... args) { ret = arg1 * (args * ...); }

        static inline void accumulate(Arg ret, Arg arg1) { ret *= arg1; }

        template <typename... Args>
        static inline void accumulate(Arg ret, Arg arg1, Args... args) { ret *= arg1 * (args * ...); }
    };

    template <typename T, size_t S>
    struct op<T, S, Operon::NodeType::Div> {
        using Arg = Eigen::Ref<typename Eigen::DenseBase<Eigen::Array<T, S, Eigen::Dynamic, Eigen::ColMajor>>::ColXpr, Eigen::Unaligned, Eigen::Stride<S, 1>>;

        static inline void apply(Arg ret, Arg arg1) { ret = arg1.inverse(); }

        template <typename... Args>
        static inline void apply(Arg ret, Arg arg1, Args... args) { ret = arg1 / (args * ...); }

        static inline void accumulate(Arg ret, Arg arg1) { ret /= arg1; }

        template <typename... Args>
        static inline void accumulate(Arg ret, Arg arg1, Args... args) { ret /= arg1 * (args * ...); }
    };

    // dispatching mechanism
    // compared to the simple/naive way of evaluating n-ary symbols, this method has the following advantages:
    // 1) improved performance: the naive method accumulates into the result for each argument, leading to unnecessary assignments
    // 2) minimizing the number of intermediate steps which might improve floating point accuracy of some operations
    //    if arity > 5, one accumulation is performed every 5 args
    template <typename T, size_t S, Operon::NodeType N>
    inline void dispatch_op(Eigen::DenseBase<Eigen::Array<T, S, Eigen::Dynamic, Eigen::ColMajor>>& m, Operon::Vector<Node> const& nodes, size_t parentIndex)
    {
        auto result = m.col(parentIndex);

        using f = op<T, S, N>;
        const auto g = [](bool cont, decltype(result) res, auto&&... args) { cont ? f::accumulate(res, args...) : f::apply(res, args...); };
        const auto nextArg = [&](size_t i) { return i - (nodes[i].Length + 1); };

        auto arg1 = parentIndex - 1;

        bool continued = false;

        int arity = nodes[parentIndex].Arity;
        while (arity > 0) {
            switch (arity) {
            case 1: {
                g(continued, result, m.col(arg1));
                arity = 0;
                break;
            }
            case 2: {
                auto arg2 = nextArg(arg1);
                g(continued, result, m.col(arg1), m.col(arg2));
                arity = 0;
                break;
            }
            case 3: {
                auto arg2 = nextArg(arg1), arg3 = nextArg(arg2);
                g(continued, result, m.col(arg1), m.col(arg2), m.col(arg3));
                arity = 0;
                break;
            }
            default: {
                auto arg2 = nextArg(arg1), arg3 = nextArg(arg2), arg4 = nextArg(arg3);
                g(continued, result, m.col(arg1), m.col(arg2), m.col(arg3), m.col(arg4));
                arity -= 4;
                arg1 = nextArg(arg4);
                break;
            }
            }
            continued = true;
        }
    }

    template <typename T, size_t S, Operon::NodeType N>
    inline void dispatch_op_simple_binary(Eigen::DenseBase<Eigen::Array<T, S, Eigen::Dynamic, Eigen::ColMajor>>& m, Operon::Vector<Node> const& nodes, size_t parentIndex)
    {
        auto r = m.col(parentIndex);
        size_t i = parentIndex - 1;
        size_t arity = nodes[parentIndex].Arity;

        if (arity == 1) {
            op<T, S, N>::apply(r, m.col(i));
        } else {
            auto j = i - (nodes[i].Length + 1);
            op<T, S, N>::apply(r, m.col(i), m.col(j));
        }
    }

    template <typename T, size_t S, Operon::NodeType N>
    inline void dispatch_op_simple_nary(Eigen::DenseBase<Eigen::Array<T, S, Eigen::Dynamic, Eigen::ColMajor>>& m, Operon::Vector<Node> const& nodes, size_t parentIndex)
    {
        auto r = m.col(parentIndex);
        size_t arity = nodes[parentIndex].Arity;

        auto i = parentIndex - 1;

        if (arity == 1) {
            op<T, S, N>::apply(r, m.col(i));
        } else {
            r = m.col(i);

            for (size_t k = 1; k < arity; ++k) {
                i -= nodes[i].Length + 1;
                op<T, S, N>::accumulate(r, m.col(i));
            }
        }
    }

} // namespace detail
} // namespace Operon

#endif
