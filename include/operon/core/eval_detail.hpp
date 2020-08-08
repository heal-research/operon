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
constexpr size_t BATCHSIZE = 64;

namespace detail {
    // addition up to 5 arguments
    template <typename T, Operon::NodeType N = NodeType::Add>
    struct op {
        using Arg = Eigen::Ref<typename Eigen::DenseBase<Eigen::Array<T, BATCHSIZE, Eigen::Dynamic, Eigen::ColMajor>>::ColXpr, Eigen::Unaligned, Eigen::Stride<BATCHSIZE, 1>>;

        inline void operator()(Arg ret, Arg arg1) { ret = arg1; }

        template <typename... Args>
        inline void operator()(Arg ret, Arg arg1, Args... args) { ret = arg1 + (args + ...); }
    };

    template <typename T>
    struct op<T, Operon::NodeType::Sub> {
        using Arg = Eigen::Ref<typename Eigen::DenseBase<Eigen::Array<T, BATCHSIZE, Eigen::Dynamic, Eigen::ColMajor>>::ColXpr, Eigen::Unaligned, Eigen::Stride<BATCHSIZE, 1>>;

        inline void operator()(Arg ret, Arg arg1) { ret = -arg1; }

        template <typename... Args>
        inline void operator()(Arg ret, Arg arg1, Args... args) { ret = arg1 - (args + ...); }
    };

    template <typename T>
    struct op<T, Operon::NodeType::Mul> {
        using Arg = Eigen::Ref<typename Eigen::DenseBase<Eigen::Array<T, BATCHSIZE, Eigen::Dynamic, Eigen::ColMajor>>::ColXpr, Eigen::Unaligned, Eigen::Stride<BATCHSIZE, 1>>;

        inline void operator()(Arg ret, Arg arg1) { ret = arg1; }

        template <typename... Args>
        inline void operator()(Arg ret, Arg arg1, Args... args) { ret = arg1 * (args * ...); }
    };

    template <typename T>
    struct op<T, Operon::NodeType::Div> {
        using Arg = Eigen::Ref<typename Eigen::DenseBase<Eigen::Array<T, BATCHSIZE, Eigen::Dynamic, Eigen::ColMajor>>::ColXpr, Eigen::Unaligned, Eigen::Stride<BATCHSIZE, 1>>;

        inline void operator()(Arg ret, Arg arg1) { ret = arg1.inverse(); }

        template <typename... Args>
        inline void operator()(Arg ret, Arg arg1, Args... args) { ret = arg1 / (args * ...); }
    };

    // dispatching mechanism
    template <typename T, Operon::NodeType N>
    inline void dispatch_op(Eigen::DenseBase<Eigen::Array<T, BATCHSIZE, Eigen::Dynamic, Eigen::ColMajor>>& m, Operon::Vector<Node> const& nodes, size_t parentIndex)
    {
        op<T, N> op;
        auto r = m.col(parentIndex);

        int arity = nodes[parentIndex].Arity;
        auto i = parentIndex - 1;

        // a single arg must be handled separately due to Sub and Div operations
        if (arity == 1) {
            op(r, m.col(i));
            return;
        }

        // if we're consuming the arity in a loop there is no special case just stop when arity is zero
        auto nextSibling = [&](size_t i) { return i - (nodes[i].Length + 1); };

        while (arity > 0) {
            switch (arity) {
            case 1: {
                op(r, r, m.col(i));
                arity = 0;
                break;
            }
            case 2: {
                auto j1 = nextSibling(i);
                op(r, m.col(i), m.col(j1));
                arity = 0;
                break;
            }
            case 3: {
                auto j1 = nextSibling(i), j2 = nextSibling(j1);
                op(r, m.col(i), m.col(j1), m.col(j2));
                arity = 0;
                break;
            }
            case 4: {
                auto j1 = nextSibling(i), j2 = nextSibling(j1), j3 = nextSibling(j2);
                op(r, m.col(i), m.col(j1), m.col(j2), m.col(j3));
                arity = 0;
                break;
            }
            default: {
                auto j1 = nextSibling(i), j2 = nextSibling(j1), j3 = nextSibling(j2), j4 = nextSibling(j3);
                op(r, m.col(i), m.col(j1), m.col(j2), m.col(j3), m.col(j4));
                arity -= 5;
                if (arity > 0)
                    i = nextSibling(j4);
                break;
            }
            }
        }
    }

    template <typename T, Operon::NodeType N>
    inline void dispatch_op_simple_binary(Eigen::DenseBase<Eigen::Array<T, BATCHSIZE, Eigen::Dynamic, Eigen::ColMajor>>& m, Operon::Vector<Node> const& nodes, size_t parentIndex)
    {
        op<T, N> op;
        auto r = m.col(parentIndex);
        size_t i = parentIndex - 1;
        size_t arity = nodes[parentIndex].Arity;

        if (arity == 1) {
            op(r, m.col(i));
        } else {
            auto j = i - (nodes[i].Length + 1);
            op(r, m.col(i), m.col(j));
        }
    }

    template <typename T, Operon::NodeType N>
    inline void dispatch_op_simple_nary(Eigen::DenseBase<Eigen::Array<T, BATCHSIZE, Eigen::Dynamic, Eigen::ColMajor>>& m, Operon::Vector<Node> const& nodes, size_t parentIndex)
    {
        op<T, N> op;
        auto r = m.col(parentIndex);
        size_t arity = nodes[parentIndex].Arity;

        auto i = parentIndex - 1;

        r = m.col(i);

        for (size_t k = 1; k < arity; ++k) {
            i -= nodes[i].Length + 1;
            op(r, r, m.col(i));
        }
    }

} // namespace detail
} // namespace Operon

#endif
