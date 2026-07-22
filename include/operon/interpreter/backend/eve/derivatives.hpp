// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_BACKEND_EVE_DERIVATIVES_HPP
#define OPERON_BACKEND_EVE_DERIVATIVES_HPP

#include "functions.hpp"

namespace Operon::Backend {
namespace detail {
    template<typename T>
    inline auto IsNaN(T value) { return eve::all(eve::is_nan(value)); }

    template<typename Compare>
    struct FComp {
        auto operator()(auto x, auto y) const {
            using T = std::common_type_t<decltype(x), decltype(y)>;
            if ((IsNaN(x) && IsNaN(y)) || eve::all(x == y)) {
                return std::numeric_limits<T>::quiet_NaN();
            }
            if (IsNaN(x)) { return T{0}; }
            if (IsNaN(y)) { return T{1}; }
            return static_cast<T>(Compare{}(T{x}, T{y}));
        }
    };
} // namespace detail

    template<typename T, std::size_t S = Backend::BatchSize<T>>
    auto Add(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> /*primal*/, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        std::ranges::fill_n(Ptr(trace, j), S, T{1});
    }

    // Every node's forward pass multiplies its result by the node's own
    // weight w, so primal[i] equals w * f(children), not f(children) alone.
    // This function must return the derivative of f(children) alone,
    // without w — something else multiplies by w again later. Dividing
    // primal[i] by w here gives that unweighted derivative.
    template<typename T, std::size_t S>
    auto Mul(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        using W = eve::wide<T>;
        auto constexpr L = W::size();
        auto const w = static_cast<T>(nodes[i].Value);
        auto* res = Ptr(trace, j);
        // If w is 0, primal[i] is 0 too, so the shortcut below computes 0/0,
        // which is not a number. The correct derivative here is exactly 0.
        // Returning 0 directly avoids the 0/0 case.
        if (w == T{0}) { std::fill_n(res, S, T{0}); return; }
        auto const* pi = Ptr(primal, i);
        auto const* pj = Ptr(primal, j);

        for(auto s = 0UL; s < S; s += L) {
            eve::store(W{pi+s} / (w * W{pj+s}), res+s);
        }
    }

    template<typename T, std::size_t S>
    auto Sub(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> /*primal*/, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto v = (nodes[i].Arity == 1 || j < i-1) ? T{-1} : T{+1};
        std::fill_n(Ptr(trace, j), S, v);
    }

    template<typename T, std::size_t S>
    auto Div(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        using W = eve::wide<T>;
        auto constexpr L = W::size();

        auto* res = Ptr(trace, j);
        auto const* pi = Ptr(primal, i);
        auto const* pj = Ptr(primal, j);
        if (nodes[i].Arity == 1) {
            for (auto s = 0UL; s < S; s += L) {
                eve::store(-eve::rec(eve::sqr(W{pj+s})), res+s);
            }
        } else {
            auto const w = static_cast<T>(nodes[i].Value);
            if (w == T{0}) { std::fill_n(res, S, T{0}); return; } // see Mul's w==0 comment
            auto v = j == i-1 ? T{1} : T{-1};
            for (auto s = 0UL; s < S; s += L) {
                eve::store(v * W{pi+s} / (w * W{pj+s}), res+s);
            }
        }
    }

    template<typename T, std::size_t S>
    auto Aq(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        using W = eve::wide<T>;
        static constexpr auto L = W::size();

        auto const w = static_cast<T>(nodes[i].Value);
        auto* res = Ptr(trace, j);
        if (w == T{0}) { std::fill_n(res, S, T{0}); return; } // see Mul's w==0 comment
        auto const k = i-1;
        auto const* pi = Ptr(primal, i);
        auto const* pj = Ptr(primal, j);
        auto const* pk = Ptr(primal, k);

        if (j == i-1) {
            for (auto s = 0UL; s < S; s += L) {
                eve::store(W{pi+s} / (w * W{pj+s}), res+s);
            }
        } else {
            auto const w3 = w * w * w;
            for (auto s = 0UL; s < S; s += L) {
                W a{pi+s};
                W b{pj+s};
                W c{pk+s};
                W r = -b * a * a * a / (w3 * c * c);
                eve::store(r, res+s);
            }
        }
    }

    template<typename T, std::size_t S>
    auto Pow(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        using W = eve::wide<T>;
        static constexpr auto L = W::size();

        auto const w = static_cast<T>(nodes[i].Value);
        auto* res = Ptr(trace, j);
        if (w == T{0}) { std::fill_n(res, S, T{0}); return; } // see Mul's w==0 comment
        auto const* pi = Ptr(primal, i);
        auto const* pj = Ptr(primal, j);
        if (j == i-1) {
            auto const k = j - (nodes[j].Length + 1);
            auto const* pk = Ptr(primal, k);
            for (auto s = 0UL; s < S; s += L) {
                eve::store(W{pi+s} * W{pk+s} / (w * W{pj+s}), res+s);
            }
        } else {
            auto const k = i-1;
            auto const* pk = Ptr(primal, k);
            for (auto s = 0UL; s < S; s += L) {
                eve::store(W{pi+s} * eve::log(W{pk+s}) / w, res+s);
            }
        }
    }

    template<typename T, std::size_t S>
    auto Powabs(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        using W = eve::wide<T>;
        static constexpr auto L = W::size();

        auto const w = static_cast<T>(nodes[i].Value);
        auto* res = Ptr(trace, j);
        if (w == T{0}) { std::fill_n(res, S, T{0}); return; } // see Mul's w==0 comment
        auto const* pi = Ptr(primal, i);
        auto const* pj = Ptr(primal, j);
        if (j == i-1) {
            auto const k = j - (nodes[j].Length + 1);
            auto const* pk = Ptr(primal, k);
            for (auto s = 0UL; s < S; s += L) {
                eve::store(W{pi+s} * W{pk+s} * eve::sign(W{pj+s}) / (w * eve::abs(W{pj+s})), res+s);
            }
        } else {
            auto const k = i-1;
            auto const* pk = Ptr(primal, k);
            for (auto s = 0UL; s < S; s += L) {
                eve::store(W{pi+s} * eve::log(eve::abs(W{pk+s})) / w, res+s);
            }
        }
    }

    template<typename T, std::size_t S>
    auto Min(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto k = j == i - 1 ? (j - nodes[j].Length - 1) : i - 1;
        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        auto const* pk = Ptr(primal, k);
        std::transform(pj, pj+S, pk, res, detail::FComp<std::less<>>{});
    }

    template<typename T, std::size_t S>
    auto Max(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto k = j == i - 1 ? (j - nodes[j].Length - 1) : i - 1;
        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        auto const* pk = Ptr(primal, k);
        std::transform(pj, pj+S, pk, res, detail::FComp<std::greater<>>{});
    }

    template<typename T, std::size_t S>
    auto Square(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        using W = eve::wide<T>;
        static constexpr auto L = W::size();

        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        for (auto s = 0UL; s < S; s += L) {
            eve::store(T{2} * W{pj+s}, res+s);
        }
    }

    template<typename T, std::size_t S>
    auto Abs(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        using W = eve::wide<T>;
        static constexpr auto L = W::size();

        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        for (auto s = 0UL; s < S; s += L) {
            eve::store(eve::sign(W{pj+s}), res+s);
        }
    }

    template<typename T, std::size_t S>
    auto Ceil(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> /*primal*/, Backend::View<T> trace, std::integral auto  /*i*/, std::integral auto j) {
        // Derivative is zero a.e.; provides no gradient information (cf. Ceres jet.h).
        std::fill_n(Ptr(trace, j), S, T{0});
    }

    template<typename T, std::size_t S>
    auto Floor(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> /*primal*/, Backend::View<T> trace, std::integral auto  /*i*/, std::integral auto j) {
        // Derivative is zero a.e.; provides no gradient information (cf. Ceres jet.h).
        std::fill_n(Ptr(trace, j), S, T{0});
    }

    // Computes the derivative directly from the child's value, using the
    // same FastExp approximation the forward pass uses. Mul/Div/Aq/Pow/
    // Powabs above can't do this: their derivative also needs a sibling
    // child's value, so they divide this node's own weighted result by its
    // weight instead, which fails when the weight is 0. This function never
    // divides by the weight, so it has no such problem.
    template<typename T, std::size_t S>
    auto Exp(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        using W = eve::wide<T>;
        static constexpr auto L = W::size();
        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        for (auto s = 0UL; s < S; s += L) {
            eve::store(detail::FastExp(W{pj+s}), res+s);
        }
    }

    template<typename T, std::size_t S>
    auto Log(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        using W = eve::wide<T>;
        static constexpr auto L = W::size();

        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        for (auto s = 0UL; s < S; s += L) {
            eve::store(eve::rec(W{pj+s}), res+s);
        }
    }

    template<typename T, std::size_t S>
    auto Log1p(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        using W = eve::wide<T>;
        static constexpr auto L = W::size();

        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        for (auto s = 0UL; s < S; s += L) {
            eve::store(eve::rec(T{1} + W{pj+s}), res+s);
        }
    }

    template<typename T, std::size_t S>
    auto Logabs(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        using W = eve::wide<T>;
        static constexpr auto L = W::size();

        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        for (auto s = 0UL; s < S; s += L) {
            eve::store(eve::sign(W{pj+s}) / eve::abs(W{pj+s}), res+s);
        }
    }

    template<typename T, std::size_t S>
    auto Sin(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        using W = eve::wide<T>;
        static constexpr auto L = W::size();

        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        for (auto s = 0UL; s < S; s += L) {
            eve::store(eve::cos(W{pj+s}), res+s);
        }
    }

    template<typename T, std::size_t S>
    auto Cos(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        using W = eve::wide<T>;
        static constexpr auto L = W::size();

        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        for (auto s = 0UL; s < S; s += L) {
            eve::store(-eve::sin(W{pj+s}), res+s);
        }
    }

    template<typename T, std::size_t S>
    auto Tan(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        using W = eve::wide<T>;
        static constexpr auto L = W::size();

        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        for (auto s = 0UL; s < S; s += L) {
            eve::store(T{1} + eve::sqr(eve::tan(W{pj+s})), res+s);
        }
    }

    template<typename T, std::size_t S>
    auto Sinh(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        using W = eve::wide<T>;
        static constexpr auto L = W::size();

        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        for (auto s = 0UL; s < S; s += L) {
            eve::store(eve::cosh(W{pj+s}), res+s);
        }
    }

    template<typename T, std::size_t S>
    auto Cosh(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        using W = eve::wide<T>;
        static constexpr auto L = W::size();

        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        for (auto s = 0UL; s < S; s += L) {
            eve::store(eve::sinh(W{pj+s}), res+s);
        }
    }

    template<typename T, std::size_t S>
    auto Tanh(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        using W = eve::wide<T>;
        static constexpr auto L = W::size();

        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        for (auto s = 0UL; s < S; s += L) {
            eve::store(T{1} - eve::sqr(eve::tanh(W{pj+s})), res+s);
        }
    }

    template<typename T, std::size_t S>
    auto Asin(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        using W = eve::wide<T>;
        static constexpr auto L = W::size();

        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        for (auto s = 0UL; s < S; s += L) {
            eve::store(eve::rec(eve::sqrt(T{1} - eve::sqr(W{pj+s}))), res+s);
        }
    }

    template<typename T, std::size_t S>
    auto Acos(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        using W = eve::wide<T>;
        static constexpr auto L = W::size();

        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        for (auto s = 0UL; s < S; s += L) {
            eve::store(-eve::rec(eve::sqrt(T{1} - eve::sqr(W{pj+s}))), res+s);
        }
    }

    template<typename T, std::size_t S>
    auto Atan(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        using W = eve::wide<T>;
        static constexpr auto L = W::size();

        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        for (auto s = 0UL; s < S; s += L) {
            eve::store(eve::rec(T{1} + eve::sqr(W{pj+s})), res+s);
        }
    }

    template<typename T, std::size_t S>
    auto Sqrt(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        using W = eve::wide<T>;
        static constexpr auto L = W::size();

        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        for (auto s = 0UL; s < S; s += L) {
            eve::store(eve::rec(T{2} * eve::sqrt(W{pj+s})), res+s);
        }
    }

    template<typename T, std::size_t S>
    auto Sqrtabs(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        using W = eve::wide<T>;
        static constexpr auto L = W::size();

        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        for (auto s = 0UL; s < S; s += L) {
            eve::store(eve::sign(W{pj+s}) / (T{2} * eve::sqrt(eve::abs(W{pj+s}))), res+s);
        }
    }

    template<typename T, std::size_t S>
    auto Cbrt(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        using W = eve::wide<T>;
        static constexpr auto L = W::size();

        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        for (auto s = 0UL; s < S; s += L) {
            eve::store(eve::rec(T{3} * eve::sqr(eve::cbrt(W{pj+s}))), res+s);
        }
    }
}  // namespace Operon::Backend

#endif
