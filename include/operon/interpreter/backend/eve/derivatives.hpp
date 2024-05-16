// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

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

    template<typename T, std::size_t S>
    auto Mul(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        using W = eve::wide<T>;
        auto constexpr L = W::size();

        auto* res = Ptr(trace, j);
        auto const* pi = Ptr(primal, i);
        auto const* pj = Ptr(primal, j);

        for(auto s = 0UL; s < S; s += L) {
            eve::store(W{pi+s} / W{pj+s}, res+s);
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
            auto v = j == i-1 ? T{1} : T{-1};
            for (auto s = 0UL; s < S; s += L) {
                eve::store(v * W{pi+s} / W{pj+s}, res+s);
            }
        }
    }

    template<typename T, std::size_t S>
    auto Aq(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        using W = eve::wide<T>;
        static constexpr auto L = W::size();

        auto* res = Ptr(trace, j);
        auto const k = i-1;
        auto const* pi = Ptr(primal, i);
        auto const* pj = Ptr(primal, j);
        auto const* pk = Ptr(primal, k);

        if (j == i-1) {
            for (auto s = 0UL; s < S; s += L) {
                eve::store(W{pi+s} / W{pj+s}, res+s);
            }
        } else {
            for (auto s = 0UL; s < S; s += L) {
                W a{pi+s};
                W b{pj+s};
                W c{pk+s};
                W r = -b * a * a * a / (c * c);
                eve::store(r, res+s);
            }
        }
    }

    template<typename T, std::size_t S>
    auto Pow(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        using W = eve::wide<T>;
        static constexpr auto L = W::size();

        auto* res = Ptr(trace, j);
        auto const* pi = Ptr(primal, i);
        auto const* pj = Ptr(primal, j);
        if (j == i-1) {
            auto const k = j - (nodes[j].Length + 1);
            auto const* pk = Ptr(primal, k);
            for (auto s = 0UL; s < S; s += L) {
                eve::store(W{pi+s} * W{pk+s} / W{pj+s}, res+s);
            }
        } else {
            auto const k = i-1;
            auto const* pk = Ptr(primal, k);
            for (auto s = 0UL; s < S; s += L) {
                eve::store(W{pi+s} * eve::log(W{pk+s}), res+s);
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
    auto Ceil(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto  /*i*/, std::integral auto j) {
        using W = eve::wide<T>;
        static constexpr auto L = W::size();

        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        for (auto s = 0UL; s < S; s += L) {
            eve::store(eve::ceil(W{pj+s}), res+s);
        }
    }

    template<typename T, std::size_t S>
    auto Floor(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto  /*i*/, std::integral auto j) {
        using W = eve::wide<T>;
        static constexpr auto L = W::size();

        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        for (auto s = 0UL; s < S; s += L) {
            eve::store(eve::floor(W{pj+s}), res+s);
        }
    }

    template<typename T, std::size_t S>
    auto Exp(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        std::ranges::copy_n(Ptr(primal, i), S, Ptr(trace, j));
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
    auto Sqrt(std::vector<Operon::Node> const&  /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        using W = eve::wide<T>;
        static constexpr auto L = W::size();

        auto* res = Ptr(trace, j);
        auto const* pi = Ptr(primal, i);
        for (auto s = 0UL; s < S; s += L) {
            eve::store(eve::rec(T{2} * W{pi+s}), res+s);
        }
    }

    template<typename T, std::size_t S>
    auto Sqrtabs(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        using W = eve::wide<T>;
        static constexpr auto L = W::size();

        auto* res = Ptr(trace, j);
        auto const* pi = Ptr(primal, i);
        auto const* pj = Ptr(primal, j);
        for (auto s = 0UL; s < S; s += L) {
            eve::store(eve::sign(W{pj+s}) / (T{2} * W{pi+s}), res+s);
        }
    }

    template<typename T, std::size_t S>
    auto Cbrt(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        using W = eve::wide<T>;
        static constexpr auto L = W::size();

        auto* res = Ptr(trace, j);
        auto const* pi = Ptr(primal, i);
        for (auto s = 0UL; s < S; s += L) {
            eve::store(eve::rec(T{3} * eve::sqr(W{pi+s})), res+s);
        }
    }
}  // namespace Operon::Backend

#endif
