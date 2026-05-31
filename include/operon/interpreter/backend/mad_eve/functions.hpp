// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_BACKEND_MAD_EVE_FUNCTIONS_HPP
#define OPERON_BACKEND_MAD_EVE_FUNCTIONS_HPP

#include <eve/module/algo.hpp>
#include <eve/module/core.hpp>
#include <eve/module/math.hpp>

#include "mad/impl/inv.hpp"
#include "mad/impl/exp.hpp"
#include "mad/impl/log.hpp"
#include "mad/impl/pow.hpp"
#include "mad/impl/sqrt.hpp"
#include "mad/impl/tanh.hpp"
#include "mad/impl/trig.hpp"

namespace Operon::Backend {

// Precision levels for each approximated operation.
// P=0: fastest / lowest accuracy; P=1: one Newton-Raphson refinement step.
struct MadPrecision {
    static constexpr auto Div  = 1;
    static constexpr auto Exp  = 1;
    static constexpr auto Log  = 1;
    static constexpr auto Sqrt = 1;
    static constexpr auto Sin  = 0;
    static constexpr auto Cos  = 0;
    static constexpr auto Tan  = 0;
    static constexpr auto Tanh = 0;
    static constexpr auto Pow  = 0;
    static constexpr auto Inv  = 1;
};

    // utility
    template<typename T, std::size_t S>
    auto Fill(T* res, T value) {
        std::ranges::fill_n(res, S, value);
    }

    template<typename T, std::size_t S>
    auto Fill(T* res, int n, T value) {
        std::ranges::fill_n(res, n, value);
    }

    // n-ary functions
    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Add(T* res, T weight, auto const*... args) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * (W{args+i} + ...), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Mul(T* res, T weight, auto const*... args) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * (W{args+i} * ...), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Sub(T* res, T weight, auto const*... args) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::sub(W{args+i}...), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Div(T* res, T weight, auto const*... args) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            if constexpr (std::is_same_v<T, float>) {
                // left-fold: div_impl takes exactly 2 args, so unpack first + rest
                auto result = [i](auto const* first, auto const*... rest) {
                    auto q = W{first+i};
                    ((q = Mad::div_impl<MadPrecision::Div>(q, W{rest+i})), ...);
                    return q;
                }(args...);
                eve::store(weight * result, res+i);
            } else {
                eve::store(weight * eve::div(W{args+i}...), res+i);
            }
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Min(T* res, T weight, auto const*... args) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::min(W{args+i}...), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Max(T* res, T weight, auto const*... args) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::max(W{args+i}...), res+i);
        }
    }

    // binary functions
    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Aq(T* res, T weight, T const* a, T const* b) {
        eve::algo::transform_to(eve::views::zip(std::span{a, S}, b), res, [weight](auto t) {
            return weight * eve::get<0>(t) / eve::sqrt(T{1} + eve::sqr(eve::get<1>(t)));
        });
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Pow(T* res, T weight, T const* a, T const* b) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            if constexpr (std::is_same_v<T, float>) {
                eve::store(weight * Mad::pow_impl<MadPrecision::Pow>(W{a+i}, W{b+i}), res+i);
            } else {
                eve::store(weight * eve::pow(W{a+i}, W{b+i}), res+i);
            }
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Powabs(T* res, T weight, T const* a, T const* b) {
        eve::algo::transform_to(eve::views::zip(std::span{a, S}, b), res, [weight](auto t) {
            return weight * eve::pow(eve::abs(eve::get<0>(t)), eve::get<1>(t));
        });
    }

    // unary functions
    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Cpy(T* res, T weight, T const* arg) {
        std::transform(arg, arg+S, res, [weight](auto x) { return weight * x; });
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Neg(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::minus(W{arg+i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Inv(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            if constexpr (std::is_same_v<T, float>) {
                eve::store(weight * Mad::inv_impl<MadPrecision::Inv>(W{arg+i}), res+i);
            } else {
                eve::store(weight * eve::rec(W{arg+i}), res+i);
            }
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Abs(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::abs(W{arg+i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Ceil(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::ceil(W{arg+i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Floor(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::floor(W{arg+i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Square(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::sqr(W{arg+i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Exp(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            if constexpr (std::is_same_v<T, float>) {
                eve::store(weight * Mad::exp_impl<MadPrecision::Exp>(W{arg+i}), res+i);
            } else {
                eve::store(weight * eve::exp(W{arg+i}), res+i);
            }
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Log(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            if constexpr (std::is_same_v<T, float>) {
                eve::store(weight * Mad::log_impl<MadPrecision::Log>(W{arg+i}), res+i);
            } else {
                eve::store(weight * eve::log(W{arg+i}), res+i);
            }
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Log1p(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            if constexpr (std::is_same_v<T, float>) {
                eve::store(weight * Mad::log1p_impl<MadPrecision::Log>(W{arg+i}), res+i);
            } else {
                eve::store(weight * eve::log1p(W{arg+i}), res+i);
            }
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Logabs(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            if constexpr (std::is_same_v<T, float>) {
                eve::store(weight * Mad::logabs_impl<MadPrecision::Log>(W{arg+i}), res+i);
            } else {
                eve::store(weight * eve::log(eve::abs(W{arg+i})), res+i);
            }
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Sin(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            if constexpr (std::is_same_v<T, float>) {
                eve::store(weight * Mad::sin_impl<MadPrecision::Sin>(W{arg+i}), res+i);
            } else {
                eve::store(weight * eve::sin(W{arg+i}), res+i);
            }
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Cos(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            if constexpr (std::is_same_v<T, float>) {
                eve::store(weight * Mad::cos_impl<MadPrecision::Cos>(W{arg+i}), res+i);
            } else {
                eve::store(weight * eve::cos(W{arg+i}), res+i);
            }
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Tan(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            if constexpr (std::is_same_v<T, float>) {
                eve::store(weight * Mad::tan_impl<MadPrecision::Tan>(W{arg+i}), res+i);
            } else {
                eve::store(weight * eve::tan(W{arg+i}), res+i);
            }
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Asin(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::asin(W{arg+i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Acos(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::acos(W{arg+i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Atan(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::atan(W{arg+i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Sinh(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::sinh(W{arg+i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Cosh(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::cosh(W{arg+i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Tanh(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            if constexpr (std::is_same_v<T, float>) {
                eve::store(weight * Mad::tanh_impl<MadPrecision::Tanh>(W{arg+i}), res+i);
            } else {
                eve::store(weight * eve::tanh(W{arg+i}), res+i);
            }
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Sqrt(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            if constexpr (std::is_same_v<T, float>) {
                eve::store(weight * Mad::sqrt_impl<MadPrecision::Sqrt>(W{arg+i}), res+i);
            } else {
                eve::store(weight * eve::sqrt(W{arg+i}), res+i);
            }
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Sqrtabs(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            if constexpr (std::is_same_v<T, float>) {
                eve::store(weight * Mad::sqrt_impl<MadPrecision::Sqrt>(eve::abs(W{arg+i})), res+i);
            } else {
                eve::store(weight * eve::sqrt(eve::abs(W{arg+i})), res+i);
            }
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Cbrt(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::cbrt(W{arg+i}), res+i);
        }
    }

} // namespace Operon::Backend

#endif
