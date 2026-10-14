// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <eve/wide.hpp>

#include "operon/core/jet.hpp"
#include "operon/core/node.hpp"
#include "operon/interpreter/backend/eve/jet_math.hpp"
#include "operon/interpreter/functions.hpp"

namespace Operon::Test {
namespace {
    constexpr auto S = Operon::Backend::BatchSize<float>;
    constexpr auto NaN = std::numeric_limits<double>::quiet_NaN();

    using J1 = Operon::Jet<double, 1>;
    using J2 = Operon::Jet<double, 2>;

    // Scalar counterparts of the Operon-specific functions, so that one
    // generic callable below serves as the Jet function and as its scalar
    // reference (for finite differences): Jet arguments resolve to the
    // Operon::* overloads through ADL, plain floating-point arguments to these.
    namespace Ref {
        auto square(double x) -> double { return x * x; }
        auto logabs(double x) -> double { return std::log(std::abs(x)); }
        auto sqrtabs(double x) -> double { return std::sqrt(std::abs(x)); }
        auto aq(double x, double y) -> double { return x / std::sqrt(1.0 + y * y); }
        auto powabs(double x, double y) -> double { return std::pow(std::abs(x), y); }
    } // namespace Ref

    auto Near(double actual, double expected, double rtol) -> bool
    {
        if (std::isnan(actual) || std::isnan(expected)) {
            return std::isnan(actual) && std::isnan(expected);
        }
        if (std::isinf(actual) || std::isinf(expected)) {
            return actual == expected;
        }
        return std::abs(actual - expected) <= rtol * std::max(1.0, std::abs(expected));
    }

    // The backend kernels run in float on float-rounded inputs.
    constexpr auto BackendRtol = 2e-4;
    auto Round(double x) -> double { return static_cast<double>(static_cast<float>(x)); }

    template <typename F> auto CentralDiff(F f, double x) -> double
    {
        auto const h = 1e-5 * std::max(1.0, std::abs(x));
        return (f(x + h) - f(x - h)) / (2 * h);
    }

    // Derivative kernels of the active interpreter backend (Backend::Sin etc.).
    // They are exercised in float, the interpreter's scalar type (their fast
    // approximations are only meaningful there), at S lane values: the child
    // value columns are filled by the helpers below and the kernel's
    // derivative lands in the column of the child it is asked for.
    using Kernel = std::function<void(std::vector<Operon::Node> const&, Operon::Backend::View<float const, S>,
        Operon::Backend::View<float, S>, int, int)>;

    // d/dx at the given points; one child (column 0) and the node (column 1).
    auto BackendUnary(Kernel const& kernel, std::vector<double> const& xs) -> std::vector<double>
    {
        std::vector<float> primal(2 * S, 0.0F);
        std::vector<float> trace(2 * S, 0.0F);
        std::ranges::transform(xs, primal.begin(), [](double x) { return static_cast<float>(x); });
        Operon::Backend::View<float const, S> pv(primal.data(), S, 2);
        Operon::Backend::View<float, S> tv(trace.data(), S, 2);
        std::vector<Operon::Node> const nodes(2, Operon::Node::Constant(1.0));
        kernel(nodes, pv, tv, 1, 0);
        return { trace.begin(), trace.begin() + static_cast<std::ptrdiff_t>(xs.size()) };
    }

    // d/dx (child == 1, the child at i - 1) or d/dy (child == 0) of a binary
    // node with columns [y, x, f(x, y)].
    auto BackendBinary(Kernel const& kernel, double x, double y, double fxy, int child) -> double
    {
        std::vector<float> primal(3 * S, 0.0F);
        std::vector<float> trace(3 * S, 0.0F);
        std::ranges::fill_n(primal.begin(), S, static_cast<float>(y));
        std::ranges::fill_n(primal.begin() + S, S, static_cast<float>(x));
        std::ranges::fill_n(primal.begin() + 2 * S, S, static_cast<float>(fxy));
        Operon::Backend::View<float const, S> pv(primal.data(), S, 3);
        Operon::Backend::View<float, S> tv(trace.data(), S, 3);
        std::vector<Operon::Node> const nodes(3, Operon::Node::Constant(1.0));
        kernel(nodes, pv, tv, 2, child);
        return static_cast<double>(trace[static_cast<std::size_t>(child) * S]);
    }

    // Kernels are overloaded templates with deduced index types, so they are
    // wrapped by name here instead of passed as function pointers.
#define JET_KERNEL(Name)                                                                                               \
    Kernel([](auto const& n, auto pv, auto tv, int i, int j) { Operon::Backend::Name<float, S>(n, pv, tv, i, j); })

    template <typename F> struct UnaryCase {
        char const* name;
        F fn;
        std::vector<double> points;
        Kernel kernel;
    };
    template <typename F> UnaryCase(char const*, F, std::vector<double>, Kernel) -> UnaryCase<F>;

    template <typename F> struct BinaryCase {
        char const* name;
        F fn;
        std::vector<std::pair<double, double>> points;
        Kernel kernel; // empty: no backend kernel for this function
    };
    template <typename F> BinaryCase(char const*, F, std::vector<std::pair<double, double>>, Kernel) -> BinaryCase<F>;

    // clang-format off
    auto const UnaryCases = std::make_tuple(
        UnaryCase { "abs",    [](auto const& x) { using std::abs;   return abs(x); },   std::vector { -2.5, -0.3, 0.4, 3.1 },  JET_KERNEL(Abs) },
        UnaryCase { "acos",   [](auto const& x) { using std::acos;  return acos(x); },  std::vector { -0.8, -0.3, 0.2, 0.7 },  JET_KERNEL(Acos) },
        UnaryCase { "asin",   [](auto const& x) { using std::asin;  return asin(x); },  std::vector { -0.8, -0.3, 0.2, 0.7 },  JET_KERNEL(Asin) },
        UnaryCase { "atan",   [](auto const& x) { using std::atan;  return atan(x); },  std::vector { -3.0, -0.5, 0.8, 2.5 },  JET_KERNEL(Atan) },
        UnaryCase { "cbrt",   [](auto const& x) { using std::cbrt;  return cbrt(x); },  std::vector { -8.3, 0.5, 2.7, 9.1 },   JET_KERNEL(Cbrt) },
        UnaryCase { "ceil",   [](auto const& x) { using std::ceil;  return ceil(x); },  std::vector { -2.5, -0.3, 0.4, 3.1 },  JET_KERNEL(Ceil) },
        UnaryCase { "cos",    [](auto const& x) { using std::cos;   return cos(x); },   std::vector { -2.0, -0.5, 0.7, 3.3 },  JET_KERNEL(Cos) },
        UnaryCase { "cosh",   [](auto const& x) { using std::cosh;  return cosh(x); },  std::vector { -2.0, -0.5, 0.7, 1.9 },  JET_KERNEL(Cosh) },
        UnaryCase { "exp",    [](auto const& x) { using std::exp;   return exp(x); },   std::vector { -2.0, -0.5, 0.7, 1.9 },  JET_KERNEL(Exp) },
        UnaryCase { "floor",  [](auto const& x) { using std::floor; return floor(x); }, std::vector { -2.5, -0.3, 0.4, 3.1 },  JET_KERNEL(Floor) },
        UnaryCase { "log",    [](auto const& x) { using std::log;   return log(x); },   std::vector { 0.2, 0.9, 2.5, 7.0 },    JET_KERNEL(Log) },
        UnaryCase { "log1p",  [](auto const& x) { using std::log1p; return log1p(x); }, std::vector { -0.7, -0.2, 0.5, 3.0 },  JET_KERNEL(Log1p) },
        UnaryCase { "sin",    [](auto const& x) { using std::sin;   return sin(x); },   std::vector { -2.0, -0.5, 0.7, 3.3 },  JET_KERNEL(Sin) },
        UnaryCase { "sinh",   [](auto const& x) { using std::sinh;  return sinh(x); },  std::vector { -2.0, -0.5, 0.7, 1.9 },  JET_KERNEL(Sinh) },
        UnaryCase { "sqrt",   [](auto const& x) { using std::sqrt;  return sqrt(x); },  std::vector { 0.2, 0.9, 2.5, 7.0 },    JET_KERNEL(Sqrt) },
        UnaryCase { "tan",    [](auto const& x) { using std::tan;   return tan(x); },   std::vector { -1.2, -0.4, 0.6, 1.3 },  JET_KERNEL(Tan) },
        UnaryCase { "tanh",   [](auto const& x) { using std::tanh;  return tanh(x); },  std::vector { -2.0, -0.5, 0.7, 1.9 },  JET_KERNEL(Tanh) },
        UnaryCase { "square", [](auto const& x) { using Ref::square;  return square(x); },  std::vector { -3.0, -0.4, 0.5, 2.2 }, JET_KERNEL(Square) },
        UnaryCase { "logabs", [](auto const& x) { using Ref::logabs;  return logabs(x); },  std::vector { -3.0, -0.4, 0.5, 2.2 }, JET_KERNEL(Logabs) },
        UnaryCase { "sqrtabs",[](auto const& x) { using Ref::sqrtabs; return sqrtabs(x); }, std::vector { -3.0, -0.4, 0.5, 2.2 }, JET_KERNEL(Sqrtabs) });

    auto const BinaryCases = std::make_tuple(
        BinaryCase { "aq",     [](auto const& x, auto const& y) { using Ref::aq;     return aq(x, y); },     std::vector<std::pair<double, double>> { { -2.0, 1.5 }, { 0.7, -0.4 }, { 1.9, 2.3 }, { -0.5, -1.1 } }, JET_KERNEL(Aq) },
        BinaryCase { "pow",    [](auto const& x, auto const& y) { using std::pow;    return pow(x, y); },    std::vector<std::pair<double, double>> { { 1.5, 2.2 }, { 0.4, -1.3 }, { 2.9, 0.5 }, { 0.8, 3.0 } },     JET_KERNEL(Pow) },
        BinaryCase { "powabs", [](auto const& x, auto const& y) { using Ref::powabs; return powabs(x, y); }, std::vector<std::pair<double, double>> { { -1.5, 2.2 }, { 0.4, -1.3 }, { -2.9, 0.5 }, { 0.8, 3.0 } },  JET_KERNEL(Powabs) },
        BinaryCase { "fmin",   [](auto const& x, auto const& y) { using std::fmin;   return fmin(x, y); },   std::vector<std::pair<double, double>> { { 1.0, 2.0 }, { 3.0, -1.0 }, { -0.5, 0.5 } },                   JET_KERNEL(Min) },
        BinaryCase { "fmax",   [](auto const& x, auto const& y) { using std::fmax;   return fmax(x, y); },   std::vector<std::pair<double, double>> { { 1.0, 2.0 }, { 3.0, -1.0 }, { -0.5, 0.5 } },                   JET_KERNEL(Max) });
    // clang-format on

    template <typename Tuple, typename F> void ForEach(Tuple const& cases, F const& f)
    {
        std::apply([&](auto const&... c) { (f(c), ...); }, cases);
    }

    // Jet<T,N> evaluation of a function whose arguments are seeded in their own tangent directions.
    template <typename F> auto EvalUnary(F const& f, double x) -> J1 { return f(J1 { x, 0 }); }
    template <typename F> auto EvalBinary(F const& f, double x, double y) -> J2 { return f(J2 { x, 0 }, J2 { y, 1 }); }
} // namespace

TEST_CASE("Jet: unary functions match finite differences and the backend derivative kernels", "[jet]")
{
    constexpr auto rtol = 1e-6;
    ForEach(UnaryCases, [&](auto const& c) {
        DYNAMIC_SECTION(c.name)
        {
            auto const kernel = BackendUnary(c.kernel, c.points);
            for (auto i = 0UL; i < c.points.size(); ++i) {
                auto const x = c.points[i];
                auto const j = EvalUnary(c.fn, x);
                auto const jr = EvalUnary(c.fn, Round(x));
                INFO("x = " << x << ", jet tangent = " << jr.v[0] << ", backend kernel = " << kernel[i]);
                CHECK(Near(j.a, c.fn(x), 1e-12));
                CHECK(Near(j.v[0], CentralDiff([&](double t) { return c.fn(t); }, x), rtol));
                CHECK(Near(jr.v[0], kernel[i], BackendRtol));
            }
        }
    });
}

TEST_CASE("Jet: binary functions match finite differences and the backend derivative kernels", "[jet]")
{
    constexpr auto rtol = 1e-6;
    ForEach(BinaryCases, [&](auto const& c) {
        DYNAMIC_SECTION(c.name)
        {
            for (auto const& [x, y] : c.points) {
                auto const j = EvalBinary(c.fn, x, y);
                auto const fxy = c.fn(x, y);
                auto const jr = EvalBinary(c.fn, Round(x), Round(y));
                INFO("x = " << x << ", y = " << y);
                CHECK(Near(j.a, fxy, 1e-12));
                CHECK(Near(j.v[0], CentralDiff([&](double t) { return c.fn(t, y); }, x), rtol));
                CHECK(Near(j.v[1], CentralDiff([&](double t) { return c.fn(x, t); }, y), rtol));
                CHECK(Near(jr.v[0], BackendBinary(c.kernel, Round(x), Round(y), jr.a, 1), BackendRtol));
                CHECK(Near(jr.v[1], BackendBinary(c.kernel, Round(x), Round(y), jr.a, 0), BackendRtol));
            }
        }
    });
}

TEST_CASE("Jet: mixed Jet/scalar overloads of the binary functions", "[jet]")
{
    constexpr auto rtol = 1e-6;
    auto const check = [&](auto const& f) {
        for (auto const x : { 0.4, 1.3, 2.7 }) {
            auto const j = EvalUnary(f, x);
            CHECK(Near(j.a, f(x), 1e-12));
            CHECK(Near(j.v[0], CentralDiff(f, x), rtol));
        }
    };
    SECTION("pow(Jet, T)")
    {
        check([](auto const& x) {
            using std::pow;
            return pow(x, 1.7);
        });
    }
    SECTION("pow(T, Jet)")
    {
        check([](auto const& x) {
            using std::pow;
            return pow(2.3, x);
        });
    }
    SECTION("pow(Jet, int)")
    {
        check([](auto const& x) {
            using std::pow;
            return pow(x, 3);
        });
    }
    SECTION("powabs(Jet, T)")
    {
        check([](auto const& x) {
            using Ref::powabs;
            return powabs(x, 1.7);
        });
    }
    SECTION("powabs(T, Jet)")
    {
        check([](auto const& x) {
            using Ref::powabs;
            return powabs(-2.3, x);
        });
    }
    SECTION("aq(Jet, T)")
    {
        check([](auto const& x) {
            using Ref::aq;
            return aq(x, 0.7);
        });
    }
    SECTION("aq(T, Jet)")
    {
        check([](auto const& x) {
            using Ref::aq;
            return aq(1.9, x);
        });
    }
    SECTION("fmin(Jet, T)")
    {
        check([](auto const& x) {
            using std::fmin;
            return fmin(x, 1.0);
        });
    }
    SECTION("fmin(T, Jet)")
    {
        check([](auto const& x) {
            using std::fmin;
            return fmin(1.0, x);
        });
    }
    SECTION("fmax(Jet, T)")
    {
        check([](auto const& x) {
            using std::fmax;
            return fmax(x, 1.0);
        });
    }
    SECTION("fmax(T, Jet)")
    {
        check([](auto const& x) {
            using std::fmax;
            return fmax(1.0, x);
        });
    }

    SECTION("pow(Jet, T) is finite at a zero base where the exponent is a constant")
    {
        auto const j = Operon::pow(J1 { 0.0, 0 }, 2.0);
        CHECK(j.a == 0.0);
        CHECK(j.v[0] == 0.0);
    }
}

TEST_CASE("Jet: pow with a negative base and an integral exponent", "[jet]")
{
    // Value is real; d/dx = g * x^(g-1) regardless of whether the exponent is a constant Jet (zero tangent).
    SECTION("pow(Jet, Jet) with a zero exponent tangent is differentiable in the base")
    {
        auto const r = Operon::pow(J2 { -2.0, 0 }, J2 { 3.0 });
        CHECK(r.a == -8.0);
        CHECK(Near(r.v[0], 12.0, 1e-12)); // 3 * (-2)^2
        CHECK(r.v[1] == 0.0);

        auto const q = Operon::pow(J1 { -1.5, 0 }, J1 { 2.0 });
        CHECK(q.a == 2.25);
        CHECK(Near(q.v[0], -3.0, 1e-12)); // 2 * (-1.5)
    }

    SECTION("pow(Jet, Jet): a nonzero exponent tangent has no derivative (NaN), per tangent component")
    {
        auto const r = Operon::pow(J2 { -2.0, 0 }, J2 { 3.0, 1 });
        CHECK(r.a == -8.0);
        CHECK(Near(r.v[0], 12.0, 1e-12));
        CHECK(std::isnan(r.v[1]));
    }

    SECTION("pow(T, Jet) with a negative constant base")
    {
        auto const zero = Operon::pow(-2.0, J2 { 3.0 });
        CHECK(zero.a == -8.0);
        CHECK(zero.v[0] == 0.0);
        CHECK(zero.v[1] == 0.0);

        auto const seeded = Operon::pow(-2.0, J2 { 3.0, 1 });
        CHECK(seeded.a == -8.0);
        CHECK(seeded.v[0] == 0.0);
        CHECK(std::isnan(seeded.v[1]));
    }

    SECTION("a negative base with a non-integral exponent is not real-valued: NaN value and tangent")
    {
        auto const r = Operon::pow(J1 { -2.0, 0 }, J1 { 2.5 });
        CHECK(std::isnan(r.a));
        CHECK(std::isnan(r.v[0]));
    }
}

TEST_CASE("Jet: arithmetic with Jet and scalar operands in both orders", "[jet]")
{
    constexpr auto rtol = 1e-6;
    auto const f = [](auto const& x, auto const& y) {
        auto r = ((x + 2.0) * (y - 3.0)) / (x * y + 1.5) - 1.0 / x + y / 4.0 - 5.0 * x + (-x) + (+y) - (2.0 - y)
            + (x - y) * 0.25 + (x + y) / y;
        r += x;
        r -= y;
        r *= 1.5;
        r /= y;
        r += 0.5;
        r -= 0.25;
        r *= 2.0;
        r /= 4.0;
        return r;
    };
    for (auto const& [x, y] : std::vector<std::pair<double, double>> { { 1.3, 2.1 }, { -0.7, 0.9 }, { 2.4, -1.6 } }) {
        auto const j = EvalBinary(f, x, y);
        CHECK(Near(j.a, f(x, y), 1e-12));
        CHECK(Near(j.v[0], CentralDiff([&](double t) { return f(t, y); }, x), rtol));
        CHECK(Near(j.v[1], CentralDiff([&](double t) { return f(x, t); }, y), rtol));
    }
}

TEST_CASE("Jet: constructors, tangent layout and primal-only comparisons", "[jet]")
{
    using J3 = Operon::Jet<double, 3>;
    static_assert(J3::DIMENSION == 3);
    static_assert(std::is_same_v<J3::Scalar, double>);

    SECTION("default and value constructors have a zero tangent")
    {
        J3 const zero;
        CHECK(zero.a == 0.0);
        CHECK(zero.v == std::array<double, 3> { 0, 0, 0 });
        J3 const c { 4.5 };
        CHECK(c.a == 4.5);
        CHECK(c.v == std::array<double, 3> { 0, 0, 0 });
    }

    SECTION("seeding sets exactly one tangent component to one")
    {
        J3 const x { 4.5, 1 };
        CHECK(x.a == 4.5);
        CHECK(x.v == std::array<double, 3> { 0, 1, 0 });
    }

    SECTION("comparisons look at the primal part only")
    {
        J3 const a { 1.0, 0 };
        J3 const b { 2.0, 2 };
        CHECK(a < b);
        CHECK(a <= b);
        CHECK(b > a);
        CHECK(b >= a);
        CHECK(a != b);
        CHECK(a == J3 { 1.0, 1 });
        CHECK(a < 1.5);
        CHECK(1.5 < b);
        CHECK(a == 1.0);
        CHECK(2.0 == b);
        CHECK(b >= 2.0);
        CHECK(0.0 <= a);
    }
}

TEST_CASE("Jet: tangent directions are independent", "[jet]")
{
    using J3 = Operon::Jet<double, 3>;
    auto const f = [](auto const& x, auto const& y, auto const& z) {
        using std::exp;
        using std::sin;
        return x * sin(y) + exp(z) * x * y;
    };
    constexpr auto x0 = 0.7;
    constexpr auto y0 = -1.3;
    constexpr auto z0 = 0.4;

    SECTION("one pass seeds all three directions")
    {
        auto const r = f(J3 { x0, 0 }, J3 { y0, 1 }, J3 { z0, 2 });
        CHECK(Near(r.a, x0 * std::sin(y0) + std::exp(z0) * x0 * y0, 1e-12));
        CHECK(Near(r.v[0], std::sin(y0) + std::exp(z0) * y0, 1e-12));
        CHECK(Near(r.v[1], x0 * std::cos(y0) + std::exp(z0) * x0, 1e-12));
        CHECK(Near(r.v[2], std::exp(z0) * x0 * y0, 1e-12));
    }

    SECTION("an unseeded argument contributes no tangent")
    {
        auto const r = f(J3 { x0, 0 }, J3 { y0 }, J3 { z0 });
        CHECK(Near(r.v[0], std::sin(y0) + std::exp(z0) * y0, 1e-12));
        CHECK(r.v[1] == 0.0);
        CHECK(r.v[2] == 0.0);
    }

    SECTION("seeding the same variable in two components yields two copies of its derivative")
    {
        J3 const x { x0, std::array<double, 3> { 1.0, 0.0, 3.0 } };
        auto const r = x * x;
        CHECK(Near(r.v[0], 2 * x0, 1e-12));
        CHECK(r.v[1] == 0.0);
        CHECK(Near(r.v[2], 6 * x0, 1e-12));
    }

    SECTION("per-lane results equal N independent single-direction runs")
    {
        auto const all = f(J3 { x0, 0 }, J3 { y0, 1 }, J3 { z0, 2 });
        auto const dx = f(J1 { x0, 0 }, J1 { y0 }, J1 { z0 });
        auto const dy = f(J1 { x0 }, J1 { y0, 0 }, J1 { z0 });
        auto const dz = f(J1 { x0 }, J1 { y0 }, J1 { z0, 0 });
        CHECK(all.v[0] == dx.v[0]);
        CHECK(all.v[1] == dy.v[0]);
        CHECK(all.v[2] == dz.v[0]);
    }
}

TEST_CASE("Jet: user lambdas with using-declarations resolve the Jet overloads through ADL", "[jet]")
{
    // Written like the auto-diff fallback test in dispatch_table.cpp.
    auto const primal = [](auto const& v) -> auto {
        using std::sin, std::cos;
        return sin(v) + cos(v);
    };
    auto const x = 0.7;
    auto const r = primal(J1 { x, 0 });
    CHECK(Near(r.a, std::sin(x) + std::cos(x), 1e-12));
    CHECK(Near(r.v[0], std::cos(x) - std::sin(x), 1e-12));
    CHECK(primal(x) == std::sin(x) + std::cos(x)); // the same lambda still works on plain scalars

    auto const gaussian = [](auto const& v) -> auto {
        using std::exp;
        return exp(-v * v);
    };
    auto const g = gaussian(J1 { x, 0 });
    CHECK(Near(g.v[0], -2 * x * std::exp(-x * x), 1e-12));
}

TEST_CASE("Jet: tangent conventions at non-smooth points agree with the backend", "[jet]")
{
    auto const sameTangent
        = [](double actual, double expected) { return std::isnan(expected) ? std::isnan(actual) : actual == expected; };

    SECTION("abs has a zero tangent at zero and sgn(x) elsewhere")
    {
        auto const xs = std::vector { 0.0, -0.0, 1.5, -1.5 };
        auto const kernel = BackendUnary(JET_KERNEL(Abs), xs);
        for (auto i = 0UL; i < xs.size(); ++i) {
            CHECK(sameTangent(Operon::abs(J1 { xs[i], 0 }).v[0] + 0.0, kernel[i] + 0.0));
        }
        CHECK(Operon::abs(J1 { 0.0, 0 }).v[0] == 0.0);
    }

    SECTION("ceil and floor have zero tangents, also at integers")
    {
        auto const xs = std::vector { 2.0, -3.0, 0.0, 2.5, -2.5 };
        auto const ceilKernel = BackendUnary(JET_KERNEL(Ceil), xs);
        auto const floorKernel = BackendUnary(JET_KERNEL(Floor), xs);
        for (auto i = 0UL; i < xs.size(); ++i) {
            auto const c = Operon::ceil(J1 { xs[i], 0 });
            auto const f = Operon::floor(J1 { xs[i], 0 });
            CHECK(c.a == std::ceil(xs[i]));
            CHECK(f.a == std::floor(xs[i]));
            CHECK(c.v[0] == ceilKernel[i]);
            CHECK(f.v[0] == floorKernel[i]);
            CHECK(c.v[0] == 0.0);
            CHECK(f.v[0] == 0.0);
        }
    }

    SECTION("logabs and sqrtabs have a NaN tangent at zero (0/0), as in the backend")
    {
        auto const xs = std::vector { 0.0 };
        CHECK(std::isnan(BackendUnary(JET_KERNEL(Logabs), xs)[0]));
        CHECK(std::isnan(BackendUnary(JET_KERNEL(Sqrtabs), xs)[0]));
        CHECK(std::isnan(Operon::logabs(J1 { 0.0, 0 }).v[0]));
        CHECK(std::isnan(Operon::sqrtabs(J1 { 0.0, 0 }).v[0]));
    }

    SECTION("powabs carries the sgn(x) factor: zero tangent in x at x = 0")
    {
        // |x|^y is smooth at x = 0 for y > 1 with derivative 0 and, as for abs,
        // the tangent of |x| at 0 is 0 (y = 1). The backend kernel divides
        // 0 by 0 here (NaN), which the Jet deliberately does not reproduce.
        CHECK(Operon::powabs(J2 { 0.0, 0 }, J2 { 2.0, 1 }).v[0] == 0.0);
        CHECK(Operon::powabs(J2 { 0.0, 0 }, J2 { 1.0, 1 }).v[0] == 0.0);
        CHECK(Operon::powabs(J2 { 0.0, 0 }, 3.0).v[0] == 0.0);
        // the exponent partial |x|^y log|x| is 0, not 0 * -inf, where |x|^y == 0
        CHECK(Operon::powabs(J2 { 0.0, 0 }, J2 { 2.0, 1 }).v[1] == 0.0);
        CHECK(Operon::pow(J2 { 0.0, 0 }, J2 { 2.0, 1 }).v[0] == 0.0);
        CHECK(Operon::pow(J2 { 0.0, 0 }, J2 { 2.0, 1 }).v[1] == 0.0);
        // away from zero the sign of x selects the branch
        CHECK(Operon::powabs(J2 { -2.0, 0 }, J2 { 3.0, 1 }).v[0] < 0.0);
        CHECK(Operon::powabs(J2 { 2.0, 0 }, J2 { 3.0, 1 }).v[0] > 0.0);
    }

    SECTION("aq is differentiable at x = 0; the backend kernel divides 0 by 0 there")
    {
        auto const y = 1.5;
        auto const r = Operon::aq(J2 { 0.0, 0 }, J2 { y, 1 });
        CHECK(r.a == 0.0);
        CHECK(Near(r.v[0], 1.0 / std::sqrt(1.0 + y * y), 1e-12));
        CHECK(r.v[1] == 0.0); // d/dy [x / sqrt(1 + y^2)] = -x y / (1 + y^2)^(3/2) = 0 at x = 0
    }

    SECTION("fmin/fmax: ties and NaN operands select weights like the backend")
    {
        struct Case {
            double x;
            double y;
        };
        for (auto const& c : { Case { 1.0, 1.0 }, Case { 1.0, 2.0 }, Case { 2.0, 1.0 }, Case { NaN, 2.0 },
                 Case { 2.0, NaN }, Case { NaN, NaN } }) {
            INFO("x = " << c.x << ", y = " << c.y);
            auto const fmin = Operon::fmin(J2 { c.x, 0 }, J2 { c.y, 1 });
            auto const fmax = Operon::fmax(J2 { c.x, 0 }, J2 { c.y, 1 });
            CHECK(sameTangent(fmin.v[0], BackendBinary(JET_KERNEL(Min), c.x, c.y, std::fmin(c.x, c.y), 1)));
            CHECK(sameTangent(fmin.v[1], BackendBinary(JET_KERNEL(Min), c.x, c.y, std::fmin(c.x, c.y), 0)));
            CHECK(sameTangent(fmax.v[0], BackendBinary(JET_KERNEL(Max), c.x, c.y, std::fmax(c.x, c.y), 1)));
            CHECK(sameTangent(fmax.v[1], BackendBinary(JET_KERNEL(Max), c.x, c.y, std::fmax(c.x, c.y), 0)));
        }
        // spelled out: a tie is NaN in both operands; a NaN operand selects the other one
        CHECK(std::isnan(Operon::fmin(J2 { 1.0, 0 }, J2 { 1.0, 1 }).v[0]));
        CHECK(std::isnan(Operon::fmax(J2 { 1.0, 0 }, J2 { 1.0, 1 }).v[1]));
        auto const nanFirst = Operon::fmin(J2 { NaN, 0 }, J2 { 2.0, 1 });
        CHECK(nanFirst.a == 2.0);
        CHECK(nanFirst.v[0] == 0.0);
        CHECK(nanFirst.v[1] == 1.0);
    }
}

// ---- eve::wide scalar kind ----

namespace {
    using W = eve::wide<float>;
    constexpr auto Lanes = W::size();

    // Lane i takes point i % points.size().
    auto MakeWide(std::vector<double> const& points) -> W
    {
        return W { [&](auto i, auto) {
            return static_cast<float>(points[static_cast<std::size_t>(i) % points.size()]);
        } };
    }

    // A scalar constant of the same kind as x's primal (float or eve::wide<float>).
    template <typename J> auto K(J const& /*x*/, float v) -> typename J::Scalar { return typename J::Scalar(v); }

    auto NearF(float actual, float expected) -> bool
    {
        if (std::isnan(actual) || std::isnan(expected)) {
            return std::isnan(actual) && std::isnan(expected);
        }
        if (std::isinf(actual) || std::isinf(expected)) {
            return actual == expected;
        }
        return std::abs(actual - expected) <= 5e-5F * std::max(1.0F, std::abs(expected));
    }

    // Every lane of a Jet<W, N> result must equal the scalar Jet<float, N> result for that lane.
    template <int N>
    void CheckLanes(Operon::Jet<W, N> const& wide, std::array<Operon::Jet<float, N>, Lanes> const& scalar)
    {
        for (auto lane = 0UL; lane < Lanes; ++lane) {
            INFO("lane " << lane);
            CHECK(NearF(wide.a.get(lane), scalar[lane].a));
            for (auto k = 0; k < N; ++k) {
                INFO("tangent " << k);
                CHECK(
                    NearF(wide.v[static_cast<std::size_t>(k)].get(lane), scalar[lane].v[static_cast<std::size_t>(k)]));
            }
        }
    }

    template <int N, typename F> void CheckWideUnary(F const& f, std::vector<double> const& points)
    {
        using JW = Operon::Jet<W, N>;
        using JS = Operon::Jet<float, N>;
        auto const xs = MakeWide(points);
        // seed each tangent direction of the argument; with N > 1 later
        // components are seeded through a second argument-dependent term
        auto const x = JW { xs, 0 };
        std::array<JS, Lanes> expected;
        for (auto lane = 0UL; lane < Lanes; ++lane) {
            expected[lane] = f(JS { xs.get(lane), 0 });
        }
        CheckLanes<N>(f(x), expected);
    }

    template <int N, typename F> void CheckWideBinary(F const& f, std::vector<std::pair<double, double>> const& points)
    {
        using JW = Operon::Jet<W, N>;
        using JS = Operon::Jet<float, N>;
        std::vector<double> px;
        std::vector<double> py;
        for (auto const& [a, b] : points) {
            px.push_back(a);
            py.push_back(b);
        }
        auto const xs = MakeWide(px);
        auto const ys = MakeWide(py);
        std::array<JS, Lanes> expected;
        for (auto lane = 0UL; lane < Lanes; ++lane) {
            expected[lane] = f(JS { xs.get(lane), 0 }, JS { ys.get(lane), N - 1 });
        }
        CheckLanes<N>(f(JW { xs, 0 }, JW { ys, N - 1 }), expected);
    }
} // namespace

TEST_CASE("Jet<eve::wide>: every lane equals the scalar Jet result", "[jet][eve]")
{
    SECTION("unary functions, one tangent")
    {
        ForEach(UnaryCases, [&](auto const& c) {
            DYNAMIC_SECTION(c.name) { CheckWideUnary<1>(c.fn, c.points); }
        });
    }

    SECTION("unary functions, two tangents (second one unseeded)")
    {
        ForEach(UnaryCases, [&](auto const& c) {
            DYNAMIC_SECTION(c.name) { CheckWideUnary<2>(c.fn, c.points); }
        });
    }

    SECTION("binary functions, two tangents")
    {
        ForEach(BinaryCases, [&](auto const& c) {
            DYNAMIC_SECTION(c.name) { CheckWideBinary<2>(c.fn, c.points); }
        });
    }

    SECTION("binary functions, one tangent shared by both operands")
    {
        ForEach(BinaryCases, [&](auto const& c) {
            DYNAMIC_SECTION(c.name) { CheckWideBinary<1>(c.fn, c.points); }
        });
    }

    SECTION("mixed Jet/scalar overloads")
    {
        auto const points = std::vector { 0.4, 1.3, 2.7, 0.9 };
        CheckWideUnary<2>(
            [](auto const& x) {
                using std::pow;
                return pow(x, K(x, 1.7F));
            },
            points);
        CheckWideUnary<2>(
            [](auto const& x) {
                using std::pow;
                return pow(K(x, 2.3F), x);
            },
            points);
        CheckWideUnary<2>(
            [](auto const& x) {
                using Ref::powabs;
                return powabs(x, K(x, 1.7F));
            },
            points);
        CheckWideUnary<2>(
            [](auto const& x) {
                using Ref::aq;
                return aq(K(x, 1.9F), x);
            },
            points);
        CheckWideUnary<2>(
            [](auto const& x) {
                using std::fmin;
                return fmin(x, K(x, 1.0F));
            },
            points);
        CheckWideUnary<2>(
            [](auto const& x) {
                using std::fmax;
                return fmax(K(x, 1.0F), x);
            },
            points);
    }

    SECTION("arithmetic with scalar operands in both orders, comparisons and compound assignment")
    {
        auto const f = [](auto const& x, auto const& y) {
            auto const k = [&](float v) { return K(x, v); };
            auto r = ((x + k(2.0F)) * (y - k(3.0F))) / (x * y + k(1.5F)) - k(1.0F) / x + y / k(4.0F) - k(5.0F) * x
                + (-x) + (+y) - (k(2.0F) - y) + (x - y) * k(0.25F) + (x + y) / y;
            r += x;
            r -= y;
            r *= k(1.5F);
            r /= y;
            r += k(0.5F);
            r -= k(0.25F);
            return r;
        };
        CheckWideBinary<2>(f, { { 1.3, 2.1 }, { -0.7, 0.9 }, { 2.4, -1.6 }, { 0.5, 0.5 } });
    }

    SECTION("NaN weights of fmin/fmax ties are per lane")
    {
        // lane 0: tie -> NaN tangents; lane 1: x selected; lane 2: y selected; lane 3: NaN operand
        auto const f = [](auto const& x, auto const& y) {
            using std::fmin;
            return fmin(x, y);
        };
        CheckWideBinary<2>(f, { { 1.0, 1.0 }, { 1.0, 2.0 }, { 3.0, -1.0 }, { static_cast<double>(NaN), 2.0 } });
    }

    SECTION("pow with a negative base and an integral exponent is per lane")
    {
        // lanes 0/1: negative base with an integral exponent (tangent NaN only for the exponent component);
        // lanes 2/3: positive base, negative base with an even exponent
        auto const f = [](auto const& x, auto const& y) {
            using std::pow;
            return pow(x, y);
        };
        CheckWideBinary<2>(f, { { -2.0, 3.0 }, { -1.5, 2.0 }, { 2.0, 3.0 }, { -0.5, 4.0 } });
        CheckWideUnary<2>(
            [](auto const& x) {
                using std::pow;
                return pow(x, K(x, 3.0F));
            },
            std::vector { -2.0, -1.5, 2.0, -0.5 });
    }
}

// Eigen interop is a test-harness concern (jet_eigen_traits.hpp); the oracle
// tests in autodiff.cpp exercise it end to end.

} // namespace Operon::Test

#undef JET_KERNEL
