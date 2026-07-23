// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/core/tree_diff.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <stdexcept>

#include "operon/core/contracts.hpp"
#include "operon/core/hash_registry.hpp"
#include "operon/core/subtree.hpp"

namespace Operon {

namespace {

using Nodes  = Operon::Vector<Node>;
using Hashes = Operon::Vector<Operon::Hash>;
using Memo   = Operon::Map<Operon::Hash, std::size_t>;

constexpr std::size_t Zero = std::numeric_limits<std::size_t>::max();

// Mix two 64-bit hashes (Boost-style but with a larger multiplier). Note:
// Mix(0, 0) == 0 is a fixed point of this formula (composed_function.hpp's
// DiffMix salts against exactly this, since it mixes directly against raw
// original-tree indices where h[0] == 0 is normal). It's out of reach here
// only because Deriv() below always combines *derivative*-dag hashes (never
// a raw original-tree index of 0) through this function — an invariant of
// how Deriv() is written, not something this function itself enforces.
auto Mix(uint64_t a, uint64_t b) -> uint64_t {
    return a ^ (b * 0x9e3779b97f4a7c15ULL + (a << 6U) + (a >> 2U));
}

// Append a derivative node (always Optimize=false) and record its hash.
void Push(Nodes& dag, Hashes& h, Node n, uint64_t hash) {
    n.Optimize = false;
    dag.push_back(n);
    h.push_back(hash);
}

// Append a Ref node pointing to `target`; NOT hash-consed (each occurrence
// occupies its own physical slot adjacent to the parent operator).
auto AppendRef(Nodes& dag, Hashes& h, std::size_t target) -> std::size_t {
    EXPECT(target <= std::numeric_limits<uint16_t>::max());
    auto hash = Mix(static_cast<uint64_t>(NodeType::Ref), h[target]);
    Push(dag, h, Node::Ref(static_cast<uint16_t>(target)), hash);
    return dag.size() - 1;
}

// Get or create an unweighted Variable copy of `origNode` (Value=1, Optimize=false).
// Used for d(w * X_i)/dw = X_i: the unweighted column without the learned weight.
auto GetVar(Nodes& dag, Memo& memo, Hashes& h,
            Node const& origNode, std::size_t origIdx) -> std::size_t {
    auto varHash = Mix(static_cast<uint64_t>(NodeType::Variable),
                       Mix(origNode.HashValue, ~origIdx));
    if (auto it = memo.find(varHash); it != memo.end()) { return it->second; }
    Node col     = origNode;
    col.Value    = 1.0F;
    col.Optimize = false;
    auto idx     = dag.size();
    Push(dag, h, col, varHash);
    memo.insert_or_assign(varHash, idx);
    return idx;
}

// Get or create a constant node with the given value (hash-consed by value).
auto GetConst(Nodes& dag, Memo& memo, Hashes& h, Scalar val) -> std::size_t {
    auto hash = Mix(static_cast<uint64_t>(NodeType::Constant),
                    std::bit_cast<uint64_t>(static_cast<double>(val)));
    if (auto it = memo.find(hash); it != memo.end()) { return it->second; }
    auto idx = dag.size();
    Push(dag, h, Node::Constant(val), hash);
    memo.insert_or_assign(hash, idx);
    return idx;
}

// Build unary op(a) where `a` is an existing dag index.
// Layout appended: [Ref(a), UnaryOp]
// Hash-consed so that identical expressions share a single node.
auto MakeUnary(Nodes& dag, Memo& memo, Hashes& h,
               BuiltinOp op, std::size_t a) -> std::size_t {
    auto opHash = Mix(static_cast<uint64_t>(op), h[a]);
    if (auto it = memo.find(opHash); it != memo.end()) { return it->second; }
    AppendRef(dag, h, a);
    auto n = Node::Function(static_cast<Operon::Hash>(op), 1); // sets Arity=Length=1
    auto idx = dag.size();
    Push(dag, h, n, opHash);
    memo.insert_or_assign(opHash, idx);
    return idx;
}

// Build bin_op(a, b) where `a` is the "first/nearer" child (j = parent-1)
// and `b` is the "second/farther" child (k) in postfix order.
// For non-commutative ops: Sub(a,b)=a-b, Div(a,b)=a/b, Pow(a,b)=a^b.
// Layout appended: [Ref(b), Ref(a), BinaryOp]
// Hash-consed; note that Mul(a,b) and Mul(b,a) get different hash keys
// (less deduplication but no correctness issue).
auto MakeBinary(Nodes& dag, Memo& memo, Hashes& h,
                BuiltinOp op, std::size_t a, std::size_t b) -> std::size_t {
    auto opHash = Mix(Mix(static_cast<uint64_t>(op), h[a]), h[b]);
    if (auto it = memo.find(opHash); it != memo.end()) { return it->second; }
    AppendRef(dag, h, b); // farther child first
    AppendRef(dag, h, a); // nearer child second
    auto n = Node::Function(static_cast<Operon::Hash>(op), 2); // sets Arity=Length=2
    auto idx = dag.size();
    Push(dag, h, n, opHash);
    memo.insert_or_assign(opHash, idx);
    return idx;
}

// Sum a list of non-zero terms via a left-associative chain of binary Add nodes.
auto AddTerms(Nodes& dag, Memo& memo, Hashes& h,
              Operon::Vector<std::size_t> const& terms) -> std::size_t {
    if (terms.empty()) { return Zero; }
    auto result = terms[0];
    for (std::size_t k = 1; k < terms.size(); ++k) {
        result = MakeBinary(dag, memo, h, BuiltinOp::Add, result, terms[k]);
    }
    return result;
}

// Registry of symbolic derivative rules for unary functions, keyed by
// Node::HashValue. A global, function-local-static singleton (mirroring
// Node::RegisterName/Descriptions() in node.cpp) so BuildJacobianDag/
// BuildHessianDag's public signature doesn't need a registry parameter
// threaded through every call site. Populated once with the built-in rules
// (RegisterBuiltinSymbolicDerivs, below) plus whatever user-defined rules
// Operon::RegisterUnarySymbolicDeriv() adds. All writes happen at setup
// time, before any parallel evaluation starts reading via Deriv() from
// worker threads — the same read-only-after-setup contract DispatchTable's
// own (non-locking) Operon::Map already relies on, so no sharded-lock map
// type is needed here.
using SymbolicDerivRegistry = HashRegistry<UnarySymbolicDerivRule>;

auto SymbolicDerivRules() -> SymbolicDerivRegistry&
{
    static SymbolicDerivRegistry registry;
    return registry;
}

// Binary counterpart. No built-ins registered here (unlike
// SymbolicDerivRules()) — every binary built-in is either hardcoded
// directly in Deriv() (Add/Mul/Sub/Div/Pow) or deliberately excluded
// (Aq/Powabs/Fmin/Fmax, "not yet differentiated") — so this registry
// exists purely for user-defined/composed binary functions, starting
// empty, no lazy-init-builtins step needed.
using BinarySymbolicDerivRegistry = HashRegistry<BinarySymbolicDerivRule>;

auto BinaryDerivRules() -> BinarySymbolicDerivRegistry&
{
    static BinarySymbolicDerivRegistry registry;
    return registry;
}

// Register the built-in unary derivative rules exactly once. Lives here
// (rather than in StandardLibrary, which has no visibility into this TU's
// private hash-consing helpers MakeUnary/MakeBinary/GetConst) mirroring
// StandardLibrary::RegisterNames()'s lazy-static-lambda-once pattern.
// Abs/Sqrtabs/Floor/Ceil are intentionally left unregistered (non-smooth or
// not yet implemented); an unregistered op yields Zero.
void RegisterBuiltinSymbolicDerivs()
{
    static auto const registered = [] {
        auto& reg = SymbolicDerivRules();

        reg.Register(Operon::Hash(BuiltinOp::Exp),
            [](Nodes& dag, Memo& memo, Hashes& h, std::size_t /*i*/, std::size_t j) -> std::size_t {
                return MakeUnary(dag, memo, h, BuiltinOp::Exp, j); // d exp(j)/dj = exp(j), recomputed fresh from j
            });

        UnarySymbolicDerivRule logRule =
            [](Nodes& dag, Memo& memo, Hashes& h, std::size_t /*i*/, std::size_t j) -> std::size_t {
                auto c1 = GetConst(dag, memo, h, Scalar{1});
                return MakeBinary(dag, memo, h, BuiltinOp::Div, c1, j);
            };
        reg.Register(Operon::Hash(BuiltinOp::Log), logRule);
        reg.Register(Operon::Hash(BuiltinOp::Logabs), logRule);

        reg.Register(Operon::Hash(BuiltinOp::Log1p),
            [](Nodes& dag, Memo& memo, Hashes& h, std::size_t /*i*/, std::size_t j) -> std::size_t {
                auto c1 = GetConst(dag, memo, h, Scalar{1});
                auto onePlusJ = MakeBinary(dag, memo, h, BuiltinOp::Add, c1, j);
                return MakeBinary(dag, memo, h, BuiltinOp::Div, c1, onePlusJ);
            });

        reg.Register(Operon::Hash(BuiltinOp::Sin),
            [](Nodes& dag, Memo& memo, Hashes& h, std::size_t /*i*/, std::size_t j) -> std::size_t {
                return MakeUnary(dag, memo, h, BuiltinOp::Cos, j);
            });

        reg.Register(Operon::Hash(BuiltinOp::Cos),
            [](Nodes& dag, Memo& memo, Hashes& h, std::size_t /*i*/, std::size_t j) -> std::size_t {
                auto sinJ = MakeUnary(dag, memo, h, BuiltinOp::Sin, j);
                auto neg1 = GetConst(dag, memo, h, Scalar{-1});
                return MakeBinary(dag, memo, h, BuiltinOp::Mul, neg1, sinJ);
            });

        reg.Register(Operon::Hash(BuiltinOp::Tan),
            [](Nodes& dag, Memo& memo, Hashes& h, std::size_t /*i*/, std::size_t j) -> std::size_t {
                auto c1   = GetConst(dag, memo, h, Scalar{1});
                auto tanJ = MakeUnary(dag, memo, h, BuiltinOp::Tan, j); // recomputed fresh from j
                auto sqJ  = MakeUnary(dag, memo, h, BuiltinOp::Square, tanJ);
                return MakeBinary(dag, memo, h, BuiltinOp::Add, c1, sqJ);
            });

        reg.Register(Operon::Hash(BuiltinOp::Acos),
            [](Nodes& dag, Memo& memo, Hashes& h, std::size_t /*i*/, std::size_t j) -> std::size_t {
                auto c1    = GetConst(dag, memo, h, Scalar{1});
                auto sqJ   = MakeUnary(dag, memo, h, BuiltinOp::Square, j);
                auto denom = MakeBinary(dag, memo, h, BuiltinOp::Sub, c1, sqJ);
                auto sqD   = MakeUnary(dag, memo, h, BuiltinOp::Sqrt, denom);
                auto recip = MakeBinary(dag, memo, h, BuiltinOp::Div, c1, sqD);
                auto neg1  = GetConst(dag, memo, h, Scalar{-1});
                return MakeBinary(dag, memo, h, BuiltinOp::Mul, neg1, recip);
            });

        reg.Register(Operon::Hash(BuiltinOp::Asin),
            [](Nodes& dag, Memo& memo, Hashes& h, std::size_t /*i*/, std::size_t j) -> std::size_t {
                auto c1    = GetConst(dag, memo, h, Scalar{1});
                auto sqJ   = MakeUnary(dag, memo, h, BuiltinOp::Square, j);
                auto denom = MakeBinary(dag, memo, h, BuiltinOp::Sub, c1, sqJ);
                auto sqD   = MakeUnary(dag, memo, h, BuiltinOp::Sqrt, denom);
                return MakeBinary(dag, memo, h, BuiltinOp::Div, c1, sqD);
            });

        reg.Register(Operon::Hash(BuiltinOp::Atan),
            [](Nodes& dag, Memo& memo, Hashes& h, std::size_t /*i*/, std::size_t j) -> std::size_t {
                auto c1    = GetConst(dag, memo, h, Scalar{1});
                auto sqJ   = MakeUnary(dag, memo, h, BuiltinOp::Square, j);
                auto denom = MakeBinary(dag, memo, h, BuiltinOp::Add, c1, sqJ);
                return MakeBinary(dag, memo, h, BuiltinOp::Div, c1, denom);
            });

        reg.Register(Operon::Hash(BuiltinOp::Sinh),
            [](Nodes& dag, Memo& memo, Hashes& h, std::size_t /*i*/, std::size_t j) -> std::size_t {
                return MakeUnary(dag, memo, h, BuiltinOp::Cosh, j);
            });

        reg.Register(Operon::Hash(BuiltinOp::Cosh),
            [](Nodes& dag, Memo& memo, Hashes& h, std::size_t /*i*/, std::size_t j) -> std::size_t {
                return MakeUnary(dag, memo, h, BuiltinOp::Sinh, j);
            });

        reg.Register(Operon::Hash(BuiltinOp::Tanh),
            [](Nodes& dag, Memo& memo, Hashes& h, std::size_t /*i*/, std::size_t j) -> std::size_t {
                auto c1    = GetConst(dag, memo, h, Scalar{1});
                auto tanhJ = MakeUnary(dag, memo, h, BuiltinOp::Tanh, j); // recomputed fresh from j
                auto sqJ   = MakeUnary(dag, memo, h, BuiltinOp::Square, tanhJ);
                return MakeBinary(dag, memo, h, BuiltinOp::Sub, c1, sqJ);
            });

        // 1/(2*sqrt(j)), not sqrt(j)/(2*j) — the latter is algebraically
        // equal but introduces a spurious 0/0 at j == 0 that this form
        // (a true +inf singularity, matching real calculus) does not.
        reg.Register(Operon::Hash(BuiltinOp::Sqrt),
            [](Nodes& dag, Memo& memo, Hashes& h, std::size_t /*i*/, std::size_t j) -> std::size_t {
                auto c1    = GetConst(dag, memo, h, Scalar{1});
                auto c2    = GetConst(dag, memo, h, Scalar{2});
                auto sqrtJ = MakeUnary(dag, memo, h, BuiltinOp::Sqrt, j); // recomputed fresh from j
                auto denom = MakeBinary(dag, memo, h, BuiltinOp::Mul, c2, sqrtJ);
                return MakeBinary(dag, memo, h, BuiltinOp::Div, c1, denom);
            });

        // 1/(3*cbrt(j)^2), same singularity-avoidance reasoning as Sqrt above.
        reg.Register(Operon::Hash(BuiltinOp::Cbrt),
            [](Nodes& dag, Memo& memo, Hashes& h, std::size_t /*i*/, std::size_t j) -> std::size_t {
                auto c1      = GetConst(dag, memo, h, Scalar{1});
                auto c3      = GetConst(dag, memo, h, Scalar{3});
                auto cbrtJ   = MakeUnary(dag, memo, h, BuiltinOp::Cbrt, j); // recomputed fresh from j
                auto sqCbrtJ = MakeUnary(dag, memo, h, BuiltinOp::Square, cbrtJ);
                auto denom   = MakeBinary(dag, memo, h, BuiltinOp::Mul, c3, sqCbrtJ);
                return MakeBinary(dag, memo, h, BuiltinOp::Div, c1, denom);
            });

        reg.Register(Operon::Hash(BuiltinOp::Square),
            [](Nodes& dag, Memo& memo, Hashes& h, std::size_t /*i*/, std::size_t j) -> std::size_t {
                auto c2 = GetConst(dag, memo, h, Scalar{2});
                return MakeBinary(dag, memo, h, BuiltinOp::Mul, c2, j);
            });

        return true;
    }();
    static_cast<void>(registered);
}

// Forward declaration for mutual recursion.
auto Deriv(Nodes const& orig, Nodes& dag, Memo& memo, Hashes& h,
           std::size_t i, std::size_t targetC) -> std::size_t;

// Collect child indices of orig[i] into a small vector.
auto ChildIndices(Nodes const& orig, std::size_t i) -> Operon::Vector<std::size_t> {
    Operon::Vector<std::size_t> cs;
    cs.reserve(orig[i].Arity);
    for (auto c : Subtree<Node const>{Operon::Span<Node const>{orig}, i}.Indices()) {
        cs.push_back(c);
    }
    return cs;
}

// Compute the symbolic derivative of orig[i] w.r.t. constant at index targetC.
// Returns the dag index of the result expression, or Zero to signal "zero".
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
auto Deriv(Nodes const& orig, Nodes& dag, Memo& memo, Hashes& h,
           std::size_t i, std::size_t targetC) -> std::size_t {
    auto const& n = orig[i];

    // --- leaves ---
    if (n.IsConstant()) {
        return i == targetC ? GetConst(dag, memo, h, Scalar{1}) : Zero;
    }
    if (n.IsVariable()) {
        // d(w * X_i)/dw = X_i when differentiating w.r.t. this node's own weight.
        if (n.Optimize && i == targetC) { return GetVar(dag, memo, h, orig[i], i); }
        return Zero;
    }
    if (n.IsRef()) {
        // Ref is a structural alias; forward the derivative to its target.
        return Deriv(orig, dag, memo, h, orig[i].RefTo, targetC);
    }

    // Every Function node's forward evaluation multiplies its op result by
    // its own weight (Backend::Add/Mul/.../Pow all take a `weight` factor;
    // see functions.hpp), and the numeric reverse-mode sweep
    // (ReverseTraceGeneric) applies this same node's weight exactly once,
    // uniformly across every op type, when propagating into any child. The
    // per-op rules below (hardcoded Add/Mul/Sub/Div/Pow and the unary/binary
    // registries) all compute the *unweighted* local derivative — this
    // wrapper applies the node's own weight once at each branch's return,
    // mirroring that same uniform step instead of leaving it to each rule.
    auto const w = n.Value;
    auto applyWeight = [&](std::size_t x) -> std::size_t {
        // w == 0 short-circuits to Zero rather than emitting Mul(0, x): d(0*f)/dc
        // is exactly 0 regardless of x, but x can still evaluate to Inf at a
        // domain singularity (e.g. Sqrt/Log at j == 0), and Mul(Const(0), Inf)
        // is NaN under IEEE-754, not 0 — this check is load-bearing, not mere
        // defense-in-depth.
        if (x == Zero || w == Scalar{0}) { return Zero; }
        if (w == Scalar{1}) { return x; }
        auto c = GetConst(dag, memo, h, w);
        return MakeBinary(dag, memo, h, BuiltinOp::Mul, c, x);
    };

    auto const children = ChildIndices(orig, i);
    auto const arity    = static_cast<std::size_t>(n.Arity);

    // --- n-ary Add: d(Σ jk)/dc = Σ d(jk)/dc ---
    if (n.IsAddition()) {
        Operon::Vector<std::size_t> terms;
        for (auto c : children) {
            auto dc = Deriv(orig, dag, memo, h, c, targetC);
            if (dc != Zero) { terms.push_back(dc); }
        }
        return applyWeight(AddTerms(dag, memo, h, terms));
    }

    // --- n-ary Mul: product rule ---
    if (n.IsMultiplication()) {
        Operon::Vector<std::size_t> terms;
        for (std::size_t m = 0; m < arity; ++m) {
            auto dm = Deriv(orig, dag, memo, h, children[m], targetC);
            if (dm == Zero) { continue; }
            // product of all other children (references into the original tree)
            std::size_t prod = Zero;
            for (std::size_t l = 0; l < arity; ++l) {
                if (l == m) { continue; }
                prod = (prod == Zero)
                    ? children[l]
                    : MakeBinary(dag, memo, h, BuiltinOp::Mul, prod, children[l]);
            }
            terms.push_back((prod == Zero) ? dm : MakeBinary(dag, memo, h, BuiltinOp::Mul, dm, prod));
        }
        return applyWeight(AddTerms(dag, memo, h, terms));
    }

    // --- Sub ---
    if (n.IsSubtraction()) {
        if (arity == 1) {
            // Unary minus: -a. d(-a)/dc = -da
            auto dj = Deriv(orig, dag, memo, h, children[0], targetC);
            if (dj == Zero) { return Zero; }
            auto neg1 = GetConst(dag, memo, h, Scalar{-1});
            return applyWeight(MakeBinary(dag, memo, h, BuiltinOp::Mul, neg1, dj));
        }
        // arity >= 2: treat as +first - rest (this covers plain binary
        // subtraction too: with only two children, "the rest" is just the
        // second child).
        Operon::Vector<std::size_t> pos;
        Operon::Vector<std::size_t> neg;
        for (std::size_t m = 0; m < arity; ++m) {
            auto dm = Deriv(orig, dag, memo, h, children[m], targetC);
            if (dm == Zero) { continue; }
            if (m == 0) { pos.push_back(dm); } else { neg.push_back(dm); }
        }
        auto p = AddTerms(dag, memo, h, pos);
        auto q = AddTerms(dag, memo, h, neg);
        if (p == Zero && q == Zero) { return Zero; }
        if (q == Zero) { return applyWeight(p); }
        if (p == Zero) {
            auto neg1 = GetConst(dag, memo, h, Scalar{-1});
            return applyWeight(MakeBinary(dag, memo, h, BuiltinOp::Mul, neg1, q));
        }
        return applyWeight(MakeBinary(dag, memo, h, BuiltinOp::Sub, p, q));
    }

    // --- Div ---
    if (n.IsDivision()) {
        if (arity == 1) {
            // 1/j: d(1/j)/dc = -dj / j^2
            auto dj = Deriv(orig, dag, memo, h, children[0], targetC);
            if (dj == Zero) { return Zero; }
            auto j      = children[0];
            auto j2     = MakeBinary(dag, memo, h, BuiltinOp::Mul, j, j);
            auto neg1   = GetConst(dag, memo, h, Scalar{-1});
            auto negDj = MakeBinary(dag, memo, h, BuiltinOp::Mul, neg1, dj);
            return applyWeight(MakeBinary(dag, memo, h, BuiltinOp::Div, negDj, j2));
        }
        if (arity == 2) {
            // a/b: d = (da*b - a*db) / b^2
            auto j  = children[0]; // numerator (nearer)
            auto k  = children[1]; // denominator (farther)
            auto dj = Deriv(orig, dag, memo, h, j, targetC);
            auto dk = Deriv(orig, dag, memo, h, k, targetC);
            if (dj == Zero && dk == Zero) { return Zero; }
            // When only the numerator depends on x, simplify da/b directly.
            // Avoids (da*b)/b^2 which produces 0*Inf=NaN when b=Inf.
            if (dk == Zero) {
                return applyWeight(MakeBinary(dag, memo, h, BuiltinOp::Div, dj, k));
            }
            auto k2 = MakeBinary(dag, memo, h, BuiltinOp::Mul, k, k);
            std::size_t num = Zero;
            if (dj != Zero) {
                num = MakeBinary(dag, memo, h, BuiltinOp::Mul, dj, k);
            }
            {
                auto term = MakeBinary(dag, memo, h, BuiltinOp::Mul, j, dk);
                if (num == Zero) {
                    auto neg1 = GetConst(dag, memo, h, Scalar{-1});
                    num = MakeBinary(dag, memo, h, BuiltinOp::Mul, neg1, term);
                } else {
                    num = MakeBinary(dag, memo, h, BuiltinOp::Sub, num, term);
                }
            }
            return applyWeight(MakeBinary(dag, memo, h, BuiltinOp::Div, num, k2));
        }
        return Zero; // arity > 2 not yet supported
    }

    // --- Pow(j, k) = j^k ---
    if (n.IsPow()) {
        auto j  = children[0]; // base (nearer)
        auto k  = children[1]; // exponent (farther)
        auto dj = Deriv(orig, dag, memo, h, j, targetC);
        auto dk = Deriv(orig, dag, memo, h, k, targetC);
        if (dj == Zero && dk == Zero) { return Zero; }
        // Fresh, unweighted j^k, recomputed from children.
        auto powJK = MakeBinary(dag, memo, h, BuiltinOp::Pow, j, k);
        Operon::Vector<std::size_t> terms;
        if (dj != Zero) {
            // d/dj: k * j^k / j = k * result / j
            auto ki   = MakeBinary(dag, memo, h, BuiltinOp::Mul, k, powJK); // k * result
            auto term = MakeBinary(dag, memo, h, BuiltinOp::Div, ki, j);
            terms.push_back(MakeBinary(dag, memo, h, BuiltinOp::Mul, dj, term));
        }
        if (dk != Zero) {
            // d/dk: j^k * ln(j) = result * log(j)
            auto logJ = MakeUnary(dag, memo, h, BuiltinOp::Log, j);
            auto t     = MakeBinary(dag, memo, h, BuiltinOp::Mul, powJK, logJ);
            terms.push_back(MakeBinary(dag, memo, h, BuiltinOp::Mul, dk, t));
        }
        return applyWeight(AddTerms(dag, memo, h, terms));
    }

    // --- Aq, Powabs, Fmin, Fmax: not yet differentiated ---
    if (n.IsAq() || n.IsPowabs() || n.IsOp<BuiltinOp::Fmin, BuiltinOp::Fmax>()) {
        return Zero;
    }

    // --- arity-2 user-defined/composed functions ---
    // Registry lookup for any binary Function node not already handled
    // above (Add/Mul/Sub/Div/Pow are hardcoded; Aq/Powabs/Fmin/Fmax are
    // explicitly excluded) — same chain-rule-sum pattern Pow's hardcoded
    // case already uses. A miss degrades to Zero, matching the unary
    // registry's own miss convention.
    if (arity == 2) {
        auto const* rule = BinaryDerivRules().TryGet(n.HashValue);
        if (rule != nullptr) {
            auto j  = children[0]; // nearer
            auto k  = children[1]; // farther
            auto dj = Deriv(orig, dag, memo, h, j, targetC);
            auto dk = Deriv(orig, dag, memo, h, k, targetC);
            if (dj == Zero && dk == Zero) { return Zero; }
            auto [fpj, fpk] = (*rule)(dag, memo, h, i, j, k);
            Operon::Vector<std::size_t> terms;
            if (dj != Zero && fpj != Zero) {
                terms.push_back(MakeBinary(dag, memo, h, BuiltinOp::Mul, fpj, dj));
            }
            if (dk != Zero && fpk != Zero) {
                terms.push_back(MakeBinary(dag, memo, h, BuiltinOp::Mul, fpk, dk));
            }
            return applyWeight(AddTerms(dag, memo, h, terms));
        }
    }

    // --- unary nodes ---
    if (arity == 1) {
        auto j  = children[0];
        auto dj = Deriv(orig, dag, memo, h, j, targetC);
        if (dj == Zero) { return Zero; }

        // Look up this operation's derivative rule (see
        // RegisterBuiltinSymbolicDerivs above for the 16 built-in rules). A
        // miss — including Abs/Sqrtabs/Floor/Ceil, which are deliberately
        // left unregistered — yields Zero.
        auto const* rule = SymbolicDerivRules().TryGet(n.HashValue);
        if (rule == nullptr) { return Zero; }
        auto fp = (*rule)(dag, memo, h, i, j); // symbolic f'(j)

        if (fp == Zero) { return Zero; }
        return applyWeight(MakeBinary(dag, memo, h, BuiltinOp::Mul, fp, dj));
    }

    return Zero;
}

// Shared first-pass logic: copy original nodes into `dag`, build identity
// hashes, collect optimizable constants, and differentiate the tree root
// w.r.t. each constant. Returns the constants vector (needed by the Hessian
// second pass). The caller owns `dag`, `memo`, and `h` so that BuildHessianDag
// can continue appending to them.
auto DifferentiateFirstOrder(
    Tree const& tree, Nodes& dag, Memo& memo, Hashes& h,
    Operon::Vector<std::size_t>& roots, std::size_t reserveFactor
) -> Operon::Vector<std::size_t>
{
    RegisterBuiltinSymbolicDerivs();

    auto const& orig = tree.Nodes();
    auto const n     = orig.size();

    dag = orig;
    dag.reserve(n * reserveFactor);

    memo.clear();
    h.clear();
    h.reserve(n * reserveFactor);
    for (std::size_t i = 0; i < n; ++i) {
        h.push_back(static_cast<uint64_t>(i));
    }

    Operon::Vector<std::size_t> constants;
    for (std::size_t i = 0; i < n; ++i) {
        if (orig[i].Optimize) { constants.push_back(i); }
    }

    roots.resize(constants.size(), Zero);
    for (std::size_t ci = 0; ci < constants.size(); ++ci) {
        roots[ci] = Deriv(orig, dag, memo, h, n - 1, constants[ci]);
    }

    return constants;
}

} // anonymous namespace

void RegisterUnarySymbolicDeriv(Operon::Hash hash, UnarySymbolicDerivRule rule)
{
    RegisterBuiltinSymbolicDerivs();
    SymbolicDerivRules().Register(hash, std::move(rule));
}

auto HasUnarySymbolicDeriv(Operon::Hash hash) -> bool
{
    RegisterBuiltinSymbolicDerivs();
    return SymbolicDerivRules().Contains(hash);
}

auto GetUnarySymbolicDeriv(Operon::Hash hash) -> UnarySymbolicDerivRule const*
{
    RegisterBuiltinSymbolicDerivs();
    return SymbolicDerivRules().TryGet(hash);
}

void RegisterBinarySymbolicDeriv(Operon::Hash hash, BinarySymbolicDerivRule rule)
{
    // Unlike RegisterUnarySymbolicDeriv (which forces built-in
    // registration first so a colliding write throws immediately), the
    // binary registry has no built-in entries to collide with — every
    // binary built-in is hardcoded directly in Deriv() (Add/Mul/Sub/Div/
    // Pow) or on its explicit "not yet differentiated" exclusion list
    // (Aq/Powabs/Fmin/Fmax), and Deriv() checks those by hash *before*
    // ever consulting this registry. A hash matching one of them would
    // still silently *accept* the write here, but the rule would then
    // never fire — Deriv() never reaches the registry consult for that
    // hash. Reject it explicitly instead of letting it silently no-op.
    static constexpr std::array<Operon::Hash, 9> handledElsewhere {
        Operon::Hash(BuiltinOp::Add), Operon::Hash(BuiltinOp::Mul),
        Operon::Hash(BuiltinOp::Sub), Operon::Hash(BuiltinOp::Div),
        Operon::Hash(BuiltinOp::Pow), Operon::Hash(BuiltinOp::Aq),
        Operon::Hash(BuiltinOp::Powabs), Operon::Hash(BuiltinOp::Fmin),
        Operon::Hash(BuiltinOp::Fmax),
    };
    if (std::ranges::find(handledElsewhere, hash) != handledElsewhere.end()) {
        throw std::invalid_argument(
            "RegisterBinarySymbolicDeriv: hash matches a built-in op Deriv() already handles "
            "directly (Add/Mul/Sub/Div/Pow/Aq/Powabs/Fmin/Fmax) — a rule registered here would "
            "never be consulted");
    }
    BinaryDerivRules().Register(hash, std::move(rule));
}

auto HasBinarySymbolicDeriv(Operon::Hash hash) -> bool
{
    return BinaryDerivRules().Contains(hash);
}

auto GetBinarySymbolicDeriv(Operon::Hash hash) -> BinarySymbolicDerivRule const*
{
    return BinaryDerivRules().TryGet(hash);
}

auto GetSymbolicDerivConst(Nodes& dag, Memo& memo, Hashes& h, Scalar val) -> std::size_t
{
    return GetConst(dag, memo, h, val);
}

auto MakeSymbolicDerivUnary(Nodes& dag, Memo& memo, Hashes& h, BuiltinOp op, std::size_t a) -> std::size_t
{
    return MakeUnary(dag, memo, h, op, a);
}

auto MakeSymbolicDerivBinary(Nodes& dag, Memo& memo, Hashes& h, BuiltinOp op, std::size_t a, std::size_t b) -> std::size_t
{
    return MakeBinary(dag, memo, h, op, a, b);
}

auto BuildJacobianDag(Tree const& tree) -> JacobianDag {
    JacobianDag dag;
    Memo memo;
    Hashes h;
    DifferentiateFirstOrder(tree, dag.Nodes, memo, h, dag.Roots, 8);
    dag.OriginalSize = tree.Nodes().size();
    return dag;
}

auto BuildHessianDag(Tree const& tree) -> HessianDag {
    HessianDag result;
    Memo memo;
    Hashes h;
    auto const constants = DifferentiateFirstOrder(
        tree, result.Nodes, memo, h, result.JacobianRoots, 32);

    result.OriginalSize = tree.Nodes().size();
    auto const p = constants.size();
    result.NumParams = p;

    // Snapshot: the second pass calls Deriv(snapshot, result.Nodes, ...) where
    // snapshot is the source and result.Nodes is the destination. A copy is
    // needed because result.Nodes grows during the second pass, which would
    // invalidate iterators/references if used as both source and destination.
    auto const snapshot = result.Nodes;

    result.HessianRoots.resize(p * (p + 1) / 2, Zero);
    for (std::size_t i = 0; i < p; ++i) {
        if (result.JacobianRoots[i] == Zero) { continue; }
        for (std::size_t j = i; j < p; ++j) {
            result.HessianRoots[result.UpperIdx(i, j)] =
                Deriv(snapshot, result.Nodes, memo, h,
                      result.JacobianRoots[i], constants[j]);
        }
    }

    return result;
}

} // namespace Operon
