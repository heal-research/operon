// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include "operon/core/tree_diff.hpp"

#include <bit>
#include <limits>

#include "operon/core/contracts.hpp"
#include "operon/core/subtree.hpp"

namespace Operon {

namespace {

using Nodes  = Operon::Vector<Node>;
using Hashes = Operon::Vector<Operon::Hash>;
using Memo   = Operon::Map<Operon::Hash, std::size_t>;

constexpr std::size_t Zero = std::numeric_limits<std::size_t>::max();

// Mix two 64-bit hashes (Boost-style but with a larger multiplier).
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
               NodeType op, std::size_t a) -> std::size_t {
    auto opHash = Mix(static_cast<uint64_t>(op), h[a]);
    if (auto it = memo.find(opHash); it != memo.end()) { return it->second; }
    AppendRef(dag, h, a);
    Node n{op};
    n.Length = 1; // one Ref child with Length=0
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
                NodeType op, std::size_t a, std::size_t b) -> std::size_t {
    auto opHash = Mix(Mix(static_cast<uint64_t>(op), h[a]), h[b]);
    if (auto it = memo.find(opHash); it != memo.end()) { return it->second; }
    AppendRef(dag, h, b); // farther child first
    AppendRef(dag, h, a); // nearer child second
    Node n{op};
    n.Arity  = 2;
    n.Length = 2; // two Ref children, each Length=0
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
        result = MakeBinary(dag, memo, h, NodeType::Add, result, terms[k]);
    }
    return result;
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

    auto const children = ChildIndices(orig, i);
    auto const arity    = static_cast<std::size_t>(n.Arity);

    // --- n-ary Add: d(Σ jk)/dc = Σ d(jk)/dc ---
    if (n.IsAddition()) {
        Operon::Vector<std::size_t> terms;
        for (auto c : children) {
            auto dc = Deriv(orig, dag, memo, h, c, targetC);
            if (dc != Zero) { terms.push_back(dc); }
        }
        return AddTerms(dag, memo, h, terms);
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
                    : MakeBinary(dag, memo, h, NodeType::Mul, prod, children[l]);
            }
            terms.push_back((prod == Zero) ? dm : MakeBinary(dag, memo, h, NodeType::Mul, dm, prod));
        }
        return AddTerms(dag, memo, h, terms);
    }

    // --- Sub ---
    if (n.IsSubtraction()) {
        auto dj = Deriv(orig, dag, memo, h, children[0], targetC);
        if (arity == 1) {
            // Unary minus: -a. d(-a)/dc = -da
            if (dj == Zero) { return Zero; }
            auto neg1 = GetConst(dag, memo, h, Scalar{-1});
            return MakeBinary(dag, memo, h, NodeType::Mul, neg1, dj);
        }
        if (arity == 2) {
            auto dk = Deriv(orig, dag, memo, h, children[1], targetC);
            if (dj == Zero && dk == Zero) { return Zero; }
            if (dk == Zero) { return dj; }
            if (dj == Zero) {
                auto neg1 = GetConst(dag, memo, h, Scalar{-1});
                return MakeBinary(dag, memo, h, NodeType::Mul, neg1, dk);
            }
            return MakeBinary(dag, memo, h, NodeType::Sub, dj, dk);
        }
        // arity > 2: treat as +first - rest
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
        if (q == Zero) { return p; }
        if (p == Zero) {
            auto neg1 = GetConst(dag, memo, h, Scalar{-1});
            return MakeBinary(dag, memo, h, NodeType::Mul, neg1, q);
        }
        return MakeBinary(dag, memo, h, NodeType::Sub, p, q);
    }

    // --- Div ---
    if (n.IsDivision()) {
        if (arity == 1) {
            // 1/j: d(1/j)/dc = -dj / j^2
            auto dj = Deriv(orig, dag, memo, h, children[0], targetC);
            if (dj == Zero) { return Zero; }
            auto j      = children[0];
            auto j2     = MakeBinary(dag, memo, h, NodeType::Mul, j, j);
            auto neg1   = GetConst(dag, memo, h, Scalar{-1});
            auto negDj = MakeBinary(dag, memo, h, NodeType::Mul, neg1, dj);
            return MakeBinary(dag, memo, h, NodeType::Div, negDj, j2);
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
                return MakeBinary(dag, memo, h, NodeType::Div, dj, k);
            }
            auto k2 = MakeBinary(dag, memo, h, NodeType::Mul, k, k);
            std::size_t num = Zero;
            if (dj != Zero) {
                num = MakeBinary(dag, memo, h, NodeType::Mul, dj, k);
            }
            {
                auto term = MakeBinary(dag, memo, h, NodeType::Mul, j, dk);
                if (num == Zero) {
                    auto neg1 = GetConst(dag, memo, h, Scalar{-1});
                    num = MakeBinary(dag, memo, h, NodeType::Mul, neg1, term);
                } else {
                    num = MakeBinary(dag, memo, h, NodeType::Sub, num, term);
                }
            }
            return MakeBinary(dag, memo, h, NodeType::Div, num, k2);
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
        Operon::Vector<std::size_t> terms;
        if (dj != Zero) {
            // d/dj: k * j^k / j = k * result / j
            auto ki   = MakeBinary(dag, memo, h, NodeType::Mul, k, i); // k * result
            auto term = MakeBinary(dag, memo, h, NodeType::Div, ki, j);
            terms.push_back(MakeBinary(dag, memo, h, NodeType::Mul, dj, term));
        }
        if (dk != Zero) {
            // d/dk: j^k * ln(j) = result * log(j)
            auto logJ = MakeUnary(dag, memo, h, NodeType::Log, j);
            auto t     = MakeBinary(dag, memo, h, NodeType::Mul, i, logJ);
            terms.push_back(MakeBinary(dag, memo, h, NodeType::Mul, dk, t));
        }
        return AddTerms(dag, memo, h, terms);
    }

    // --- Aq, Powabs, Fmin, Fmax: not yet differentiated ---
    if (n.IsAq() || n.IsPowabs() || n.Is<NodeType::Fmin, NodeType::Fmax>()) {
        return Zero;
    }

    // --- unary nodes ---
    if (arity == 1) {
        auto j  = children[0];
        auto dj = Deriv(orig, dag, memo, h, j, targetC);
        if (dj == Zero) { return Zero; }

        std::size_t fp = Zero; // symbolic f'(j)

        switch (n.Type) {
        case NodeType::Exp:
            // d exp(j)/dj = exp(j) = result
            fp = i;
            break;

        case NodeType::Log:
        case NodeType::Logabs: {
            // d log(j)/dj = 1/j
            auto c1 = GetConst(dag, memo, h, Scalar{1});
            fp = MakeBinary(dag, memo, h, NodeType::Div, c1, j);
            break;
        }

        case NodeType::Log1p: {
            // d log(1+j)/dj = 1/(1+j)
            auto c1 = GetConst(dag, memo, h, Scalar{1});
            auto onePlusJ = MakeBinary(dag, memo, h, NodeType::Add, c1, j);
            fp = MakeBinary(dag, memo, h, NodeType::Div, c1, onePlusJ);
            break;
        }

        case NodeType::Sin:
            // d sin(j)/dj = cos(j)
            fp = MakeUnary(dag, memo, h, NodeType::Cos, j);
            break;

        case NodeType::Cos: {
            // d cos(j)/dj = -sin(j)
            auto sinJ = MakeUnary(dag, memo, h, NodeType::Sin, j);
            auto neg1  = GetConst(dag, memo, h, Scalar{-1});
            fp = MakeBinary(dag, memo, h, NodeType::Mul, neg1, sinJ);
            break;
        }

        case NodeType::Tan: {
            // d tan(j)/dj = 1 + tan^2(j) = 1 + result^2
            auto c1   = GetConst(dag, memo, h, Scalar{1});
            auto sqI = MakeUnary(dag, memo, h, NodeType::Square, i);
            fp = MakeBinary(dag, memo, h, NodeType::Add, c1, sqI);
            break;
        }

        case NodeType::Acos: {
            // d acos(j)/dj = -1/sqrt(1 - j^2)
            auto c1     = GetConst(dag, memo, h, Scalar{1});
            auto sqJ   = MakeUnary(dag, memo, h, NodeType::Square, j);
            auto denom  = MakeBinary(dag, memo, h, NodeType::Sub, c1, sqJ);
            auto sqD   = MakeUnary(dag, memo, h, NodeType::Sqrt, denom);
            auto recip  = MakeBinary(dag, memo, h, NodeType::Div, c1, sqD);
            auto neg1   = GetConst(dag, memo, h, Scalar{-1});
            fp = MakeBinary(dag, memo, h, NodeType::Mul, neg1, recip);
            break;
        }

        case NodeType::Asin: {
            // d asin(j)/dj = 1/sqrt(1 - j^2)
            auto c1    = GetConst(dag, memo, h, Scalar{1});
            auto sqJ  = MakeUnary(dag, memo, h, NodeType::Square, j);
            auto denom = MakeBinary(dag, memo, h, NodeType::Sub, c1, sqJ);
            auto sqD  = MakeUnary(dag, memo, h, NodeType::Sqrt, denom);
            fp = MakeBinary(dag, memo, h, NodeType::Div, c1, sqD);
            break;
        }

        case NodeType::Atan: {
            // d atan(j)/dj = 1/(1 + j^2)
            auto c1    = GetConst(dag, memo, h, Scalar{1});
            auto sqJ  = MakeUnary(dag, memo, h, NodeType::Square, j);
            auto denom = MakeBinary(dag, memo, h, NodeType::Add, c1, sqJ);
            fp = MakeBinary(dag, memo, h, NodeType::Div, c1, denom);
            break;
        }

        case NodeType::Sinh:
            // d sinh(j)/dj = cosh(j)
            fp = MakeUnary(dag, memo, h, NodeType::Cosh, j);
            break;

        case NodeType::Cosh:
            // d cosh(j)/dj = sinh(j)
            fp = MakeUnary(dag, memo, h, NodeType::Sinh, j);
            break;

        case NodeType::Tanh: {
            // d tanh(j)/dj = 1 - tanh^2(j) = 1 - result^2
            auto c1   = GetConst(dag, memo, h, Scalar{1});
            auto sqI = MakeUnary(dag, memo, h, NodeType::Square, i);
            fp = MakeBinary(dag, memo, h, NodeType::Sub, c1, sqI);
            break;
        }

        case NodeType::Sqrt: {
            // d sqrt(j)/dj = result / (2 * j)
            auto c2   = GetConst(dag, memo, h, Scalar{2});
            auto twoJ = MakeBinary(dag, memo, h, NodeType::Mul, c2, j);
            fp = MakeBinary(dag, memo, h, NodeType::Div, i, twoJ);
            break;
        }

        case NodeType::Cbrt: {
            // d cbrt(j)/dj = result / (3 * j)
            auto c3    = GetConst(dag, memo, h, Scalar{3});
            auto threeJ = MakeBinary(dag, memo, h, NodeType::Mul, c3, j);
            fp = MakeBinary(dag, memo, h, NodeType::Div, i, threeJ);
            break;
        }

        case NodeType::Square: {
            // d j^2/dj = 2 * j
            auto c2 = GetConst(dag, memo, h, Scalar{2});
            fp = MakeBinary(dag, memo, h, NodeType::Mul, c2, j);
            break;
        }

        // Non-smooth or not-yet-implemented: return zero gradient.
        case NodeType::Abs:
        case NodeType::Sqrtabs:
        case NodeType::Floor:
        case NodeType::Ceil:
        default:
            return Zero;
        }

        if (fp == Zero) { return Zero; }
        return MakeBinary(dag, memo, h, NodeType::Mul, fp, dj);
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
    auto const& orig = tree.Nodes();
    auto const n     = orig.size();

    dag = orig;
    dag.reserve(n * reserveFactor);

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

    // Snapshot the DAG so the second pass can navigate first-order derivative
    // nodes without aliasing the growing result.Nodes vector.
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
