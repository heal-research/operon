// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_ALGORITHMS_ENUMERATION_CANONICALIZER_HPP
#define OPERON_ALGORITHMS_ENUMERATION_CANONICALIZER_HPP

#include <string>

#include "operon/core/tree.hpp"
#include "operon/operon_export.hpp"

namespace Operon {

// One tree's canonical algebraic identity, used by the enumerator to group
// pre-canonical Expression-bucket candidates that represent the same
// parametric family into one canonical class (see
// GrammarEnumerationAlgorithm::Run) - so only one representative per class
// needs its coefficients fit.
struct CanonicalExpression {
    // Deterministic string identity: two trees produce an equal Key iff this
    // canonicalizer could prove them algebraically equivalent up to
    // commutative reordering, Sub/Div normalization, bounded distribution of
    // multiplication over addition, and treating every optimizable
    // (Optimize == true) Constant as an interchangeable free parameter
    // (their fitted *values* never distinguish two structural families - see
    // CanonicalizeEnumerationTree's doc comment below). Never treats two
    // trees as equal on a false positive; may fail to prove equivalence
    // (distinct Key) for something a fuller symbolic engine would recognize
    // - see the bounded-expansion note below.
    std::string Key;
    // A Reduce()+Simplify()'d, structurally-sorted (Tree::Sort(), by
    // content) copy of the input tree - a deterministic normal form paired
    // with Key, useful for stable secondary tie-breaks (e.g. lexicographic
    // comparison) and debugging. Not itself further algebraically rewritten
    // (no Sub/Div normalization or expansion) - only Key carries that.
    Tree Representative;
};

// Computes `tree`'s canonical algebraic identity (see CanonicalExpression).
//
// Two structurally different trees canonicalize to the same Key when they
// represent the same free-parameter family up to:
//   - Sub(a,b,...) normalized to additive negation, Div(a,b) and unary Div
//     (1/x) normalized to multiplicative inverse.
//   - Flattening and sorting Add/Mul children (commutativity).
//   - Bounded distribution of multiplication over addition into a sorted
//     sum-of-monomials form (each monomial a sorted product of opaque
//     bases raised to an integer power, plus a numeric coefficient).
//     Square(x), Pow(x, <small fixed integer>), and x*x*...*x all reduce to
//     the same base^n monomial factor.
//   - Folding every Optimize==true Constant within one monomial into a
//     single "free" marker: this DP engine never lets two positions share
//     one Constant node (no Ref-based sharing), so every optimizable
//     Constant it produces is an independent free parameter, and two
//     monomials differing only in how many/which independent free
//     parameters they carry (e.g. `K*x` vs `K1*K2*x`, or `K*x` vs `-K*x` -
//     negating a free parameter is just refitting it to the opposite sign)
//     are the same family. A *fixed* (Optimize == false) Constant's value
//     (e.g. Cube's exponent 3, TenExp's base 10, Log10Abs's 1/ln(10) scale)
//     is never anonymized this way - it's part of the structural identity.
//   - Transcendental applications (Exp/Log/Logabs/Sin/Sqrt/Sqrtabs/Cbrt/Aq)
//     and Pow whose exponent isn't a small fixed integer are treated as
//     opaque functions: their own arguments are still recursively
//     canonicalized (so an equivalent subexpression nested inside one is
//     still recognized), but no identity is applied to the function itself
//     (e.g. log(a*b) is never rewritten to log(a)+log(b)).
//
// Bounded expansion: multiplying two sums whose monomial-count product
// would exceed a fixed cap does not expand - the whole product is instead
// folded into one opaque monomial keyed by its (still sorted, still
// commutative-safe) factor multiset. This can miss a duplicate a full
// polynomial expansion would have found, but never incorrectly merges two
// inequivalent expressions - correctness (never a false positive) is
// prioritized over completeness (recall), per the algorithm's contract.
[[nodiscard]] OPERON_EXPORT auto CanonicalizeEnumerationTree(Operon::Tree const& tree) -> CanonicalExpression;

} // namespace Operon

#endif
