// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/algorithms/enumeration_canonicalizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "operon/core/node.hpp"

namespace Operon {

namespace {
    // Cap on the number of monomials a single Mul's full distribution may
    // produce before falling back to an opaque (unexpanded but still
    // commutative-sorted) product - see the header's bounded-expansion note.
    constexpr std::size_t ExpansionCap = 256;

    // One term of a canonicalized sum-of-monomials form: Coeff * K^HasFree *
    // prod(base^exp for (base,exp) in Factors). `Factors` maps an opaque
    // base key (a variable, or a nested opaque-function/product key) to its
    // integer exponent; HasFree marks that this monomial also carries at
    // least one independent free (Optimize==true) parameter - see the
    // header's doc comment on why that's sound to anonymize down to a bare
    // flag instead of tracking count/identity.
    struct Monomial {
        std::map<std::string, int> Factors;
        double Coeff{1.0};
        bool HasFree{false};
    };
    using Sum = std::vector<Monomial>;

    auto FormatNumber(double v) -> std::string
    {
        std::array<char, 64> buf{};
        auto n = std::snprintf(buf.data(), buf.size(), "%.17g", v);
        return std::string(buf.data(), buf.data() + std::max(n, 0));
    }

    auto SerializeFactors(std::map<std::string, int> const& factors) -> std::string
    {
        std::string s;
        bool first = true;
        for (auto const& [base, exp] : factors) { // std::map is already key-sorted
            if (!first) { s += "*"; }
            first = false;
            s += base;
            if (exp != 1) { s += "^" + std::to_string(exp); }
        }
        return s;
    }

    auto SerializeMonomial(Monomial const& m) -> std::string
    {
        std::string s = m.HasFree ? std::string("K") : FormatNumber(m.Coeff);
        auto factors = SerializeFactors(m.Factors);
        if (!factors.empty()) { s += "*" + factors; }
        return s;
    }

    // Deterministic order over monomials for both sorting the final Sum and
    // grouping equal-Factors monomials for merging - keyed on the
    // (HasFree, Factors) identity, not on Coeff (two monomials with the same
    // base structure but different fixed coefficients must be adjacent so
    // MergeSum can fold them, and must compare by identity before value).
    auto MonomialIdentity(Monomial const& m) -> std::string
    {
        return (m.HasFree ? std::string("1|") : std::string("0|")) + SerializeFactors(m.Factors);
    }

    // Groups monomials sharing the same (HasFree, Factors) identity, summing
    // Coeff for fixed-only groups (dropping an exact-zero result - the
    // "x + (-x) = 0" fold, extended to post-expansion cancellation) and
    // collapsing a free group to one representative (HasFree monomials are
    // refit as a unit regardless of how many independent parameters
    // contributed to the group - see the header's doc comment). Sorted by
    // MonomialIdentity for a deterministic final Sum.
    auto MergeSum(Sum sum) -> Sum
    {
        std::ranges::sort(sum, {}, MonomialIdentity);
        Sum out;
        for (std::size_t i = 0; i < sum.size();) {
            std::size_t j = i + 1;
            auto const identity = MonomialIdentity(sum[i]);
            bool hasFree = sum[i].HasFree;
            double coeff = sum[i].Coeff;
            while (j < sum.size() && MonomialIdentity(sum[j]) == identity) {
                hasFree = hasFree || sum[j].HasFree;
                coeff += sum[j].Coeff;
                ++j;
            }
            if (!hasFree && std::abs(coeff) < 1e-300) {
                // exact-zero cancellation for the constant-only monomial (empty Factors, Coeff -> 0)
                // and for any other fixed-only monomial family that summed to zero.
                i = j;
                continue;
            }
            out.push_back(Monomial{ .Factors = sum[i].Factors, .Coeff = hasFree ? 1.0 : coeff, .HasFree = hasFree });
            i = j;
        }
        std::ranges::sort(out, {}, MonomialIdentity);
        return out;
    }

    auto NegateSum(Sum sum) -> Sum
    {
        // Negating a HasFree monomial is a no-op: its free parameter will be
        // refit to the opposite sign, representing the exact same family -
        // see the header's doc comment. Only fixed monomials actually flip.
        for (auto& m : sum) {
            if (!m.HasFree) { m.Coeff = -m.Coeff; }
        }
        return sum;
    }

    // Opaque fallback for a Mul whose full distribution would exceed
    // ExpansionCap: one monomial keyed by the sorted multiset of the
    // factors' own (still fully recursively canonicalized) serialized sums
    // - still commutative-safe (reordering the original Mul's operands
    // yields the same key) without paying distribution's combinatorial
    // cost, and never incorrectly merges two different structures.
    auto OpaqueProduct(std::vector<Sum> const& factors) -> Sum
    {
        std::vector<std::string> keys;
        keys.reserve(factors.size());
        bool hasFree = false;
        for (auto const& f : factors) {
            for (auto const& m : f) { hasFree = hasFree || m.HasFree; }
            std::vector<std::string> ms;
            ms.reserve(f.size());
            for (auto const& m : f) { ms.push_back(SerializeMonomial(m)); }
            std::ranges::sort(ms);
            std::string s = "(";
            for (std::size_t i = 0; i < ms.size(); ++i) { if (i) { s += "+"; } s += ms[i]; }
            s += ")";
            keys.push_back(std::move(s));
        }
        std::ranges::sort(keys);
        std::map<std::string, int> factorMap;
        for (auto& k : keys) { ++factorMap[k]; }
        return Sum{ Monomial{ .Factors = std::move(factorMap), .Coeff = 1.0, .HasFree = hasFree } };
    }

    // Multiplies two sums via full distribution, merging the result. Falls
    // back to OpaqueProduct({a, b}) when the pre-merge monomial count would
    // exceed ExpansionCap.
    auto MultiplySum(Sum const& a, Sum const& b) -> Sum
    {
        if (a.size() * b.size() > ExpansionCap) { return OpaqueProduct({ a, b }); }
        Sum out;
        out.reserve(a.size() * b.size());
        for (auto const& ma : a) {
            for (auto const& mb : b) {
                std::map<std::string, int> factors = ma.Factors;
                for (auto const& [base, exp] : mb.Factors) {
                    auto [it, inserted] = factors.try_emplace(base, exp);
                    if (!inserted) {
                        it->second += exp;
                        if (it->second == 0) { factors.erase(it); }
                    }
                }
                out.push_back(Monomial{
                    .Factors = std::move(factors),
                    .Coeff = ma.Coeff * mb.Coeff,
                    .HasFree = ma.HasFree || mb.HasFree,
                });
            }
        }
        return MergeSum(std::move(out));
    }

    auto InvertSum(Sum const& s) -> std::optional<Sum>
    {
        // Only a pure monomial (no addition) is invertible within this
        // multiplicative system - a genuine sum denominator (e.g. y+1) has
        // no closed monomial form for 1/(y+1), so the caller falls back to
        // an opaque Div/Inv representation instead.
        if (s.size() != 1) { return std::nullopt; }
        auto const& m = s.front();
        if (m.HasFree) { return std::nullopt; } // 1/K has no stable canonical form either
        if (m.Coeff == 0.0) { return std::nullopt; }
        std::map<std::string, int> inverted;
        for (auto const& [base, exp] : m.Factors) { inverted[base] = -exp; }
        return Sum{ Monomial{ .Factors = std::move(inverted), .Coeff = 1.0 / m.Coeff, .HasFree = false } };
    }

    // Extracts a small fixed non-negative integer exponent from an
    // already-canonicalized operand Sum, iff it is exactly one fixed
    // (non-free) constant-only monomial with an integral value in
    // [0, MaxIntegerExponent] - i.e. the operand was a plain
    // Node::Constant(n, Optimize=false) leaf (see ProductionOperand::Fixed,
    // used by Cube's exponent operand). Returns nullopt otherwise (e.g. a
    // general SimpleExpr exponent operand, or a non-integer/out-of-range
    // fixed value), in which case Pow is treated opaquely.
    constexpr int MaxIntegerExponent = 8;
    auto AsSmallIntegerExponent(Sum const& s) -> std::optional<int>
    {
        if (s.size() != 1) { return std::nullopt; }
        auto const& m = s.front();
        if (m.HasFree || !m.Factors.empty()) { return std::nullopt; }
        auto rounded = std::llround(m.Coeff);
        if (std::abs(m.Coeff - static_cast<double>(rounded)) > 1e-9) { return std::nullopt; }
        if (rounded < 0 || rounded > MaxIntegerExponent) { return std::nullopt; }
        return static_cast<int>(rounded);
    }

    auto OpaqueLeaf(std::string key, bool hasFree = false) -> Sum
    {
        return Sum{ Monomial{ .Factors = { { std::move(key), 1 } }, .Coeff = 1.0, .HasFree = hasFree } };
    }

    auto SerializeSum(Sum const& s) -> std::string
    {
        std::vector<std::string> parts;
        parts.reserve(s.size());
        for (auto const& m : s) { parts.push_back(SerializeMonomial(m)); }
        std::ranges::sort(parts); // already MergeSum-sorted by identity, but re-sort lexicographically for the key text itself
        std::string out;
        for (std::size_t i = 0; i < parts.size(); ++i) { if (i) { out += "+"; } out += parts[i]; }
        return out.empty() ? std::string("0") : out;
    }

    class Canonicalizer {
    public:
        explicit Canonicalizer(Operon::Tree const& tree) : nodes_(tree.Nodes()) { }

        auto Canonicalize(std::size_t i) -> Sum
        {
            auto const& n = nodes_[i];

            if (n.IsConstant()) {
                if (n.Optimize) { return Sum{ Monomial{ .Factors = {}, .Coeff = 1.0, .HasFree = true } }; }
                return Sum{ Monomial{ .Factors = {}, .Coeff = static_cast<double>(n.Value), .HasFree = false } };
            }
            if (n.IsVariable()) {
                return OpaqueLeaf("v" + std::to_string(n.HashValue));
            }
            if (n.IsRef()) {
                // Not produced by the enumeration engine (no Ref-based sharing in its output) - treated
                // conservatively/opaquely rather than dereferenced, since RefTo's target may not itself have been
                // canonicalized in this traversal order.
                return OpaqueLeaf("ref" + std::to_string(n.RefTo));
            }

            auto children = Indices(i);
            std::vector<Sum> childSums;
            childSums.reserve(children.size());
            for (auto j : children) { childSums.push_back(Canonicalize(j)); }

            if (n.IsOp<BuiltinOp::Add>()) {
                Sum out;
                for (auto& c : childSums) { out.insert(out.end(), c.begin(), c.end()); }
                return MergeSum(std::move(out));
            }
            if (n.IsOp<BuiltinOp::Sub>()) {
                Sum out = childSums.front();
                for (std::size_t k = 1; k < childSums.size(); ++k) {
                    auto neg = NegateSum(std::move(childSums[k]));
                    out.insert(out.end(), neg.begin(), neg.end());
                }
                if (childSums.size() == 1) { out = NegateSum(std::move(out)); } // unary Sub = negation
                return MergeSum(std::move(out));
            }
            if (n.IsOp<BuiltinOp::Mul>()) {
                Sum acc = childSums.front();
                for (std::size_t k = 1; k < childSums.size(); ++k) { acc = MultiplySum(acc, childSums[k]); }
                return acc;
            }
            if (n.IsOp<BuiltinOp::Div>()) {
                if (childSums.size() == 1) { // unary Div = inversion (1/x)
                    if (auto inv = InvertSum(childSums.front())) { return *inv; }
                    return OpaqueFunction("inv", childSums);
                }
                Sum acc = childSums.front();
                for (std::size_t k = 1; k < childSums.size(); ++k) {
                    if (auto inv = InvertSum(childSums[k])) { acc = MultiplySum(acc, *inv); }
                    else { return OpaqueFunction("div", childSums); }
                }
                return acc;
            }
            if (n.IsOp<BuiltinOp::Square>()) {
                return MultiplySum(childSums.front(), childSums.front());
            }
            if (n.IsOp<BuiltinOp::Pow>()) {
                if (auto exp = AsSmallIntegerExponent(childSums[1])) {
                    Sum acc{ Monomial{ .Factors = {}, .Coeff = 1.0, .HasFree = false } }; // multiplicative identity
                    for (int k = 0; k < *exp; ++k) { acc = MultiplySum(acc, childSums[0]); }
                    return acc;
                }
                return OpaqueFunction("pow", childSums);
            }

            // Every remaining built-in op (Exp/Log/Logabs/Log1p/Sin/Cos/Tan/Sinh/Cosh/Tanh/Sqrt/Sqrtabs/Cbrt/Abs/
            // Floor/Ceil/Aq/Powabs/Fmin/Fmax) is treated as an opaque function - see the header's doc comment.
            // Sorting the child keys only for the *commutative* ones (Fmin/Fmax - neither reachable from this
            // grammar, but handled for robustness) keeps the fallback safe if that ever changes; every other op's
            // argument order is semantically significant and must not be reordered.
            return OpaqueFunction(n.Name(), childSums, /*commutative=*/n.IsCommutative());
        }

    private:
        auto OpaqueFunction(std::string const& name, std::vector<Sum> const& childSums, bool commutative = false) -> Sum
        {
            std::vector<std::string> args;
            args.reserve(childSums.size());
            for (auto const& c : childSums) { args.push_back(SerializeSum(c)); }
            if (commutative) { std::ranges::sort(args); }
            bool hasFree = false;
            for (auto const& c : childSums) { for (auto const& m : c) { hasFree = hasFree || m.HasFree; } }
            std::string key = name + "(";
            for (std::size_t i = 0; i < args.size(); ++i) { if (i) { key += ","; } key += args[i]; }
            key += ")";
            return OpaqueLeaf(std::move(key), hasFree);
        }

        auto Indices(std::size_t i) const -> std::vector<std::size_t>
        {
            std::vector<std::size_t> idx;
            idx.reserve(nodes_[i].Arity);
            for (auto j : Operon::Tree::Indices(nodes_, i)) { idx.push_back(j); }
            return idx;
        }

        Operon::Span<Operon::Node const> nodes_;
    };
} // namespace

auto CanonicalizeEnumerationTree(Operon::Tree const& tree) -> CanonicalExpression
{
    Tree representative{ tree };
    representative.Reduce().Simplify();
    (void)representative.Hash(Operon::HashMode::Strict);
    representative.Sort();

    if (tree.Empty()) { return CanonicalExpression{ .Key = "0", .Representative = std::move(representative) }; }

    Canonicalizer canon{ tree };
    auto sum = canon.Canonicalize(tree.Nodes().size() - 1);
    return CanonicalExpression{ .Key = SerializeSum(sum), .Representative = std::move(representative) };
}

} // namespace Operon
