// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <functional>
#include <iterator>
#include <optional>
#include <vector>

#include "operon/core/tree.hpp"
#include "operon/hash/hash.hpp"
#include "operon/core/constants.hpp"
#include "operon/core/node.hpp"
#include "operon/core/types.hpp"

namespace Operon {
auto Tree::UpdateNodes() -> Tree&
{
    if (nodes_.empty()) { return *this; }
    for (size_t i = 0; i < nodes_.size(); ++i) {
        auto& s = nodes_[i];

        s.Depth = 1;
        s.Length = s.Arity;
        s.Parent = 0;

        if (s.IsLeaf()) {
            continue;
        }

        for (auto& p : Tree::Nodes(nodes_, i)) {
            s.Length += p.Length;
            s.Depth = std::max(s.Depth, p.Depth);
            p.Parent = i;
        }

        ++s.Depth;
    }
    nodes_.back().Level = 1;

    for (auto it = nodes_.rbegin() + 1; it < nodes_.rend(); ++it) {
        it->Level = static_cast<uint16_t>(nodes_[it->Parent].Level + 1);
    }

    return *this;
}

auto Tree::Reduce() -> Tree&
{
    bool reduced = false;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        auto& s = nodes_[i];
        if (s.IsLeaf() || !s.IsCommutative()) {
            continue;
        }
        auto arity{ s.Arity };
        for (auto& p : Children(i)) {
            if (s.HashValue == p.HashValue) {
                p.IsEnabled = false;
                arity = static_cast<uint16_t>(arity + p.Arity - 1);
                reduced = true;
            }
        }
        s.Arity = arity;
    }

    // if anything was reduced (nodes were disabled), copy remaining enabled nodes
    if (reduced) {
        std::erase_if(nodes_, [](auto const& n) -> auto { return !n.IsEnabled; });
    }
    // else, nothing to do
    return this->UpdateNodes();
}

// Sort each function node's children according to node type and hash value
// - note that entire child subtrees / subarrays are reordered inside the nodes array
// - this method assumes node hashes are computed, usually it is preceded by a call to tree.Hash()
auto Tree::Sort() -> Tree&
{
    // preallocate memory to reduce fragmentation
    Operon::Vector<Operon::Node> sorted = nodes_;

    Operon::Vector<size_t> children;
    children.reserve(nodes_.size());

    auto start = nodes_.begin();

    for (size_t i = 0; i < nodes_.size(); ++i) {
        auto& s = nodes_[i];

        if (s.IsLeaf()) {
            continue;
        }

        auto arity = s.Arity;
        auto size = s.Length;

        if (s.IsCommutative()) {
            if (arity == size) {
                std::stable_sort(start + i - size, start + i); // NOLINT
            } else {
                std::ranges::copy(Indices(i), std::back_inserter(children));
                std::stable_sort(children.begin(), children.end(), [&](auto a, auto b) -> auto { return nodes_[a] < nodes_[b]; }); // sort child indices

                auto pos = sorted.begin() + i - size; // NOLINT
                for (auto j : children) {
                    auto& c = nodes_[j];
                    std::copy_n(start + static_cast<int64_t>(j) - c.Length, c.Length + 1, pos);
                    pos += c.Length + 1;
                }
                children.clear();
            }
        }
    }
    nodes_.swap(sorted);
    return this->UpdateNodes();
}

auto Tree::GetCoefficients() const -> std::vector<Operon::Scalar>
{
    std::vector<Operon::Scalar> coefficients;
    for (auto const& n : nodes_) {
        if (n.Optimize) {
            coefficients.push_back(n.Value);
        }
    }
    return coefficients;
}

void Tree::GetCoefficients(std::vector<Operon::Scalar>& out) const
{
    out.clear();
    for (auto const& n : nodes_) {
        if (n.Optimize) { out.push_back(n.Value); }
    }
}

void Tree::SetCoefficients(Operon::Span<Operon::Scalar const> coefficients)
{
    size_t idx = 0;
    for (auto& s : nodes_) {
        if (s.Optimize) { s.Value = coefficients[idx++]; }
    }
}

auto Tree::Depth() const noexcept -> size_t
{
    return Empty() ? 0 : nodes_.back().Depth;
}

auto Tree::VisitationLength() const noexcept -> size_t
{
    return std::transform_reduce(nodes_.begin(), nodes_.end(), 0UL, std::plus<> {}, [](const auto& node) -> auto { return node.Length + 1; });
}

auto Tree::Hash(Operon::HashMode mode) const -> Tree const&
{
    std::vector<size_t> childIndices;
    childIndices.reserve(nodes_.size());

    std::vector<Operon::Hash> hashes;
    hashes.reserve(nodes_.size());

    Operon::Hasher const hasher;

    for (size_t i = 0; i < nodes_.size(); ++i) {
        auto const& n = nodes_[i];

        if (n.IsLeaf()) {
            if (n.IsRef()) {
                // A Ref inherits the hash of its target so structurally equivalent
                // subexpressions produce the same tree hash regardless of sharing.
                EXPECT(n.RefTo < i); // must be a backward reference
                n.CalculatedHashValue = nodes_[n.RefTo].CalculatedHashValue;
            } else {
                n.CalculatedHashValue = n.HashValue;
                if (mode == Operon::HashMode::Strict) {
                    n.CalculatedHashValue += hasher(reinterpret_cast<uint8_t const*>(&n.Value), sizeof(n.Value)); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
                }
            }
            continue;
        }

        std::ranges::copy(Indices(i), std::back_inserter(childIndices));

        auto begin = childIndices.begin();
        auto end = begin + n.Arity;

        if (n.IsCommutative()) {
            std::stable_sort(begin, end, [&](auto a, auto b) -> auto { return nodes_[a] < nodes_[b]; });
        }
        std::transform(begin, end, std::back_inserter(hashes), [&](auto j) -> auto { return nodes_[j].CalculatedHashValue; });
        hashes.push_back(n.HashValue);

        n.CalculatedHashValue = hasher(std::bit_cast<uint8_t*>(hashes.data()), sizeof(Operon::Hash) * hashes.size()); // NOLINT
        childIndices.clear();
        hashes.clear();
    }

    return *this;
}

auto Tree::Simplify() -> Tree& {
    using NT = NodeType;
    using S  = Operon::Scalar;

    if (nodes_.empty()) { return *this; }

    // Replace the span [i-L, i] with a single Const node.
    // Disables all old children; repurposes node i as Const(value).
    auto foldToConst = [&](std::size_t i, S value) {
        for (auto j = i - nodes_[i].Length; j < i; ++j) { nodes_[j].IsEnabled = false; }
        nodes_[i] = Node::Constant(static_cast<double>(value));
    };

    bool changed = true;
    while (changed) {
        changed = false;

        for (std::size_t i = 0; i < nodes_.size(); ++i) {
            auto& n = nodes_[i];
            if (!n.IsEnabled || n.IsLeaf()) { continue; }

            // Direct child root indices, in first-operand-first order.
            std::vector<std::size_t> ch;
            ch.reserve(n.Arity);
            for (auto j : Indices(i)) { ch.push_back(j); }

            // --- Constant folding: all direct children are Const leaves ---
            // Running until convergence lets deeper const subtrees fold up in
            // subsequent passes (bottom-up, one level per pass).
            bool allConst = std::all_of(ch.begin(), ch.end(),
                [&](std::size_t j) { return nodes_[j].IsConstant(); });

            if (allConst && !ch.empty()) {
                std::optional<S> acc;
                bool handled = true;
                S val{};
                switch (n.Type) {
                case NT::Add:
                    val = S{0};
                    for (auto j : ch) { val += nodes_[j].Value; }
                    break;
                case NT::Mul:
                    val = S{1};
                    for (auto j : ch) { val *= nodes_[j].Value; }
                    break;
                case NT::Sub:
                    for (auto j : ch) { acc = acc ? *acc - nodes_[j].Value : nodes_[j].Value; }
                    val = acc.value_or(S{0});
                    break;
                case NT::Div:
                    for (auto j : ch) { acc = acc ? *acc / nodes_[j].Value : nodes_[j].Value; }
                    val = acc.value_or(S{0});
                    break;
                case NT::Fmin:
                    for (auto j : ch) { acc = acc ? std::min(*acc, nodes_[j].Value) : nodes_[j].Value; }
                    val = acc.value_or(S{0});
                    break;
                case NT::Fmax:
                    for (auto j : ch) { acc = acc ? std::max(*acc, nodes_[j].Value) : nodes_[j].Value; }
                    val = acc.value_or(S{0});
                    break;
                case NT::Pow:    val = std::pow(nodes_[ch[0]].Value, nodes_[ch[1]].Value);                break;
                case NT::Powabs: val = std::pow(std::abs(nodes_[ch[0]].Value), nodes_[ch[1]].Value);     break;
                case NT::Aq:     { S y = nodes_[ch[1]].Value; val = nodes_[ch[0]].Value / std::sqrt(S{1} + (y*y)); break; }
                case NT::Exp:    val = std::exp(nodes_[ch[0]].Value);                                     break;
                case NT::Log:    val = std::log(nodes_[ch[0]].Value);                                     break;
                case NT::Log1p:  val = std::log1p(nodes_[ch[0]].Value);                                  break;
                case NT::Logabs: val = std::log(std::abs(nodes_[ch[0]].Value));                          break;
                case NT::Sin:    val = std::sin(nodes_[ch[0]].Value);                                     break;
                case NT::Cos:    val = std::cos(nodes_[ch[0]].Value);                                     break;
                case NT::Tan:    val = std::tan(nodes_[ch[0]].Value);                                     break;
                case NT::Sinh:   val = std::sinh(nodes_[ch[0]].Value);                                    break;
                case NT::Cosh:   val = std::cosh(nodes_[ch[0]].Value);                                    break;
                case NT::Tanh:   val = std::tanh(nodes_[ch[0]].Value);                                    break;
                case NT::Sqrt:   val = std::sqrt(nodes_[ch[0]].Value);                                    break;
                case NT::Sqrtabs:val = std::sqrt(std::abs(nodes_[ch[0]].Value));                         break;
                case NT::Cbrt:   val = std::cbrt(nodes_[ch[0]].Value);                                    break;
                case NT::Square: val = nodes_[ch[0]].Value * nodes_[ch[0]].Value;                        break;
                case NT::Abs:    val = std::abs(nodes_[ch[0]].Value);                                     break;
                case NT::Floor:  val = std::floor(nodes_[ch[0]].Value);                                   break;
                case NT::Ceil:   val = std::ceil(nodes_[ch[0]].Value);                                    break;
                default:         handled = false;                                                          break;
                }
                if (handled) {
                    foldToConst(i, val); // n.Value == 1.0 for all function nodes
                    changed = true;
                    break; // see the `if (changed) { break; }` below - same reasoning
                }
            }

            // --- Identity and annihilator rules ---
            switch (n.Type) {
            case NT::Add: {
                auto newArity = n.Arity;
                for (auto j : ch) {
                    if (nodes_[j].IsConstant() && nodes_[j].Value == S{0}) {
                        nodes_[j].IsEnabled = false;
                        --newArity;
                        changed = true;
                    }
                }
                if      (newArity == 0) { foldToConst(i, S{0}); }
                else if (newArity == 1) { n.IsEnabled = false; }
                else                   { n.Arity = newArity; }
                break;
            }
            case NT::Mul: {
                bool hasZero = std::any_of(ch.begin(), ch.end(),
                    [&](std::size_t j) { return nodes_[j].IsConstant() && nodes_[j].Value == S{0}; });
                if (hasZero) { foldToConst(i, S{0}); changed = true; break; }
                auto newArity = n.Arity;
                for (auto j : ch) {
                    if (nodes_[j].IsConstant() && nodes_[j].Value == S{1}) {
                        nodes_[j].IsEnabled = false;
                        --newArity;
                        changed = true;
                    }
                }
                if      (newArity == 0) { foldToConst(i, S{1}); }
                else if (newArity == 1) { n.IsEnabled = false; }
                else                   { n.Arity = newArity; }
                break;
            }
            case NT::Sub: {
                // Remove Const(0) subtrahends (all children except the first).
                auto newArity = n.Arity;
                for (std::size_t ci = 1; ci < ch.size(); ++ci) {
                    auto j = ch[ci];
                    if (nodes_[j].IsConstant() && nodes_[j].Value == S{0}) {
                        nodes_[j].IsEnabled = false;
                        --newArity;
                        changed = true;
                    }
                }
                if      (newArity == 0) { foldToConst(i, S{0}); }
                else if (newArity == 1) { n.IsEnabled = false; } // only minuend left
                else                   { n.Arity = newArity; }
                break;
            }
            case NT::Div: {
                // Remove Const(1) denominators (all children except the first).
                auto newArity = n.Arity;
                for (std::size_t ci = 1; ci < ch.size(); ++ci) {
                    auto j = ch[ci];
                    if (nodes_[j].IsConstant() && nodes_[j].Value == S{1}) {
                        nodes_[j].IsEnabled = false;
                        --newArity;
                        changed = true;
                    }
                }
                if      (newArity == 0) { foldToConst(i, S{1}); }
                else if (newArity == 1) { n.IsEnabled = false; } // only numerator left
                else                   { n.Arity = newArity; }
                break;
            }
            case NT::Pow: {
                if (ch.size() != 2) { break; }
                auto const baseIdx = ch[0];
                auto const expIdx  = ch[1];
                if (nodes_[expIdx].IsConstant()) {
                    if (nodes_[expIdx].Value == S{0}) {
                        foldToConst(i, S{1}); changed = true; // x^0 = 1
                    } else if (nodes_[expIdx].Value == S{1}) {
                        nodes_[expIdx].IsEnabled = false;
                        n.IsEnabled = false;               // x^1 = x
                        changed = true;
                    } else if (nodes_[expIdx].Value == S{2}) {
                        nodes_[expIdx].IsEnabled = false;  // Pow(x,2) → Square(x)
                        n.Type = NT::Square;
                        n.HashValue = Node(NT::Square).HashValue;
                        n.Arity = 1;
                        changed = true;
                    } else if (nodes_[expIdx].Value == S{0.5}) {
                        nodes_[expIdx].IsEnabled = false;  // Pow(x,0.5) → Sqrt(x)
                        n.Type = NT::Sqrt;
                        n.HashValue = Node(NT::Sqrt).HashValue;
                        n.Arity = 1;
                        changed = true;
                    }
                } else if (nodes_[baseIdx].IsConstant() && nodes_[baseIdx].Value == S{1}) {
                    foldToConst(i, S{1}); changed = true;  // 1^x = 1
                }
                break;
            }
            case NT::Powabs: {
                if (ch.size() != 2) { break; }
                auto const expIdx = ch[1];
                if (nodes_[expIdx].IsConstant() && nodes_[expIdx].Value == S{0}) {
                    foldToConst(i, S{1}); changed = true;  // |x|^0 = 1
                }
                // Note: |x|^1 = |x| ≠ x in general, so we do NOT simplify Powabs(x,1).
                break;
            }
            case NT::Log:
            case NT::Logabs: {
                // log(exp(x)) = x for all x; log|exp(x)| = x likewise.
                if (ch.size() == 1 && nodes_[ch[0]].Type == NT::Exp) {
                    n.IsEnabled = false;
                    nodes_[ch[0]].IsEnabled = false;
                    changed = true;
                }
                break;
            }
            case NT::Sqrt:
            case NT::Sqrtabs: {
                // sqrt(x^2) = |x|;  sqrt(|x^2|) = |x|
                if (ch.size() == 1 && nodes_[ch[0]].Type == NT::Square) {
                    nodes_[ch[0]].IsEnabled = false;
                    n.Type = NT::Abs;
                    n.HashValue = Node(NT::Abs).HashValue;
                    changed = true;
                }
                break;
            }
            default: break;
            }

            // foldToConst above overwrites node i in place as a Length-0
            // Constant leaf, while erase_if/UpdateNodes haven't run yet this
            // pass - Indices()/Children() for any later j read Length to
            // step between siblings (see Subtree::SubtreeIterator::
            // operator++), so continuing this scan past a just-folded node
            // risks walking off into already-disabled leftover nodes instead
            // of the true next sibling. The other rewrites in this switch
            // (identity/annihilator disables, Pow/Sqrt type rewrites) don't
            // touch Length and would be safe to keep scanning past, but
            // bailing out unconditionally on any `changed` is simplest and
            // cheap here - Simplify() only runs per-candidate in
            // algorithms/enumeration.cpp, not in the GP hot loop. Resync
            // (erase_if + UpdateNodes) before any later index is examined;
            // the outer while(changed) picks the scan back up from a
            // consistent array.
            if (changed) { break; }
        }

        if (changed) {
            std::erase_if(nodes_, [](auto const& nd) { return !nd.IsEnabled; });
            UpdateNodes();
        }
    }

    return *this;
}

} // namespace Operon
