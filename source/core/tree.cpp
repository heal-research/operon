// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "operon/core/tree.hpp"
#include "operon/hash/hash.hpp"
#include "operon/core/constants.hpp"
#include "operon/core/node.hpp"
#include "operon/core/types.hpp"
#include "subtree_rewrite.hpp"

namespace Operon {
auto Tree::Splice(size_t i) const -> Tree
{
    auto const span = detail::DescribeSubtree(Operon::Span<Node const>{nodes_}, i);
    return Tree(detail::CopySubtree(Operon::Span<Node const>{nodes_}, span)).UpdateNodes();
}
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

auto Tree::Validate() const -> tl::expected<void, TreeValidationError>
{
    if (nodes_.empty()) { return {}; }

    struct CompletedSubtree {
        size_t First;
        size_t Root;
        size_t Depth;
    };

    std::vector<CompletedSubtree> stack;
    stack.reserve(nodes_.size());
    std::vector<size_t> parents(nodes_.size());
    std::vector<size_t> lengths(nodes_.size());
    std::vector<size_t> depths(nodes_.size());

    for (size_t i = 0; i < nodes_.size(); ++i) {
        auto const& node = nodes_[i];
        switch (node.Type) {
        case NodeType::Constant:
        case NodeType::Variable:
        case NodeType::Ref:
        case NodeType::Function:
            break;
        default:
            return tl::make_unexpected(TreeValidationError::InvalidNodeType);
        }
        if (node.IsRef() && node.RefTo >= i) {
            return tl::make_unexpected(TreeValidationError::RefNotBackward);
        }

        if (node.IsFunction() == node.IsLeaf()) {
            return tl::make_unexpected(node.IsFunction()
                ? TreeValidationError::FunctionArityZero
                : TreeValidationError::TerminalArityNonZero);
        }
        if (node.IsLeaf()) {
            depths[i] = 1;
            stack.push_back({i, i, 1});
            continue;
        }

        auto const arity = static_cast<size_t>(node.Arity);
        if (stack.size() < arity) {
            return tl::make_unexpected(TreeValidationError::MissingChildren);
        }

        auto const firstChild = stack.size() - arity;
        auto const first      = stack[firstChild].First;
        if (firstChild > 0 && stack[firstChild - 1].Root + 1 != first) {
            return tl::make_unexpected(TreeValidationError::ChildSubtreesNotContiguous);
        }
        if (stack.back().Root + 1 != i) {
            return tl::make_unexpected(TreeValidationError::ChildSubtreesNotAdjacent);
        }

        size_t depth{1};
        for (auto j = firstChild; j < stack.size(); ++j) {
            parents[stack[j].Root] = i;
            depth = std::max(depth, stack[j].Depth + 1);
        }
        stack.erase(stack.begin() + static_cast<std::ptrdiff_t>(firstChild), stack.end());
        stack.push_back({first, i, depth});
        lengths[i] = i - first;
        depths[i]  = depth;
    }

    if (stack.size() != 1) {
        return tl::make_unexpected(TreeValidationError::MultipleRoots);
    }
    if (stack.front().First != 0 || stack.front().Root != nodes_.size() - 1) {
        return tl::make_unexpected(TreeValidationError::RootDoesNotCoverTree);
    }

    std::vector<size_t> levels(nodes_.size());
    levels.back() = 1;
    for (size_t i = nodes_.size() - 1; i > 0; --i) {
        levels[i - 1] = levels[parents[i - 1]] + 1;
    }

    for (size_t i = 0; i < nodes_.size(); ++i) {
        auto const& node = nodes_[i];
        if (lengths[i] > std::numeric_limits<uint16_t>::max()
            || depths[i] > std::numeric_limits<uint16_t>::max()
            || parents[i] > std::numeric_limits<uint16_t>::max()
            || levels[i] > std::numeric_limits<uint16_t>::max()) {
            return tl::make_unexpected(TreeValidationError::DerivedMetadataOverflow);
        }
        if (node.Length != lengths[i]) {
            return tl::make_unexpected(TreeValidationError::LengthMismatch);
        }
        if (node.Depth != depths[i]) {
            return tl::make_unexpected(TreeValidationError::DepthMismatch);
        }
        if (node.Parent != parents[i]) {
            return tl::make_unexpected(TreeValidationError::ParentMismatch);
        }
        if (node.Level != levels[i]) {
            return tl::make_unexpected(TreeValidationError::LevelMismatch);
        }
    }
    return {};
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
    // Each entry identifies the original node currently held at its position.
    // Refs retain their original target while children are rearranged, then are
    // rewritten once from this final old-to-new mapping.
    Operon::Vector<Operon::Node> sorted = nodes_;
    Operon::Vector<size_t> origins(sorted.size());
    std::iota(origins.begin(), origins.end(), size_t{0});

    Operon::Vector<size_t> destinations(sorted.size());
    std::iota(destinations.begin(), destinations.end(), size_t{0});

    struct ChildSpan {
        Operon::Node Root;
        size_t First;
        size_t Size;
    };
    Operon::Vector<ChildSpan> children;
    children.reserve(nodes_.size());

    for (size_t i = 0; i < sorted.size(); ++i) {
        auto const& parent = sorted[i];
        if (parent.IsLeaf() || !parent.IsCommutative()) {
            continue;
        }

        auto const first = i - parent.Length;
        if (parent.Arity == parent.Length) {
            for (size_t j = first; j < i; ++j) {
                children.push_back({ sorted[j], j, 1U });
            }
        } else {
            for (auto const j : Tree::Indices(sorted, i)) {
                auto const size = static_cast<size_t>(sorted[j].Length) + 1U;
                children.push_back({ sorted[j], j + 1U - size, size });
            }
        }
        std::stable_sort(children.begin(), children.end(), [](ChildSpan const& lhs, ChildSpan const& rhs) {
            return lhs.Root < rhs.Root;
        });

        Operon::Vector<Operon::Node> const buffer(sorted.begin() + static_cast<std::ptrdiff_t>(first), sorted.begin() + static_cast<std::ptrdiff_t>(i));
        Operon::Vector<size_t> const sourceOrigins(origins.begin() + static_cast<std::ptrdiff_t>(first), origins.begin() + static_cast<std::ptrdiff_t>(i));
        Operon::Vector<Operon::Node> reordered;
        Operon::Vector<size_t> reorderedOrigins;
        reordered.reserve(buffer.size());
        reorderedOrigins.reserve(sourceOrigins.size());

        for (auto const& child : children) {
            auto const offset = child.First - first;
            std::copy_n(buffer.begin() + static_cast<std::ptrdiff_t>(offset), static_cast<std::ptrdiff_t>(child.Size), std::back_inserter(reordered));
            std::copy_n(sourceOrigins.begin() + static_cast<std::ptrdiff_t>(offset), static_cast<std::ptrdiff_t>(child.Size), std::back_inserter(reorderedOrigins));
        }

        for (size_t j = 0; j < reorderedOrigins.size(); ++j) {
            destinations[reorderedOrigins[j]] = first + j;
        }
        auto const preservesRefs = std::ranges::all_of(reordered, [&](Node const& node) {
            return !node.IsRef() || destinations[node.RefTo] < first + static_cast<size_t>(&node - reordered.data());
        });
        if (!preservesRefs) {
            for (size_t j = 0; j < sourceOrigins.size(); ++j) {
                destinations[sourceOrigins[j]] = first + j;
            }
            children.clear();
            continue;
        }

        std::ranges::copy(reordered, sorted.begin() + static_cast<std::ptrdiff_t>(first));
        std::ranges::copy(reorderedOrigins, origins.begin() + static_cast<std::ptrdiff_t>(first));
        children.clear();
    }

    for (size_t i = 0; i < sorted.size(); ++i) {
        if (sorted[i].IsRef()) {
            sorted[i].RefTo = static_cast<uint16_t>(destinations[sorted[i].RefTo]);
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
    if (coefficients.size() != static_cast<size_t>(CoefficientsCount())) {
        throw std::invalid_argument("coefficient count must match the number of optimizable tree nodes");
    }
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
    using BO = BuiltinOp;
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
                switch (n.HashValue) {
                case Operon::Hash(BO::Add):
                    val = S{0};
                    for (auto j : ch) { val += nodes_[j].Value; }
                    break;
                case Operon::Hash(BO::Mul):
                    val = S{1};
                    for (auto j : ch) { val *= nodes_[j].Value; }
                    break;
                case Operon::Hash(BO::Sub):
                    // arity 1 is negation (-x, see InfixFormatter::FormatNode), not identity -
                    // the n-ary accumulator below is only correct for arity >= 2.
                    if (ch.size() == 1) { val = -nodes_[ch[0]].Value; break; }
                    for (auto j : ch) { acc = acc ? *acc - nodes_[j].Value : nodes_[j].Value; }
                    val = acc.value_or(S{0});
                    break;
                case Operon::Hash(BO::Div):
                    // arity 1 is inversion (1/x, see InfixFormatter::FormatNode), not identity.
                    if (ch.size() == 1) { val = S{1} / nodes_[ch[0]].Value; break; }
                    for (auto j : ch) { acc = acc ? *acc / nodes_[j].Value : nodes_[j].Value; }
                    val = acc.value_or(S{0});
                    break;
                case Operon::Hash(BO::Fmin):
                    for (auto j : ch) { acc = acc ? std::min(*acc, nodes_[j].Value) : nodes_[j].Value; }
                    val = acc.value_or(S{0});
                    break;
                case Operon::Hash(BO::Fmax):
                    for (auto j : ch) { acc = acc ? std::max(*acc, nodes_[j].Value) : nodes_[j].Value; }
                    val = acc.value_or(S{0});
                    break;
                case Operon::Hash(BO::Pow):    val = std::pow(nodes_[ch[0]].Value, nodes_[ch[1]].Value);                break;
                case Operon::Hash(BO::Powabs): val = std::pow(std::abs(nodes_[ch[0]].Value), nodes_[ch[1]].Value);     break;
                case Operon::Hash(BO::Aq):     { S y = nodes_[ch[1]].Value; val = nodes_[ch[0]].Value / std::sqrt(S{1} + (y*y)); break; }
                case Operon::Hash(BO::Exp):    val = std::exp(nodes_[ch[0]].Value);                                     break;
                case Operon::Hash(BO::Log):    val = std::log(nodes_[ch[0]].Value);                                     break;
                case Operon::Hash(BO::Log1p):  val = std::log1p(nodes_[ch[0]].Value);                                  break;
                case Operon::Hash(BO::Logabs): val = std::log(std::abs(nodes_[ch[0]].Value));                          break;
                case Operon::Hash(BO::Sin):    val = std::sin(nodes_[ch[0]].Value);                                     break;
                case Operon::Hash(BO::Cos):    val = std::cos(nodes_[ch[0]].Value);                                     break;
                case Operon::Hash(BO::Tan):    val = std::tan(nodes_[ch[0]].Value);                                     break;
                case Operon::Hash(BO::Sinh):   val = std::sinh(nodes_[ch[0]].Value);                                    break;
                case Operon::Hash(BO::Cosh):   val = std::cosh(nodes_[ch[0]].Value);                                    break;
                case Operon::Hash(BO::Tanh):   val = std::tanh(nodes_[ch[0]].Value);                                    break;
                case Operon::Hash(BO::Sqrt):   val = std::sqrt(nodes_[ch[0]].Value);                                    break;
                case Operon::Hash(BO::Sqrtabs):val = std::sqrt(std::abs(nodes_[ch[0]].Value));                         break;
                case Operon::Hash(BO::Cbrt):   val = std::cbrt(nodes_[ch[0]].Value);                                    break;
                case Operon::Hash(BO::Square): val = nodes_[ch[0]].Value * nodes_[ch[0]].Value;                        break;
                case Operon::Hash(BO::Abs):    val = std::abs(nodes_[ch[0]].Value);                                     break;
                case Operon::Hash(BO::Floor):  val = std::floor(nodes_[ch[0]].Value);                                   break;
                case Operon::Hash(BO::Ceil):   val = std::ceil(nodes_[ch[0]].Value);                                    break;
                default:         handled = false;                                                          break;
                }
                if (handled) {
                    foldToConst(i, val); // n.Value == 1.0 for all function nodes
                    changed = true;
                    break; // see the `if (changed) { break; }` below - same reasoning
                }
            }

            // --- Identity and annihilator rules ---
            switch (n.HashValue) {
            case Operon::Hash(BO::Add): {
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
            case Operon::Hash(BO::Mul): {
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
            case Operon::Hash(BO::Sub): {
                // Arity 1 is negation (-x), not identity - nothing to fold here
                // (the loop below never runs at arity 1, which would otherwise
                // leave newArity==1 and incorrectly disable the node as if its
                // one child were the whole value).
                if (n.Arity < 2) { break; }
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
            case Operon::Hash(BO::Div): {
                // Arity 1 is inversion (1/x), not identity - same reasoning as Sub above.
                if (n.Arity < 2) { break; }
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
            case Operon::Hash(BO::Pow): {
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
                        n.HashValue = Operon::Hash(BO::Square); // Type stays Function - only HashValue distinguishes ops now
                        n.Arity = 1;
                        changed = true;
                    } else if (nodes_[expIdx].Value == S{0.5}) {
                        nodes_[expIdx].IsEnabled = false;  // Pow(x,0.5) → Sqrt(x)
                        n.HashValue = Operon::Hash(BO::Sqrt);
                        n.Arity = 1;
                        changed = true;
                    }
                } else if (nodes_[baseIdx].IsConstant() && nodes_[baseIdx].Value == S{1}) {
                    foldToConst(i, S{1}); changed = true;  // 1^x = 1
                }
                break;
            }
            case Operon::Hash(BO::Powabs): {
                if (ch.size() != 2) { break; }
                auto const expIdx = ch[1];
                if (nodes_[expIdx].IsConstant() && nodes_[expIdx].Value == S{0}) {
                    foldToConst(i, S{1}); changed = true;  // |x|^0 = 1
                }
                // Note: |x|^1 = |x| ≠ x in general, so we do NOT simplify Powabs(x,1).
                break;
            }
            case Operon::Hash(BO::Log):
            case Operon::Hash(BO::Logabs): {
                // log(exp(x)) = x for all x; log|exp(x)| = x likewise.
                if (ch.size() == 1 && nodes_[ch[0]].IsOp<BO::Exp>()) {
                    n.IsEnabled = false;
                    nodes_[ch[0]].IsEnabled = false;
                    changed = true;
                }
                break;
            }
            case Operon::Hash(BO::Sqrt):
            case Operon::Hash(BO::Sqrtabs): {
                // sqrt(x^2) = |x|;  sqrt(|x^2|) = |x|
                if (ch.size() == 1 && nodes_[ch[0]].IsOp<BO::Square>()) {
                    nodes_[ch[0]].IsEnabled = false;
                    n.HashValue = Operon::Hash(BO::Abs); // Type stays Function
                    n.Arity = 1; // Sqrt/Sqrtabs and Abs are both unary; explicit for symmetry with the Pow retargets above
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
