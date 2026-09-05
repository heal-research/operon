// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/operators/mutation.hpp"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <random>
#include <type_traits>

#include "../core/subtree_rewrite.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/initializer.hpp"

namespace Operon {

auto DiscretePointMutation::operator()(Operon::RandomGenerator& random, Tree tree) const -> Tree
{
    auto& nodes = tree.Nodes();
    auto it = Operon::Random::Sample(random, nodes.begin(), nodes.end(), [](auto const& n) -> auto { return n.IsLeaf(); });
    ENSURE(it < nodes.end());

    auto s = std::reduce(weights_.cbegin(), weights_.cend(), Operon::Scalar { 0 }, std::plus {});
    auto r = std::uniform_real_distribution<Operon::Scalar>(0., s)(random);

    Operon::Scalar c { 0 };
    for (auto i = 0UL; i < weights_.size(); ++i) {
        c += weights_[i];
        if (c > r) {
            it->Value = values_[i];
            break;
        }
    }

    return tree;
}

auto MultiMutation::operator()(Operon::RandomGenerator& random, Tree tree) const -> Tree
{
    auto sum = std::reduce(probabilities_.begin(), probabilities_.end());
    auto r = std::uniform_real_distribution<double>(0, sum)(random);
    auto c = 0.0;
    auto i = 0U;
    for (; i < probabilities_.size(); ++i) {
        c += probabilities_[i];
        if (c > r) {
            break;
        }
    }
    auto op = operators_[i];
    return (*op)(random, std::move(tree));
}

auto ChangeVariableMutation::operator()(Operon::RandomGenerator& random, Tree tree) const -> Tree
{
    auto& nodes = tree.Nodes();
    auto it = Operon::Random::Sample(random, nodes.begin(), nodes.end(), [](auto const& n) -> auto { return n.IsVariable(); });
    if (it == nodes.end()) {
        return tree; // no variables in the tree, nothing to do
    }

    it->HashValue = it->CalculatedHashValue = *Random::Sample(random, variables_.begin(), variables_.end());
    return tree;
}

auto ChangeFunctionMutation::operator()(Operon::RandomGenerator& random, Tree tree) const -> Tree
{
    auto& nodes = tree.Nodes();

    auto it = Operon::Random::Sample(random, nodes.begin(), nodes.end(), [](auto const& n) -> auto { return !n.IsLeaf(); });
    if (it == nodes.end()) {
        return tree; // no functions in the tree, nothing to do
    }

    auto arity = static_cast<size_t>(it->Arity);

    auto n = pset_.SampleRandomSymbol(random, arity, arity);
    it->Type = n.Type;
    it->HashValue = n.HashValue;
    return tree;
}

auto ReplaceSubtreeMutation::operator()(Operon::RandomGenerator& random, Tree tree) const -> Tree
{
    auto const& nodes = tree.Nodes();
    auto i = std::uniform_int_distribution<size_t>(0, nodes.size() - 1)(random);
    auto const target = detail::DescribeSubtree(Operon::Span<Node const> { nodes }, i);
    auto const oldLen = target.Size;
    auto const oldLevel = nodes[i].Level;

    using Signed = std::make_signed_t<size_t>;
    auto const partialLength = nodes.size() - oldLen;
    auto maxLength = static_cast<Signed>(maxLength_ - partialLength);
    maxLength = std::max(maxLength, Signed { 1 });
    auto maxDepth = std::max(tree.Depth(), maxDepth_) - oldLevel + 1;

    auto const newLen = std::uniform_int_distribution<Signed>(Signed { 1 }, maxLength)(random);
    auto subtree = (*creator_)(random, static_cast<size_t>(newLen), 1, maxDepth);
    (*coefficientInitializer_)(random, subtree);
    auto rewritten = detail::RewriteSubtree(Operon::Span<Node const> { nodes }, target,
        Operon::Span<Node const> { subtree.Nodes() });
    return Tree(std::move(rewritten)).UpdateNodes();
}

auto RemoveChildMutation::operator()(Operon::RandomGenerator& random, Tree tree) const -> Tree
{
    auto& nodes = tree.Nodes();

    if (nodes.size() == 1) {
        return tree; // nothing to remove
    }

    auto it = Operon::Random::Sample(random, nodes.begin(), nodes.end() - 1); // -1 because we don't want to remove the tree root
    auto const& p = nodes[it->Parent];
    if (p.Arity > pset_.MinimumArity(p.HashValue)) {
        nodes[it->Parent].Arity--;
        nodes.erase(it - it->Length, it + 1);
        tree.UpdateNodes();
    }
    return tree;
}

auto InsertSubtreeMutation::operator()(Operon::RandomGenerator& random, Tree tree) const -> Tree
{
    if (tree.Length() >= maxLength_) {
        // we can't insert anything because the tree length is at the limit
        return tree;
    }

    auto& nodes = tree.Nodes();
    auto const* pset = creator_->GetPrimitiveSet();

    auto test = [&](auto const& node) -> auto {
        return node.template IsOp<BuiltinOp::Add, BuiltinOp::Mul, BuiltinOp::Sub, BuiltinOp::Div>() && (node.Arity < pset->MaximumArity(node.HashValue));
    };

    auto n = std::count_if(nodes.begin(), nodes.end(), test);

    if (n == 0) {
        return tree;
    }

    auto index = std::uniform_int_distribution<decltype(n)>(1, n)(random);
    size_t i = 0;
    for (; i < nodes.size(); ++i) {
        if (test(nodes[i]) && --index == 0) {
            break;
        }
    }

    auto availableLength = maxLength_ - nodes.size();
    EXPECT(availableLength > 0);

    auto availableDepth = std::max(tree.Depth(), maxDepth_) - nodes[i].Level;
    EXPECT(availableDepth > 0);

    auto newLen = std::uniform_int_distribution<size_t>(1, availableLength)(random);

    auto subtree = (*creator_)(random, newLen, 1, availableDepth);
    (*coefficientInitializer_)(random, subtree);

    Operon::Vector<Node> mutated;
    mutated.reserve(nodes.size() + newLen);

    // increase parent arity
    nodes[i].Arity++;

    using Signed = std::make_signed_t<size_t>;
    // copy nodes
    std::copy(nodes.begin(), nodes.begin() + static_cast<Signed>(i - nodes[i].Length), std::back_inserter(mutated));
    std::copy(subtree.Nodes().begin(), subtree.Nodes().end(), std::back_inserter(mutated));
    std::copy(nodes.begin() + static_cast<Signed>(i - nodes[i].Length), nodes.end(), std::back_inserter(mutated));

    return Tree(mutated).UpdateNodes();
}

auto RemoveSubtreeMutation::operator()(Operon::RandomGenerator& random, Tree tree) const -> Tree
{
    auto const& nodes = tree.Nodes();
    auto i = std::uniform_int_distribution<size_t>(0, nodes.size() - 1)(random);
    auto const target = detail::DescribeSubtree(Operon::Span<Node const> { nodes }, i);
    auto const oldLevel = nodes[i].Level;
    auto const maxDepth = std::max(tree.Depth(), maxDepth_) - oldLevel + 1;

    // Always replace with the smallest possible subtree, a single terminal.
    auto subtree = (*creator_)(random, size_t { 1 }, 1, maxDepth);
    (*coefficientInitializer_)(random, subtree);
    auto rewritten = detail::RewriteSubtree(Operon::Span<Node const> { nodes }, target,
        Operon::Span<Node const> { subtree.Nodes() });
    return Tree(std::move(rewritten)).UpdateNodes();
}
auto ShuffleSubtreesMutation::operator()(Operon::RandomGenerator& random, Tree tree) const -> Tree
{
    auto& nodes = tree.Nodes();
    auto const nFunc = std::count_if(nodes.begin(), nodes.end(), [](Node const& node) { return !node.IsLeaf(); });
    if (nFunc == 0) {
        return tree;
    }

    auto const selected = std::uniform_int_distribution<std::make_signed_t<size_t>>(1, nFunc)(random);
    auto const root = [&] {
        auto remaining = selected;
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            if (!nodes[i].IsLeaf() && --remaining == 0) {
                return i;
            }
        }
        UNREACHABLE();
    }();

    auto const span = detail::DescribeSubtree(Operon::Span<Node const> { nodes }, root);
    Operon::Vector<detail::SubtreeSpan> children;
    children.reserve(nodes[root].Arity);
    for (auto const child : tree.Indices(root)) {
        children.push_back(detail::DescribeSubtree(Operon::Span<Node const> { nodes }, child));
    }
    std::shuffle(children.begin(), children.end(), random);
    if (children.size() < 2) {
        return tree;
    }

    auto original = tree.Indices(root).begin();
    auto const unchanged = std::ranges::all_of(children, [&](detail::SubtreeSpan const& child) { return child.Root == *original++; });
    if (unchanged) {
        return tree;
    }

    if (!std::ranges::any_of(nodes, [](Node const& node) { return node.IsRef(); })) {
        Operon::Vector<Node> buffer(nodes.begin() + static_cast<std::ptrdiff_t>(span.First), nodes.begin() + static_cast<std::ptrdiff_t>(root));
        auto destination = nodes.begin() + static_cast<std::ptrdiff_t>(span.First);
        for (auto const child : children) {
            auto const first = child.First - span.First;
            std::copy_n(buffer.begin() + static_cast<std::ptrdiff_t>(first), child.Size, destination);
            destination += static_cast<std::ptrdiff_t>(child.Size);
        }
        return tree.UpdateNodes();
    }

    Operon::Vector<detail::PermutationSegment> segments;
    segments.reserve(children.size() + 3);
    if (span.First != 0) {
        segments.push_back({ 0, span.First });
    }
    for (auto const child : children) {
        segments.push_back({ child.First, child.Size });
    }
    segments.push_back({ root, 1 });
    if (root + 1 < nodes.size()) {
        segments.push_back({ root + 1, nodes.size() - root - 1 });
    }

    auto rewritten = detail::PermuteSegments(Operon::Span<Node const> { nodes }, segments);
    return rewritten ? Tree(std::move(*rewritten)).UpdateNodes() : tree;
}
} // namespace Operon
