#ifndef OPERON_HASH_ZOBRIST_HPP
#define OPERON_HASH_ZOBRIST_HPP

#include <algorithm>
#include <bit>
#include <singleton_atomic.hpp>
#include <operon/operon_export.hpp>
#include <operon/mdspan/mdspan.hpp>
#include <operon/core/individual.hpp>
#include <fluky/xoshiro256ss.hpp>
#include <type_traits>

#include <parallel_hashmap/phmap.h>

namespace Operon {
OPERON_EXPORT class Zobrist : public SingletonAtomic<Zobrist> {
    using Extents = std::extents<int, NodeTypes::Count, std::dynamic_extent>;
    using Storage = std::experimental::mdarray<Operon::Hash, Extents>;

    Storage zobrist_;

    // transposition table
    using K = Operon::Hash;
    using V = std::tuple<Operon::Individual, std::size_t>;
    using Map = phmap::parallel_flat_hash_map<K, V, std::identity, std::equal_to<>, std::allocator<std::pair<K const, V>>, 4, std::mutex>;
    Map tt_;

    std::atomic_ullong hits_; // how many cache hits in the transposition table?

public:
    Zobrist(Operon::RandomGenerator& rng, int length) : zobrist_(NodeTypes::Count, length) {
        std::generate(zobrist_.container().begin(), zobrist_.container().end(), std::ref(rng));
    }

    [[nodiscard]] auto Rows() const { return zobrist_.extent(0); }
    [[nodiscard]] auto Cols() const { return zobrist_.extent(1); }

    [[nodiscard]] auto Hits() const { return hits_.load(); }
    [[nodiscard]] auto Total() const { return tt_.size(); }

    auto TranspositionTable() -> Map& { return tt_; }
    [[nodiscard]] auto TranspositionTable() const -> Map const& { return tt_; }

    auto Insert(Operon::Hash hash, Operon::Individual const& ind) {
        ENSURE(ind.Genotype.Length() > 0);
        if (!tt_.modify_if(hash, [&](auto& t) {
            ++hits_;
            std::get<1>(t.second) += 1;
        })) {
            tt_.insert({hash, {ind, 1}});
        };
    }

    auto Insert(Operon::Hash hash, Operon::Tree const& tree) {
        Operon::Individual ind;
        ind.Genotype = tree;
        Insert(hash, ind);
    }

    [[nodiscard]] auto Contains(Operon::Hash hash) const { return tt_.contains(hash); }

    // compute hashes
    static auto Index(Operon::Node n) {
        return std::countr_zero(static_cast<std::underlying_type_t<Operon::NodeType>>(n.Type));
    }

    [[nodiscard]] auto ComputeHash(Operon::Node const& n, int j) const {
        auto i = Index(n); 
        auto h = zobrist_(i, j);
        if (n.IsVariable()) { h ^= n.HashValue; }
        return h;
    }
      
    [[nodiscard]] auto ComputeHash(Operon::Tree const& tree) const {
        Operon::Hash h{};    
        auto const& nodes = tree.Nodes();
        for (auto i = 0; i < std::ssize(nodes); ++i) {
            auto const& n = nodes[i];
            h ^= ComputeHash(n, i);
        }
        return h;
    }

    [[nodiscard]] auto ComputeHash(Operon::Tree const& tree, auto subtreeIndex) const {
        if (subtreeIndex == 0) {
            return ComputeHash(tree);
        }
        Operon::Hash h{};
        auto const& nodes = tree.Nodes();
        for (auto i : tree.Indices(subtreeIndex)) {
            auto const& n = nodes[i];
            h ^= ComputeHash(n, i);
        }
        return h;
    }
};
} // namespace Operon

#endif
