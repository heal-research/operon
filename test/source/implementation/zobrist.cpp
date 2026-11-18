// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <latch>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "../operon_test.hpp"

#include "operon/core/dataset.hpp"
#include "operon/core/node.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/tree.hpp"
#include "operon/hash/zobrist.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/initializer.hpp"

namespace Operon::Test {

namespace {
    constexpr auto Seed = 42UL;
    constexpr auto MaxLength = 50;

    // The tests below that predate CacheVisibility pin the Immediate semantics (visible at once, first writer wins,
    // lazy expiry in TryGet); the Committed semantics have their own test cases further down.
    constexpr auto Immediate(std::size_t maxAge = 0) -> ZobristOptions
    {
        return { .MaxAge = maxAge, .Visibility = CacheVisibility::Immediate };
    }

    auto MakeSetup()
    {
        auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
        auto inputs = ds.VariableHashes();
        std::erase(inputs, ds.GetVariable("Y").value().Hash);
        PrimitiveSet pset;
        pset.SetConfig(PrimitiveSet::Arithmetic);
        return std::make_tuple(std::move(ds), std::move(inputs), std::move(pset));
    }
} // namespace

TEST_CASE("Zobrist - same tree yields same hash", "[zobrist]")
{
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist const cache(rng, MaxLength, inputs);

    BalancedTreeCreator const creator { &pset, inputs, /* bias= */ 0.0, MaxLength };
    auto tree = creator(rng, 20, 1, MaxLength);

    auto h1 = cache.ComputeHash(tree);
    auto h2 = cache.ComputeHash(tree);
    REQUIRE(h1 == h2);
}

TEST_CASE("Zobrist - different coefficients yield same hash", "[zobrist]")
{
    // The hash must be coefficient-insensitive so cached fitness (post local
    // search) can be reused for any structurally identical tree.
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist const cache(rng, MaxLength, inputs);

    BalancedTreeCreator const creator { &pset, inputs, /* bias= */ 0.0, MaxLength };
    Operon::NormalCoefficientInitializer const coeffInit;

    auto tree1 = creator(rng, 20, 1, MaxLength);
    auto tree2 = tree1; // identical structure

    coeffInit(rng, tree1);
    coeffInit(rng, tree2); // different coefficients

    REQUIRE(cache.ComputeHash(tree1) == cache.ComputeHash(tree2));
}

TEST_CASE("Zobrist - position sensitivity (deterministic)", "[zobrist]")
{
    // sin(cos(c)) and cos(sin(c)) have the same node types but at different
    // positions — the position-aware hash must distinguish them.
    Operon::RandomGenerator rng(Seed);
    Zobrist const cache(rng, MaxLength, {}); // no variables needed: trees use only constants

    // postfix: [Constant, Cos, Sin]  =>  sin(cos(c))
    Tree const tree1
        = Tree({ Node(NodeType::Constant), Util::MakeOp<BuiltinOp::Cos>(), Util::MakeOp<BuiltinOp::Sin>() })
              .UpdateNodes();
    // postfix: [Constant, Sin, Cos]  =>  cos(sin(c))
    Tree const tree2
        = Tree({ Node(NodeType::Constant), Util::MakeOp<BuiltinOp::Sin>(), Util::MakeOp<BuiltinOp::Cos>() })
              .UpdateNodes();

    REQUIRE(cache.ComputeHash(tree1) != cache.ComputeHash(tree2));
}

TEST_CASE("Zobrist - commuted variables yield different hashes", "[zobrist]")
{
    // Add(X, Y) and Add(Y, X) must hash differently so the JIT cache never
    // applies a compiled function to a tree with swapped variable columns.
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist const cache(rng, MaxLength, inputs);

    auto const varX = ds.GetVariable("X1").value();
    auto const varY = ds.GetVariable("X2").value();

    Node nX(NodeType::Variable);
    nX.HashValue = varX.Hash;
    nX.IsEnabled = true;
    Node nY(NodeType::Variable);
    nY.HashValue = varY.Hash;
    nY.IsEnabled = true;

    // postfix: [X, Y, Add] => Add(X, Y)
    Tree const treeXY = Tree({ nX, nY, Util::MakeOp<BuiltinOp::Add>() }).UpdateNodes();
    // postfix: [Y, X, Add] => Add(Y, X)
    Tree const treeYX = Tree({ nY, nX, Util::MakeOp<BuiltinOp::Add>() }).UpdateNodes();

    REQUIRE(cache.ComputeHash(treeXY) != cache.ComputeHash(treeYX));
}

TEST_CASE("Zobrist - reference targets affect the hash", "[zobrist]")
{
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist const cache(rng, MaxLength, inputs);

    auto const x1 = ds.GetVariable("X1").value().Hash;
    auto const x2 = ds.GetVariable("X2").value().Hash;
    auto x = Node(NodeType::Variable, x1);
    auto y = Node(NodeType::Variable, x2);
    auto add = Node::Function(static_cast<Hash>(BuiltinOp::Add), 2);
    auto const refX = Tree({ x, y, Node::Ref(0), add }).UpdateNodes();
    auto const refY = Tree({ x, y, Node::Ref(1), add }).UpdateNodes();

    CHECK(cache.ComputeHash(refX) != cache.ComputeHash(refY));
}

TEST_CASE("Zobrist - TryGet returns false on miss", "[zobrist]")
{
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist const cache(rng, MaxLength, inputs, Immediate());

    BalancedTreeCreator const creator { &pset, inputs, /* bias= */ 0.0, MaxLength };
    auto tree = creator(rng, 10, 1, MaxLength);
    auto hash = cache.ComputeHash(tree);

    Operon::Vector<Operon::Scalar> val;
    REQUIRE_FALSE(cache.TryGet(hash, val));
    REQUIRE(cache.Hits() == 0);
    REQUIRE(cache.Lookups() == 1);
}

TEST_CASE("Zobrist - Insert then TryGet roundtrip", "[zobrist]")
{
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist cache(rng, MaxLength, inputs, Immediate());

    BalancedTreeCreator const creator { &pset, inputs, /* bias= */ 0.0, MaxLength };
    auto tree = creator(rng, 10, 1, MaxLength);
    auto hash = cache.ComputeHash(tree);

    Operon::Vector<Operon::Scalar> const stored = { Operon::Scalar { 0.5 }, Operon::Scalar { 1.0 } };
    cache.Insert(hash, stored);
    REQUIRE(cache.Size() == 1);

    Operon::Vector<Operon::Scalar> retrieved(2);
    REQUIRE(cache.TryGet(hash, retrieved));
    REQUIRE(cache.Hits() == 1);
    REQUIRE(cache.Lookups() == 1);
    REQUIRE(retrieved[0] == stored[0]);
    REQUIRE(retrieved[1] == stored[1]);
}

TEST_CASE("Zobrist - Clear resets table and hit counter", "[zobrist]")
{
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist cache(rng, MaxLength, inputs, Immediate());

    BalancedTreeCreator const creator { &pset, inputs, /* bias= */ 0.0, MaxLength };
    auto tree = creator(rng, 10, 1, MaxLength);
    auto hash = cache.ComputeHash(tree);

    Operon::Vector<Operon::Scalar> const val = { Operon::Scalar { 0.1 } };
    cache.Insert(hash, val);

    Operon::Vector<Operon::Scalar> tmp;
    std::ignore = cache.TryGet(hash, tmp); // bumps hits

    cache.Clear();
    REQUIRE(cache.Size() == 0);
    REQUIRE(cache.Hits() == 0);
    REQUIRE(cache.Lookups() == 0);

    REQUIRE_FALSE(cache.TryGet(hash, tmp));
}

TEST_CASE("Zobrist - Lookups counts every TryGet call regardless of outcome", "[zobrist]")
{
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist cache(rng, MaxLength, inputs, Immediate());

    BalancedTreeCreator const creator { &pset, inputs, /* bias= */ 0.0, MaxLength };
    auto tree1 = creator(rng, 10, 1, MaxLength);
    auto tree2 = creator(rng, 10, 1, MaxLength);
    auto hash1 = cache.ComputeHash(tree1);
    auto hash2 = cache.ComputeHash(tree2);

    Operon::Vector<Operon::Scalar> const val = { Operon::Scalar { 0.1 } };
    cache.Insert(hash1, val);

    Operon::Vector<Operon::Scalar> tmp;
    std::ignore = cache.TryGet(hash1, tmp); // hit
    std::ignore = cache.TryGet(hash2, tmp); // miss (unless hash1==hash2, vanishingly unlikely here)
    std::ignore = cache.TryGet(hash1, tmp); // hit

    REQUIRE(cache.Lookups() == 3);
    REQUIRE(cache.Hits() <= cache.Lookups());
    REQUIRE(cache.Hits() >= 2);
}

TEST_CASE("Zobrist - duplicate inserts collapse to one entry", "[zobrist]")
{
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist cache(rng, MaxLength, inputs, Immediate());

    BalancedTreeCreator const creator { &pset, inputs, /* bias= */ 0.0, MaxLength };
    auto tree = creator(rng, 10, 1, MaxLength);
    auto hash = cache.ComputeHash(tree);

    Operon::Vector<Operon::Scalar> const val = { Operon::Scalar { 0.3 } };
    cache.Insert(hash, val);
    cache.Insert(hash, val);
    cache.Insert(hash, val);

    REQUIRE(cache.Size() == 1); // only one entry
}

TEST_CASE("Zobrist - distinct built-in ops at the same position yield distinct hashes", "[zobrist]")
{
    // Regression test: ComputeHash used to index a table row by
    // NodeTypes::GetIndex(n.Type) for every non-variable node, so this
    // already held for distinct built-ins (they have distinct NodeType
    // values) - this test pins that property now that non-variable
    // identity is combined from n.HashValue instead of table-indexed.
    Operon::RandomGenerator rng(Seed);
    Zobrist const cache(rng, MaxLength, {});

    REQUIRE(
        cache.ComputeHash(Util::MakeOp<BuiltinOp::Add>(), 0) != cache.ComputeHash(Util::MakeOp<BuiltinOp::Mul>(), 0));
    REQUIRE(
        cache.ComputeHash(Util::MakeOp<BuiltinOp::Sin>(), 0) != cache.ComputeHash(Util::MakeOp<BuiltinOp::Cos>(), 0));
}

TEST_CASE("Zobrist - distinct Dynamic-hash user functions yield distinct hashes", "[zobrist]")
{
    // Regression test for the actual bug this fix addresses: previously
    // ALL NodeType::Dynamic nodes shared one table row via
    // NodeTypes::GetIndex(NodeType::Dynamic), regardless of HashValue -
    // meaning two different user-registered functions collided onto the
    // same Zobrist identity and could cause a cached fitness value to be
    // wrongly reused for a structurally different tree using a different
    // custom function at the same position.
    Operon::RandomGenerator rng(Seed);
    Zobrist const cache(rng, MaxLength, {});

    Operon::Hash const hashA = Operon::Hasher {}("userFunctionA");
    Operon::Hash const hashB = Operon::Hasher {}("userFunctionB");
    REQUIRE(hashA != hashB);

    Node const nodeA(NodeType::Function, hashA);
    Node const nodeB(NodeType::Function, hashB);

    REQUIRE(cache.ComputeHash(nodeA, 0) != cache.ComputeHash(nodeB, 0));

    // Also distinct from a built-in occupying the same tree position.
    REQUIRE(cache.ComputeHash(nodeA, 0) != cache.ComputeHash(Util::MakeOp<BuiltinOp::Add>(), 0));

    // Same Dynamic hash at two different positions must also differ - pins
    // position-sensitivity for the non-table combine path specifically,
    // not just (as the pre-existing "position sensitivity" test covers)
    // for built-ins via full-tree XOR.
    REQUIRE(cache.ComputeHash(nodeA, 0) != cache.ComputeHash(nodeA, 1));
}

TEST_CASE("Zobrist - different Optimize flags yield different hashes", "[zobrist]")
{
    Operon::RandomGenerator rng(Seed);
    Zobrist const cache(rng, MaxLength, {});

    // Two identical trees: sin(constant)
    // One has the constant optimizable, the other does not.
    Node c1(NodeType::Constant);
    c1.Value = 1.0f;
    c1.Optimize = true;
    Node c2(NodeType::Constant);
    c2.Value = 1.0f;
    c2.Optimize = false;

    Tree const tree1 = Tree({ c1, Util::MakeOp<BuiltinOp::Sin>() }).UpdateNodes();
    Tree const tree2 = Tree({ c2, Util::MakeOp<BuiltinOp::Sin>() }).UpdateNodes();

    REQUIRE(cache.ComputeHash(tree1) != cache.ComputeHash(tree2));
}

TEST_CASE("Zobrist - maxAge disabled (default) never expires entries", "[zobrist]")
{
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist cache(rng, MaxLength, inputs, Immediate()); // maxAge 0 = disabled

    BalancedTreeCreator const creator { &pset, inputs, /* bias= */ 0.0, MaxLength };
    auto tree = creator(rng, 10, 1, MaxLength);
    auto hash = cache.ComputeHash(tree);

    Operon::Vector<Operon::Scalar> const val = { Operon::Scalar { 0.1 } };
    cache.Insert(hash, val);

    // advance the clock far beyond any plausible age and confirm the entry
    // is still considered fresh, since maxAge == 0 disables expiry entirely.
    cache.SetGeneration(1'000'000);

    Operon::Vector<Operon::Scalar> retrieved;
    REQUIRE(cache.TryGet(hash, retrieved));
    REQUIRE(cache.Size() == 1);
}

TEST_CASE("Zobrist - entry older than maxAge is treated as a miss", "[zobrist]")
{
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    constexpr std::size_t maxAge = 5;
    Zobrist cache(rng, MaxLength, inputs, Immediate(maxAge));

    BalancedTreeCreator const creator { &pset, inputs, /* bias= */ 0.0, MaxLength };
    auto tree = creator(rng, 10, 1, MaxLength);
    auto hash = cache.ComputeHash(tree);

    cache.SetGeneration(0);
    Operon::Vector<Operon::Scalar> const val = { Operon::Scalar { 0.1 } };
    cache.Insert(hash, val); // stamped with generation 0

    cache.SetGeneration(maxAge + 1); // strictly older than maxAge

    Operon::Vector<Operon::Scalar> retrieved;
    REQUIRE_FALSE(cache.TryGet(hash, retrieved));
}

TEST_CASE("Zobrist - a stale-triggered miss actually removes the entry", "[zobrist]")
{
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    constexpr std::size_t maxAge = 5;
    Zobrist cache(rng, MaxLength, inputs, Immediate(maxAge));

    BalancedTreeCreator const creator { &pset, inputs, /* bias= */ 0.0, MaxLength };
    auto tree = creator(rng, 10, 1, MaxLength);
    auto hash = cache.ComputeHash(tree);

    cache.SetGeneration(0);
    Operon::Vector<Operon::Scalar> const val = { Operon::Scalar { 0.1 } };
    cache.Insert(hash, val);
    REQUIRE(cache.Size() == 1);

    cache.SetGeneration(maxAge + 1);

    Operon::Vector<Operon::Scalar> retrieved;
    REQUIRE_FALSE(cache.TryGet(hash, retrieved)); // triggers the expiry-erase
    REQUIRE(cache.Size() == 0);

    // A subsequent lookup is a plain miss against an empty cache, not a
    // repeated expiry - Lookups() should reflect two calls, no more erasing.
    REQUIRE_FALSE(cache.TryGet(hash, retrieved));
    REQUIRE(cache.Size() == 0);
}

TEST_CASE("Zobrist - an entry inserted after the clock is set to a high starting value (warm resume) is not "
          "immediately stale",
    "[zobrist]")
{
    // This verifies the Zobrist-level contract that CLI warm-resume relies
    // on (SetGeneration() before re-caching avoids the immediate-eviction
    // bug described below) - it does NOT regression-test the CLI wiring
    // itself. That fix lives in cli/source/util.cpp's ResumeFromCheckpoint
    // (calls cache->SetGeneration(cp->Generation) right after restoring
    // algo.Generation(), before the resumed population is re-evaluated);
    // cli/source/util.cpp isn't linked into this test binary (see
    // test/CMakeLists.txt), so that call site has no automated regression
    // coverage - only this narrower API-level guarantee does.
    //
    // Background: on --resume, GeneticAlgorithmBase::Generation() is
    // restored from the checkpoint, but a freshly constructed Zobrist always
    // starts at clock_ == 0. Without the fix above, resumed entries get
    // stamped generation 0 and read as ancient the moment the GA loop
    // advances the clock past the checkpoint's generation.
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    constexpr std::size_t maxAge = 5;
    Zobrist cache(rng, MaxLength, inputs, Immediate(maxAge));

    BalancedTreeCreator const creator { &pset, inputs, /* bias= */ 0.0, MaxLength };
    auto tree = creator(rng, 10, 1, MaxLength);
    auto hash = cache.ComputeHash(tree);

    constexpr std::size_t resumedGeneration = 500;
    cache.SetGeneration(resumedGeneration); // as the fixed resume path does, before re-evaluation
    Operon::Vector<Operon::Scalar> const val = { Operon::Scalar { 0.1 } };
    cache.Insert(hash, val);

    // The GA loop's own SetGeneration(Generation() + 1) call follows next.
    cache.SetGeneration(resumedGeneration + 1);

    Operon::Vector<Operon::Scalar> retrieved;
    REQUIRE(cache.TryGet(hash, retrieved));
}

TEST_CASE("Zobrist - Clear resets the generation clock", "[zobrist]")
{
    // Regression test: without resetting clock_, a Clear() between separate
    // runs (e.g. sequential invocations sharing one Zobrist instance) leaves
    // a stale-relative-to-nothing clock value; the next run's entries get
    // stamped against generation 0 while the clock remains at the old run's
    // high value, immediately reading as ancient.
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    constexpr std::size_t maxAge = 5;
    Zobrist cache(rng, MaxLength, inputs, Immediate(maxAge));

    BalancedTreeCreator const creator { &pset, inputs, /* bias= */ 0.0, MaxLength };
    auto tree = creator(rng, 10, 1, MaxLength);
    auto hash = cache.ComputeHash(tree);

    cache.SetGeneration(500);
    Operon::Vector<Operon::Scalar> const val = { Operon::Scalar { 0.1 } };
    cache.Insert(hash, val);

    cache.Clear();
    REQUIRE(cache.Size() == 0);

    // A new "run" starts its own generation count from 0. Deliberately do
    // NOT call SetGeneration(0) here - the whole point is to check that
    // Clear() itself already reset the clock; if it didn't, this Insert
    // would be stamped 0 while clock_ is still 500, and TryGet below would
    // immediately (falsely) evict it as 500 generations stale.
    cache.Insert(hash, val);

    Operon::Vector<Operon::Scalar> retrieved;
    REQUIRE(cache.TryGet(hash, retrieved));
}

TEST_CASE("Zobrist - EraseIf generation guard preserves a value refreshed between observation and erase", "[zobrist]")
{
    // Deterministic, single-threaded repro of the exact race TryGet's
    // conditional EraseIf guards against: thread A observes a stale entry
    // and records its InsertGeneration, but before A's erase runs, another
    // thread erases that same entry and a third thread recreates it fresh
    // (new InsertGeneration). A's erase, guarded by the generation it
    // originally observed, must not remove the fresh entry that replaced it.
    ZobristCache<FitnessEntry> cache;
    Operon::Hash const hash = Operon::Hasher {}("erase-guard-repro");

    cache.LazyEmplace(
        hash, [](FitnessEntry&) {},
        [](FitnessEntry& e) {
            e.Value = { Operon::Scalar { 0.1 } };
            e.InsertGeneration = 5;
        });

    // Thread A "observes" the entry while it's still stale at generation 5.
    constexpr std::uint32_t observedGen = 5;

    // Before A's erase runs: another thread erases the entry, then a third
    // thread inserts a fresh one for the same hash at a later generation.
    cache.EraseIf(hash, [](FitnessEntry const&) { return true; });
    cache.LazyEmplace(
        hash, [](FitnessEntry&) {},
        [](FitnessEntry& e) {
            e.Value = { Operon::Scalar { 0.9 } };
            e.InsertGeneration = 6;
        });

    // A's delayed erase, guarded by the generation it originally observed -
    // this must be a no-op now that the entry has moved on to generation 6.
    cache.EraseIf(hash, [&](FitnessEntry const& e) { return e.InsertGeneration == observedGen; });

    REQUIRE(cache.Size() == 1);
    bool found = false;
    cache.IfContains(hash, [&](FitnessEntry const& e) {
        found = true;
        REQUIRE(e.InsertGeneration == 6);
        REQUIRE(e.Value[0] == Operon::Scalar { 0.9 });
    });
    REQUIRE(found);
}

TEST_CASE("Zobrist - concurrent readers racing an expiry-erase-then-reinsert never lose a fresh value", "[zobrist]")
{
    // Two reader threads (not one) are needed to exercise the guard's actual
    // race under real scheduling: with a single reader, that reader is the
    // only thread that can ever erase an entry, so "another thread erased it
    // first, then it got recreated fresh before my erase ran" can never
    // happen - see the deterministic "EraseIf generation guard..." test
    // above for a targeted, guaranteed repro of that exact interleaving.
    // This test additionally stresses it under real thread scheduling.
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    constexpr std::size_t maxAge = 1;
    Zobrist cache(rng, MaxLength, inputs, Immediate(maxAge));

    BalancedTreeCreator const creator { &pset, inputs, /* bias= */ 0.0, MaxLength };
    auto tree = creator(rng, 10, 1, MaxLength);
    auto hash = cache.ComputeHash(tree);

    cache.SetGeneration(0);
    Operon::Vector<Operon::Scalar> const initial = { Operon::Scalar { 0.1 } };
    cache.Insert(hash, initial);

    std::atomic<bool> stop { false };
    Operon::Vector<Operon::Scalar> const fresh = { Operon::Scalar { 0.9 } };

    // Advance generations and keep re-inserting a fresh value for the same
    // hash, racing against two reader threads that will observe stale
    // entries and attempt to erase them - with two readers, one can erase
    // an entry the other already observed as stale, before the other's own
    // erase runs.
    std::thread writer([&]() {
        for (std::size_t g = 1; g < 2000; ++g) {
            cache.SetGeneration(g);
            cache.Insert(hash, fresh); // no-op if entry already fresh (LazyEmplace keeps existing)
            if (stop.load(std::memory_order_relaxed)) {
                break;
            }
        }
    });

    auto readerLoop = [&]() {
        Operon::Vector<Operon::Scalar> tmp;
        for (int i = 0; i < 2000; ++i) {
            std::ignore = cache.TryGet(hash, tmp); // may observe stale + erase; must not crash
        }
    };
    std::thread reader1(readerLoop);
    std::thread reader2([&]() {
        readerLoop();
        stop.store(true, std::memory_order_relaxed);
    });

    writer.join();
    reader1.join();
    reader2.join();

    // Whatever entry survived the race (if any) may legitimately be stale
    // relative to wherever the clock landed - Insert() only stamps a fresh
    // InsertGeneration when the entry didn't already exist (see the
    // "first writer wins" comment on Insert()), so a leftover stale survivor
    // would make a plain Insert()-then-TryGet flaky here, not indicative of
    // a real bug. Clear() first (deterministically resets both the map and
    // the clock) so the final check only asserts what it's meant to: a
    // fresh Insert() after the race is never lost.
    cache.Clear();
    cache.SetGeneration(0);
    cache.Insert(hash, fresh);
    Operon::Vector<Operon::Scalar> final;
    REQUIRE(cache.TryGet(hash, final));
}

// ── CacheVisibility ──────────────────────────────────────────────────────────────────────────────────────────────

namespace {
    using Fitness = Operon::Vector<Operon::Scalar>;

    auto Committed(std::size_t maxAge = 0) -> ZobristOptions
    {
        return { .MaxAge = maxAge, .Visibility = CacheVisibility::Committed };
    }

    auto MakeCache(ZobristOptions options) -> std::unique_ptr<Zobrist>
    {
        auto [ds, inputs, pset] = MakeSetup();
        Operon::RandomGenerator rng(Seed);
        return std::make_unique<Zobrist>(rng, MaxLength, inputs, options);
    }

    constexpr Operon::Hash H1 { 0x1111 };
    constexpr Operon::Hash H2 { 0x2222 };
    constexpr Operon::Hash H3 { 0x3333 };

    // Runs `fn` on a thread of its own, the way a worker runs a slot, and waits for it.
    template <typename F> void OnOtherThread(F&& fn)
    {
        std::thread t { std::forward<F>(fn) };
        t.join();
    }

    auto Get(Zobrist const& cache, Operon::Hash h) -> std::optional<Fitness>
    {
        Fitness out(1);
        if (cache.TryGet(h, out)) {
            return out;
        }
        return std::nullopt;
    }

    // Lookup from inside a slot of the cache, the way the GA loops look up.
    auto GetInSlot(Zobrist const& cache, Operon::Hash h) -> std::optional<Fitness>
    {
        auto const slot = cache.BeginSlot();
        return Get(cache, h);
    }
} // namespace

TEST_CASE("ZobristOptions - defaults are unbounded age and Committed visibility", "[zobrist][visibility]")
{
    ZobristOptions const options;
    CHECK(options.MaxAge == 0);
    CHECK(options.Visibility == CacheVisibility::Committed);

    auto const cache = MakeCache({});
    CHECK(cache->MaxAge() == 0);
    CHECK(cache->Visibility() == CacheVisibility::Committed);

    auto const immediate = MakeCache(Immediate(3));
    CHECK(immediate->MaxAge() == 3);
    CHECK(immediate->Visibility() == CacheVisibility::Immediate);
}

TEST_CASE("Zobrist Committed - an entry of another slot in the same generation is a miss, visible in the next",
    "[zobrist][visibility]")
{
    auto const cache = MakeCache(Committed());
    cache->SetGeneration(1);

    OnOtherThread([&] {
        auto const slot = cache->BeginSlot();
        cache->Insert(H1, { Operon::Scalar { 7 } });
    });
    REQUIRE(cache->Size() == 1);

    SECTION("a slot of the same generation misses")
    {
        auto const slot = cache->BeginSlot();
        CHECK_FALSE(Get(*cache, H1));
        CHECK(cache->Hits() == 0);
        CHECK(cache->Lookups() == 1);
    }

    SECTION("a lookup outside any slot sees it") { CHECK(Get(*cache, H1)); }

    SECTION("the next generation hits")
    {
        cache->SetGeneration(2);
        auto const hit = Get(*cache, H1);
        REQUIRE(hit);
        CHECK((*hit)[0] == Operon::Scalar { 7 });
        CHECK(cache->Hits() == 1);
    }
}

TEST_CASE("Zobrist Committed - a slot sees its own entries immediately, no other slot does", "[zobrist][visibility]")
{
    auto const cache = MakeCache(Committed());
    cache->SetGeneration(1);

    {
        auto const slot = cache->BeginSlot();
        cache->Insert(H1, { Operon::Scalar { 3 } });
        auto const own = Get(*cache, H1);
        REQUIRE(own);
        CHECK((*own)[0] == Operon::Scalar { 3 });
        CHECK_FALSE(Get(*cache, H2));
    }

    SECTION("once the slot is closed its entries are invisible to a new slot of the same generation")
    {
        CHECK_FALSE(GetInSlot(*cache, H1)); // a new slot does not inherit the previous one's entries
    }

    SECTION("a slot on another thread does not see them either")
    {
        bool seen = true;
        OnOtherThread([&] {
            auto const slot = cache->BeginSlot();
            seen = Get(*cache, H1).has_value();
        });
        CHECK_FALSE(seen);
    }

    SECTION("the entries are in the shared table from the next generation on")
    {
        cache->SetGeneration(2);
        CHECK(Get(*cache, H1));
    }
}

TEST_CASE("Zobrist Committed - a slot keeps its own value even when another slot's smaller one wins the entry",
    "[zobrist][visibility]")
{
    auto const cache = MakeCache(Committed());
    cache->SetGeneration(1);
    {
        auto const slot = cache->BeginSlot();
        cache->Insert(H1, { Operon::Scalar { 5 } });
        OnOtherThread([&] {
            auto const other = cache->BeginSlot();
            cache->Insert(H1, { Operon::Scalar { 2 } });
        });
        auto const own = Get(*cache, H1);
        REQUIRE(own);
        CHECK((*own)[0] == Operon::Scalar { 5 });
    }
    cache->SetGeneration(2);
    auto const shared = Get(*cache, H1);
    REQUIRE(shared);
    CHECK((*shared)[0] == Operon::Scalar { 2 });
}

TEST_CASE("Zobrist Committed - duplicate inserts keep the smaller fitness in either order", "[zobrist][visibility]")
{
    auto smallest = [](std::vector<Fitness> const& inserts) {
        auto const cache = MakeCache(Committed());
        cache->SetGeneration(1);
        for (auto const& v : inserts) {
            cache->Insert(H1, v);
        }
        CHECK(cache->Size() == 1);
        cache->SetGeneration(2);
        Fitness out(inserts.front().size());
        REQUIRE(cache->TryGet(H1, out));
        return out;
    };

    SECTION("lexicographic over the components")
    {
        Fitness const a { Operon::Scalar { 1 }, Operon::Scalar { 9 } };
        Fitness const b { Operon::Scalar { 2 }, Operon::Scalar { 0 } };
        Fitness const c { Operon::Scalar { 1 }, Operon::Scalar { 5 } };
        CHECK(smallest({ a, b, c }) == c);
        CHECK(smallest({ c, b, a }) == c);
        CHECK(smallest({ b, a, c }) == c);
        CHECK(smallest({ b, c, a }) == c);
    }

    SECTION("negative zero orders before positive zero")
    {
        Fitness const pos { Operon::Scalar { 0 } };
        Fitness const neg { -Operon::Scalar { 0 } };
        REQUIRE(std::signbit(neg[0]));
        CHECK(std::signbit(smallest({ pos, neg })[0]));
        CHECK(std::signbit(smallest({ neg, pos })[0]));
    }

    SECTION("NaN orders by sign and after every number")
    {
        auto const nan = std::numeric_limits<Operon::Scalar>::quiet_NaN();
        Fitness const number { Operon::Scalar { 1e30 } };
        Fitness const positiveNan { std::copysign(nan, Operon::Scalar { 1 }) };
        Fitness const negativeNan { std::copysign(nan, Operon::Scalar { -1 }) };

        CHECK(smallest({ positiveNan, number })[0] == number[0]);
        CHECK(smallest({ number, positiveNan })[0] == number[0]);

        auto const lowest = smallest({ number, positiveNan, negativeNan });
        CHECK(std::isnan(lowest[0]));
        CHECK(std::signbit(lowest[0])); // -NaN is below every number
        auto const reversed = smallest({ negativeNan, positiveNan, number });
        CHECK(std::isnan(reversed[0]));
        CHECK(std::signbit(reversed[0]));
    }
}

TEST_CASE("Zobrist Committed - an entry of an earlier generation is never overwritten", "[zobrist][visibility]")
{
    auto const cache = MakeCache(Committed());
    cache->SetGeneration(1);
    cache->Insert(H1, { Operon::Scalar { 5 } });
    cache->SetGeneration(2);
    cache->Insert(H1, { Operon::Scalar { 1 } }); // smaller, but the entry is already visible to lookups

    auto const kept = Get(*cache, H1);
    REQUIRE(kept);
    CHECK((*kept)[0] == Operon::Scalar { 5 });
}

TEST_CASE("Zobrist Committed - generation 0 entries are hidden from slots until the clock moves past 0",
    "[zobrist][visibility]")
{
    // The GA loop stamps offspring with Generation() + 1 (>= 1); the initial population is scored without the cache.
    // A cache that was never given a clock value stamps 0, and the rule is the same: visible once the clock differs.
    auto const cache = MakeCache(Committed());
    cache->Insert(H1, { Operon::Scalar { 4 } });
    CHECK_FALSE(GetInSlot(*cache, H1));
    {
        auto const slot = cache->BeginSlot();
        cache->Insert(H2, { Operon::Scalar { 6 } });
        CHECK(Get(*cache, H2));
    }
    CHECK_FALSE(GetInSlot(*cache, H2));

    cache->SetGeneration(1);
    CHECK(GetInSlot(*cache, H1));
    CHECK(GetInSlot(*cache, H2));

    cache->Clear(); // resets the clock: the next run starts at generation 0 again
    cache->Insert(H1, { Operon::Scalar { 4 } });
    CHECK_FALSE(GetInSlot(*cache, H1));
    cache->SetGeneration(1);
    CHECK(GetInSlot(*cache, H1));
}

TEST_CASE("Zobrist Committed - a default cache used without any slot hits entries of the same clock value",
    "[zobrist][visibility]")
{
    // Code outside the GA loops has neither a clock nor slots: lookups must not be hidden from it.
    auto const cache = MakeCache({});
    REQUIRE(cache->Visibility() == CacheVisibility::Committed);

    cache->Insert(H1, { Operon::Scalar { 3 } });
    auto const first = Get(*cache, H1);
    REQUIRE(first);
    CHECK((*first)[0] == Operon::Scalar { 3 });
    CHECK(cache->Hits() == 1);

    cache->SetGeneration(1);
    cache->Insert(H2, { Operon::Scalar { 5 } });
    CHECK(Get(*cache, H2));
    CHECK_FALSE(GetInSlot(*cache, H2)); // a slot of the GA loop still does not see its generation's entries

    cache->Insert(H2, { Operon::Scalar { 2 } }); // duplicate inserts of one clock value keep the smaller
    auto const smaller = Get(*cache, H2);
    REQUIRE(smaller);
    CHECK((*smaller)[0] == Operon::Scalar { 2 });
}

TEST_CASE("Zobrist Immediate - first writer wins and entries are visible at once", "[zobrist][visibility]")
{
    auto const cache = MakeCache(Immediate());
    cache->SetGeneration(1);

    OnOtherThread([&] {
        auto const slot = cache->BeginSlot(); // inert: Immediate has no slot-local state
        cache->Insert(H1, { Operon::Scalar { 2 } });
    });
    cache->Insert(H1, { Operon::Scalar { 1 } }); // smaller, but the first writer keeps the entry

    auto const hit = Get(*cache, H1);
    REQUIRE(hit);
    CHECK((*hit)[0] == Operon::Scalar { 2 });
    CHECK(cache->Size() == 1);
}

TEST_CASE("Zobrist Committed - maxAge expires entries at the generation boundary", "[zobrist][visibility]")
{
    constexpr std::size_t maxAge = 2;
    auto const cache = MakeCache(Committed(maxAge));
    cache->SetGeneration(1);
    cache->Insert(H1, { Operon::Scalar { 1 } }); // stamped 1
    cache->SetGeneration(2);
    cache->Insert(H2, { Operon::Scalar { 2 } }); // stamped 2
    REQUIRE(cache->Size() == 2);

    cache->SetGeneration(3); // ages 2 and 1: nothing is older than maxAge
    CHECK(cache->Size() == 2);
    cache->SetGeneration(4); // H1 is 3 generations old
    CHECK(cache->Size() == 1); // gone before any lookup touched it
    CHECK_FALSE(Get(*cache, H1));
    CHECK(Get(*cache, H2));
    cache->SetGeneration(5);
    CHECK(cache->Size() == 0);
}

TEST_CASE(
    "Zobrist Immediate - maxAge expires lazily inside TryGet, not at the generation boundary", "[zobrist][visibility]")
{
    auto const cache = MakeCache(Immediate(2));
    cache->SetGeneration(1);
    cache->Insert(H1, { Operon::Scalar { 1 } });
    cache->SetGeneration(4);
    CHECK(cache->Size() == 1);
    CHECK_FALSE(Get(*cache, H1));
    CHECK(cache->Size() == 0);
}

TEST_CASE("Zobrist Committed - slot state belongs to the cache instance", "[zobrist][visibility]")
{
    auto const a = MakeCache(Committed());
    auto const b = MakeCache(Committed());
    a->SetGeneration(1);
    b->SetGeneration(1);

    auto const slotA = a->BeginSlot();
    a->Insert(H1, { Operon::Scalar { 1 } });
    CHECK(Get(*a, H1));
    CHECK_FALSE(Get(*b, H1)); // a's open slot is not b's

    {
        auto const slotB = b->BeginSlot(); // nested on the same thread
        b->Insert(H2, { Operon::Scalar { 2 } });
        CHECK(Get(*b, H2));
        CHECK_FALSE(Get(*b, H1));
        CHECK_FALSE(Get(*a, H2));
        CHECK(Get(*a, H1)); // a's slot is still reachable underneath b's
    }
    {
        auto const again = b->BeginSlot(); // a new scope does not inherit b's closed one
        CHECK_FALSE(Get(*b, H2));
    }
    CHECK(Get(*a, H1));

    SECTION("an Insert on b outside any b slot is not slot-local")
    {
        b->Insert(H2, { Operon::Scalar { 2 } });
        CHECK(Get(*b, H2)); // no b slot is open: the shared table is consulted as a whole
        CHECK_FALSE(GetInSlot(*b, H2));
        CHECK_FALSE(Get(*a, H2));
    }
}

TEST_CASE("Zobrist Committed - concurrent slots on one cache do not see each other's entries", "[zobrist][visibility]")
{
    // Two runs sharing a cache open slots with the same slot indices at the same time; the state is per scope.
    auto const cache = MakeCache(Committed());
    cache->SetGeneration(1);

    std::latch bothInserted { 2 };
    std::atomic<int> sawOther { 0 };
    std::atomic<int> sawOwn { 0 };
    auto worker = [&](Operon::Hash mine, Operon::Hash theirs, Operon::Scalar value) {
        auto const slot = cache->BeginSlot();
        cache->Insert(mine, { value });
        bothInserted.arrive_and_wait();
        if (Get(*cache, theirs)) {
            sawOther.fetch_add(1);
        }
        if (auto const own = Get(*cache, mine); own && (*own)[0] == value) {
            sawOwn.fetch_add(1);
        }
    };
    std::thread t1 { worker, H1, H2, Operon::Scalar { 1 } };
    std::thread t2 { worker, H2, H1, Operon::Scalar { 2 } };
    t1.join();
    t2.join();
    CHECK(sawOther.load() == 0);
    CHECK(sawOwn.load() == 2);
}

TEST_CASE("Zobrist - SlotScope without a cache is inert", "[zobrist][visibility]")
{
    Zobrist::SlotScope const none { nullptr };
    auto const cache = MakeCache(Committed());
    cache->SetGeneration(1);
    cache->Insert(H1, { Operon::Scalar { 1 } });
    CHECK(Get(*cache, H1)); // an inert scope is not a slot of any cache: nothing is hidden
}

TEST_CASE("Zobrist - a const SlotScope records its slot-local values", "[zobrist][visibility]")
{
    // The scope is an RAII guard and is naturally bound as const (as the GA loops do). Its slot-local map is reached
    // through the thread-local chain, so it has to be mutable: writing it through that chain while the scope object
    // is const would otherwise be undefined behaviour.
    auto const cache = MakeCache(Committed());
    cache->SetGeneration(1);

    Zobrist::SlotScope const scope { cache.get() };
    static_assert(std::is_const_v<decltype(scope)>);

    cache->Insert(H1, { Operon::Scalar { 3 } });
    // The shared entry carries the current clock value, so only the slot-local copy can serve this lookup.
    auto const own = Get(*cache, H1);
    REQUIRE(own);
    CHECK((*own)[0] == Operon::Scalar { 3 });

    // Updating the slot-local value through the const scope replaces the previous one.
    cache->Insert(H1, { Operon::Scalar { 2 } });
    CHECK((*Get(*cache, H1))[0] == Operon::Scalar { 2 });

    // A const scope from BeginSlot() behaves the same, including across a nested scope on the same cache.
    auto const nested = cache->BeginSlot();
    cache->Insert(H2, { Operon::Scalar { 5 } });
    CHECK((*Get(*cache, H2))[0] == Operon::Scalar { 5 });
}

TEST_CASE("Zobrist - Export/Import reproduces the entries, the clock and the expiry behaviour", "[zobrist][checkpoint]")
{
    for (auto const visibility : { CacheVisibility::Committed, CacheVisibility::Immediate }) {
        CAPTURE(visibility == CacheVisibility::Committed);
        auto const source = MakeCache({ .MaxAge = 3, .Visibility = visibility });
        // entries of several generations, some of them close to expiring
        for (std::uint32_t generation = 1; generation <= 5; ++generation) {
            source->SetGeneration(generation);
            for (std::uint32_t i = 0; i < 4; ++i) {
                source->Insert(Operon::Hash { generation * 100 + i },
                    { static_cast<Operon::Scalar>(generation), static_cast<Operon::Scalar>(i) });
            }
        }
        auto const state = source->Export();
        CHECK(state.Clock == 5);
        CHECK(state.MaxAge == 3);
        CHECK(state.Visibility == visibility);
        REQUIRE(state.Entries.size() == source->Size());
        CHECK(std::ranges::is_sorted(state.Entries, {}, &ZobristEntryRecord::Key));
        CHECK(source->Export().Entries.size() == state.Entries.size()); // deterministic

        auto const restored = MakeCache({ .MaxAge = 3, .Visibility = visibility });
        restored->Import(state, 2);
        CHECK(restored->Size() == source->Size());

        // From here on both caches see the same lookups, whatever the clock does (visibility within a generation and
        // age-based expiry as the clock advances).
        for (std::uint32_t clock = 5; clock <= 12; ++clock) {
            source->SetGeneration(clock);
            restored->SetGeneration(clock);
            CAPTURE(clock);
            for (std::uint32_t generation = 1; generation <= 5; ++generation) {
                for (std::uint32_t i = 0; i < 4; ++i) {
                    auto const key = Operon::Hash { generation * 100 + i };
                    CHECK(Get(*source, key) == Get(*restored, key));
                    CHECK(GetInSlot(*source, key) == GetInSlot(*restored, key));
                }
            }
            CHECK(restored->Size() == source->Size());
        }
        CHECK(restored->Export().Entries.size() == source->Export().Entries.size());
    }
}

TEST_CASE("Zobrist - Import replaces the previous content and keeps the counters", "[zobrist][checkpoint]")
{
    auto const source = MakeCache(Committed());
    source->SetGeneration(2);
    source->Insert(H1, { Operon::Scalar { 1 } });
    auto const target = MakeCache(Committed());
    target->Insert(H2, { Operon::Scalar { 9 } });
    std::ignore = Get(*target, H2);
    REQUIRE(target->Hits() == 1);

    target->Import(source->Export(), 1);
    CHECK(target->Size() == 1);
    CHECK(Get(*target, H1).has_value());
    CHECK_FALSE(Get(*target, H2).has_value());
    CHECK(target->Hits() == 2); // counters are not part of the state; they were 1 + the lookup of H1
}

TEST_CASE("Zobrist - Import rejects a state that this cache cannot continue", "[zobrist][checkpoint]")
{
    auto const source = MakeCache({ .MaxAge = 3, .Visibility = CacheVisibility::Committed });
    source->SetGeneration(2);
    source->Insert(H1, { Operon::Scalar { 1 } });
    auto const state = source->Export();

    SECTION("another random table (a different seed)")
    {
        auto [ds, inputs, pset] = MakeSetup();
        Operon::RandomGenerator rng(Seed + 1);
        Zobrist other { rng, MaxLength, inputs, { .MaxAge = 3 } };
        CHECK_THROWS_AS(other.Import(state, 1), std::invalid_argument);
        CHECK(other.Size() == 0);
    }
    SECTION("another maximum length")
    {
        auto [ds, inputs, pset] = MakeSetup();
        Operon::RandomGenerator rng(Seed);
        Zobrist other { rng, MaxLength + 1, inputs, { .MaxAge = 3 } };
        CHECK_THROWS_AS(other.Import(state, 1), std::invalid_argument);
    }
    SECTION("another visibility")
    {
        auto const other = MakeCache({ .MaxAge = 3, .Visibility = CacheVisibility::Immediate });
        CHECK_THROWS_AS(other->Import(state, 1), std::invalid_argument);
    }
    SECTION("another maximum age")
    {
        auto const other = MakeCache({ .MaxAge = 4, .Visibility = CacheVisibility::Committed });
        CHECK_THROWS_AS(other->Import(state, 1), std::invalid_argument);
    }
    SECTION("the same table, visibility and age is accepted")
    {
        auto const other = MakeCache({ .MaxAge = 3, .Visibility = CacheVisibility::Committed });
        CHECK_NOTHROW(other->Import(state, 1));
    }
}

TEST_CASE("Zobrist - Import validates the fitness length of every entry and leaves the cache unchanged on error",
    "[zobrist][checkpoint]")
{
    auto const source = MakeCache(Committed());
    source->SetGeneration(2);
    source->Insert(H1, { Operon::Scalar { 1 }, Operon::Scalar { 2 } });
    source->Insert(H2, { Operon::Scalar { 3 }, Operon::Scalar { 4 } });
    auto state = source->Export();

    auto const target = MakeCache(Committed());
    target->Insert(H3, { Operon::Scalar { 9 } });
    SECTION("a different objective count")
    {
        CHECK_THROWS_AS(target->Import(state, 1), std::invalid_argument);
        CHECK_THROWS_AS(target->Import(state, 3), std::invalid_argument);
    }
    SECTION("a zero objective count") { CHECK_THROWS_AS(target->Import(state, 0), std::invalid_argument); }
    SECTION("a zero-length entry")
    {
        state.Entries[1].Value.clear();
        CHECK_THROWS_AS(target->Import(state, 2), std::invalid_argument);
        CHECK_THROWS_AS(target->Import(state, 0), std::invalid_argument);
    }
    SECTION("one entry of another length")
    {
        state.Entries[1].Value.push_back(Operon::Scalar { 5 });
        CHECK_THROWS_AS(target->Import(state, 2), std::invalid_argument);
    }
    // the cache still holds what it held
    CHECK(target->Size() == 1);
    CHECK(Get(*target, H3).has_value());

    CHECK_NOTHROW(target->Import(source->Export(), 2));
    CHECK(target->Size() == 2);
}

TEST_CASE("Zobrist - Import rejects an entry inserted after the clock and leaves the cache unchanged",
    "[zobrist][checkpoint]")
{
    auto const source = MakeCache(Committed());
    source->SetGeneration(3);
    source->Insert(H1, { Operon::Scalar { 1 } });
    source->Insert(H2, { Operon::Scalar { 2 } });
    auto state = source->Export();
    REQUIRE(state.Entries.size() == 2);
    REQUIRE(state.Clock == 3);

    auto const target = MakeCache(Committed());
    target->SetGeneration(7);
    target->Insert(H3, { Operon::Scalar { 9 } });
    auto const before = target->Export();

    SECTION("the last entry is one generation ahead of the clock")
    {
        state.Entries.back().InsertGeneration = state.Clock + 1;
        CHECK_THROWS_AS(target->Import(state, 1), std::invalid_argument);
    }
    SECTION("the first entry is far ahead of the clock")
    {
        state.Entries.front().InsertGeneration = std::numeric_limits<std::uint32_t>::max();
        CHECK_THROWS_AS(target->Import(state, 1), std::invalid_argument);
    }
    SECTION("a state whose entries are stamped exactly at the clock is accepted")
    {
        for (auto& entry : state.Entries) {
            entry.InsertGeneration = state.Clock;
        }
        CHECK_NOTHROW(target->Import(state, 1));
        CHECK(target->Size() == 2);
        return;
    }

    // nothing changed: the entries, their stamps and the clock are as they were
    auto const after = target->Export();
    CHECK(after.Clock == before.Clock);
    REQUIRE(after.Entries.size() == before.Entries.size());
    CHECK(after.Entries.front().Key == before.Entries.front().Key);
    CHECK(after.Entries.front().InsertGeneration == before.Entries.front().InsertGeneration);
    CHECK(Get(*target, H3).has_value());
    CHECK_FALSE(Get(*target, H1).has_value());
}

TEST_CASE("Zobrist - entries stamped later than the clock expire when the clock moved back", "[zobrist][visibility]")
{
    // A second Run() on the same cache restarts the generation clock. With a max age, both visibilities then expire
    // what the earlier run left behind (Immediate: the unsigned age wraps around; Committed: same rule in the sweep).
    for (auto const visibility : { CacheVisibility::Committed, CacheVisibility::Immediate }) {
        CAPTURE(visibility == CacheVisibility::Committed);
        auto const cache = MakeCache({ .MaxAge = 2, .Visibility = visibility });
        cache->SetGeneration(5);
        cache->Insert(H1, { Operon::Scalar { 1 } });
        cache->SetGeneration(1);
        if (visibility == CacheVisibility::Committed) {
            CHECK(cache->Size() == 0); // swept at the generation boundary
        }
        CHECK_FALSE(Get(*cache, H1));
        CHECK(cache->Size() == 0);
    }

    auto const unbounded = MakeCache(Committed());
    unbounded->SetGeneration(5);
    unbounded->Insert(H1, { Operon::Scalar { 1 } });
    unbounded->SetGeneration(1);
    CHECK(Get(*unbounded, H1)); // MaxAge 0 never expires
}

TEST_CASE(
    "Zobrist Committed - AbortGeneration erases the entries of the current clock value only", "[zobrist][visibility]")
{
    auto const cache = MakeCache(Committed());
    cache->SetGeneration(1);
    cache->Insert(H1, { Operon::Scalar { 1 } });
    cache->SetGeneration(2);
    {
        auto const slot = cache->BeginSlot();
        cache->Insert(H2, { Operon::Scalar { -10 } });
    }
    OnOtherThread([&] { cache->Insert(H3, { Operon::Scalar { -20 } }); });
    REQUIRE(cache->Size() == 3);

    cache->AbortGeneration();
    CHECK(cache->Size() == 1);
    CHECK(Get(*cache, H1));
    CHECK_FALSE(Get(*cache, H2));
    CHECK_FALSE(Get(*cache, H3));

    // The retry of the same generation starts from nothing: the discarded -10 does not take part in the minimum.
    cache->Insert(H2, { Operon::Scalar { 10 } });
    cache->SetGeneration(3);
    auto const retried = GetInSlot(*cache, H2);
    REQUIRE(retried);
    CHECK((*retried)[0] == Operon::Scalar { 10 });

    cache->AbortGeneration(); // generation 3 has no entries yet: nothing to erase
    CHECK(cache->Size() == 2);
}

TEST_CASE("Zobrist Immediate - AbortGeneration is a no-op", "[zobrist][visibility]")
{
    auto const cache = MakeCache(Immediate());
    cache->SetGeneration(2);
    cache->Insert(H1, { Operon::Scalar { 1 } });
    cache->AbortGeneration();
    CHECK(cache->Size() == 1);
    CHECK(Get(*cache, H1));
}

} // namespace Operon::Test
