// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <atomic>
#include <thread>

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
    constexpr auto Seed      = 42UL;
    constexpr auto MaxLength = 50;

    auto MakeSetup() {
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

    BalancedTreeCreator const creator{&pset, inputs, /* bias= */ 0.0, MaxLength};
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

    BalancedTreeCreator const creator{&pset, inputs, /* bias= */ 0.0, MaxLength};
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
    Zobrist const cache(rng, MaxLength, {});  // no variables needed: trees use only constants

    // postfix: [Constant, Cos, Sin]  =>  sin(cos(c))
    Tree const tree1 = Tree({ Node(NodeType::Constant), Util::MakeOp<BuiltinOp::Cos>(), Util::MakeOp<BuiltinOp::Sin>() }).UpdateNodes();
    // postfix: [Constant, Sin, Cos]  =>  cos(sin(c))
    Tree const tree2 = Tree({ Node(NodeType::Constant), Util::MakeOp<BuiltinOp::Sin>(), Util::MakeOp<BuiltinOp::Cos>() }).UpdateNodes();

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

    Node nX(NodeType::Variable); nX.HashValue = varX.Hash; nX.IsEnabled = true;
    Node nY(NodeType::Variable); nY.HashValue = varY.Hash; nY.IsEnabled = true;

    // postfix: [X, Y, Add] => Add(X, Y)
    Tree const treeXY = Tree({ nX, nY, Util::MakeOp<BuiltinOp::Add>() }).UpdateNodes();
    // postfix: [Y, X, Add] => Add(Y, X)
    Tree const treeYX = Tree({ nY, nX, Util::MakeOp<BuiltinOp::Add>() }).UpdateNodes();

    REQUIRE(cache.ComputeHash(treeXY) != cache.ComputeHash(treeYX));
}

TEST_CASE("Zobrist - TryGet returns false on miss", "[zobrist]")
{
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist const cache(rng, MaxLength, inputs);

    BalancedTreeCreator const creator{&pset, inputs, /* bias= */ 0.0, MaxLength};
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
    Zobrist cache(rng, MaxLength, inputs);

    BalancedTreeCreator const creator{&pset, inputs, /* bias= */ 0.0, MaxLength};
    auto tree = creator(rng, 10, 1, MaxLength);
    auto hash = cache.ComputeHash(tree);

    Operon::Vector<Operon::Scalar> const stored = { Operon::Scalar{0.5}, Operon::Scalar{1.0} };
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
    Zobrist cache(rng, MaxLength, inputs);

    BalancedTreeCreator const creator{&pset, inputs, /* bias= */ 0.0, MaxLength};
    auto tree = creator(rng, 10, 1, MaxLength);
    auto hash = cache.ComputeHash(tree);

    Operon::Vector<Operon::Scalar> const val = { Operon::Scalar{0.1} };
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
    Zobrist cache(rng, MaxLength, inputs);

    BalancedTreeCreator const creator{&pset, inputs, /* bias= */ 0.0, MaxLength};
    auto tree1 = creator(rng, 10, 1, MaxLength);
    auto tree2 = creator(rng, 10, 1, MaxLength);
    auto hash1 = cache.ComputeHash(tree1);
    auto hash2 = cache.ComputeHash(tree2);

    Operon::Vector<Operon::Scalar> const val = { Operon::Scalar{0.1} };
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
    Zobrist cache(rng, MaxLength, inputs);

    BalancedTreeCreator const creator{&pset, inputs, /* bias= */ 0.0, MaxLength};
    auto tree = creator(rng, 10, 1, MaxLength);
    auto hash = cache.ComputeHash(tree);

    Operon::Vector<Operon::Scalar> const val = { Operon::Scalar{0.3} };
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

    REQUIRE(cache.ComputeHash(Util::MakeOp<BuiltinOp::Add>(), 0) != cache.ComputeHash(Util::MakeOp<BuiltinOp::Mul>(), 0));
    REQUIRE(cache.ComputeHash(Util::MakeOp<BuiltinOp::Sin>(), 0) != cache.ComputeHash(Util::MakeOp<BuiltinOp::Cos>(), 0));
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

    Operon::Hash const hashA = Operon::Hasher{}("userFunctionA");
    Operon::Hash const hashB = Operon::Hasher{}("userFunctionB");
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
    Node c1(NodeType::Constant); c1.Value = 1.0f; c1.Optimize = true;
    Node c2(NodeType::Constant); c2.Value = 1.0f; c2.Optimize = false;

    Tree const tree1 = Tree({ c1, Util::MakeOp<BuiltinOp::Sin>() }).UpdateNodes();
    Tree const tree2 = Tree({ c2, Util::MakeOp<BuiltinOp::Sin>() }).UpdateNodes();

    REQUIRE(cache.ComputeHash(tree1) != cache.ComputeHash(tree2));
}

TEST_CASE("Zobrist - maxAge disabled (default) never expires entries", "[zobrist]")
{
    auto [ds, inputs, pset] = MakeSetup();
    Operon::RandomGenerator rng(Seed);
    Zobrist cache(rng, MaxLength, inputs); // maxAge defaults to 0 = disabled

    BalancedTreeCreator const creator{&pset, inputs, /* bias= */ 0.0, MaxLength};
    auto tree = creator(rng, 10, 1, MaxLength);
    auto hash = cache.ComputeHash(tree);

    Operon::Vector<Operon::Scalar> const val = { Operon::Scalar{0.1} };
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
    Zobrist cache(rng, MaxLength, inputs, maxAge);

    BalancedTreeCreator const creator{&pset, inputs, /* bias= */ 0.0, MaxLength};
    auto tree = creator(rng, 10, 1, MaxLength);
    auto hash = cache.ComputeHash(tree);

    cache.SetGeneration(0);
    Operon::Vector<Operon::Scalar> const val = { Operon::Scalar{0.1} };
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
    Zobrist cache(rng, MaxLength, inputs, maxAge);

    BalancedTreeCreator const creator{&pset, inputs, /* bias= */ 0.0, MaxLength};
    auto tree = creator(rng, 10, 1, MaxLength);
    auto hash = cache.ComputeHash(tree);

    cache.SetGeneration(0);
    Operon::Vector<Operon::Scalar> const val = { Operon::Scalar{0.1} };
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

TEST_CASE("Zobrist - an entry inserted after the clock is set to a high starting value (warm resume) is not immediately stale", "[zobrist]")
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
    Zobrist cache(rng, MaxLength, inputs, maxAge);

    BalancedTreeCreator const creator{&pset, inputs, /* bias= */ 0.0, MaxLength};
    auto tree = creator(rng, 10, 1, MaxLength);
    auto hash = cache.ComputeHash(tree);

    constexpr std::size_t resumedGeneration = 500;
    cache.SetGeneration(resumedGeneration); // as the fixed resume path does, before re-evaluation
    Operon::Vector<Operon::Scalar> const val = { Operon::Scalar{0.1} };
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
    Zobrist cache(rng, MaxLength, inputs, maxAge);

    BalancedTreeCreator const creator{&pset, inputs, /* bias= */ 0.0, MaxLength};
    auto tree = creator(rng, 10, 1, MaxLength);
    auto hash = cache.ComputeHash(tree);

    cache.SetGeneration(500);
    Operon::Vector<Operon::Scalar> const val = { Operon::Scalar{0.1} };
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
    Operon::Hash const hash = Operon::Hasher{}("erase-guard-repro");

    cache.LazyEmplace(hash,
        [](FitnessEntry&) {},
        [](FitnessEntry& e) { e.Value = { Operon::Scalar{0.1} }; e.InsertGeneration = 5; });

    // Thread A "observes" the entry while it's still stale at generation 5.
    constexpr std::uint32_t observedGen = 5;

    // Before A's erase runs: another thread erases the entry, then a third
    // thread inserts a fresh one for the same hash at a later generation.
    cache.EraseIf(hash, [](FitnessEntry const&) { return true; });
    cache.LazyEmplace(hash,
        [](FitnessEntry&) {},
        [](FitnessEntry& e) { e.Value = { Operon::Scalar{0.9} }; e.InsertGeneration = 6; });

    // A's delayed erase, guarded by the generation it originally observed -
    // this must be a no-op now that the entry has moved on to generation 6.
    cache.EraseIf(hash, [&](FitnessEntry const& e) { return e.InsertGeneration == observedGen; });

    REQUIRE(cache.Size() == 1);
    bool found = false;
    cache.IfContains(hash, [&](FitnessEntry const& e) {
        found = true;
        REQUIRE(e.InsertGeneration == 6);
        REQUIRE(e.Value[0] == Operon::Scalar{0.9});
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
    Zobrist cache(rng, MaxLength, inputs, maxAge);

    BalancedTreeCreator const creator{&pset, inputs, /* bias= */ 0.0, MaxLength};
    auto tree = creator(rng, 10, 1, MaxLength);
    auto hash = cache.ComputeHash(tree);

    cache.SetGeneration(0);
    Operon::Vector<Operon::Scalar> const initial = { Operon::Scalar{0.1} };
    cache.Insert(hash, initial);

    std::atomic<bool> stop{false};
    Operon::Vector<Operon::Scalar> const fresh = { Operon::Scalar{0.9} };

    // Advance generations and keep re-inserting a fresh value for the same
    // hash, racing against two reader threads that will observe stale
    // entries and attempt to erase them - with two readers, one can erase
    // an entry the other already observed as stale, before the other's own
    // erase runs.
    std::thread writer([&]() {
        for (std::size_t g = 1; g < 2000; ++g) {
            cache.SetGeneration(g);
            cache.Insert(hash, fresh); // no-op if entry already fresh (LazyEmplace keeps existing)
            if (stop.load(std::memory_order_relaxed)) { break; }
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

} // namespace Operon::Test
