// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "checkpoint_helpers.hpp"
#include "operon/algorithms/config.hpp"
#include "operon/algorithms/gp.hpp"
#include "operon/core/constraint.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/node.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/serialization.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/version.hpp"
#include "operon/hash/zobrist.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/generator.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/operators/mutation.hpp"
#include "operon/operators/reinserter.hpp"
#include "operon/operators/selector.hpp"
#include "operon/optimizer/optimizer.hpp"
#include "resume_validation.hpp"
#include "util.hpp"

// `--resume` against checkpoints that are well-formed files but do not belong to the run that reads them: the checks of
// cli/source/resume_validation.cpp and ResumeFromCheckpoint (cli/source/util.cpp), with the checkpoints made through
// the library API. The generic decoder limits are not what is tested here.
namespace Operon::Test {
namespace {

    using DTable = DispatchTable<Operon::Scalar>;
    using Catch::Matchers::ContainsSubstring;

    constexpr std::size_t PopulationSize = 10;
    constexpr std::size_t MaxLength = 20;
    constexpr std::size_t Seed = 7;

    struct RunSpec {
        std::size_t Generations { 3 };
        bool Cache { false };
    };

    auto Parse(std::vector<std::string> extra) -> cxxopts::ParseResult
    {
        std::vector<std::string> args { "operon_gp", "--dataset", "./data/Poly-10.csv", "--target", "Y",
            "--population-size", std::to_string(PopulationSize), "--pool-size", std::to_string(PopulationSize),
            "--maxlength", std::to_string(MaxLength), "--seed", std::to_string(Seed) };
        args.insert(args.end(), extra.begin(), extra.end());
        std::vector<char*> argv;
        for (auto& arg : args) {
            argv.push_back(arg.data());
        }
        return ParseOptions(InitOptions("operon_gp", "test"), static_cast<int>(argv.size()), argv.data());
    }

    // A small GP over Poly-10 whose generation limit is `run.Generations` (with a cache if `run.Cache`); `fn` receives
    // the algorithm, its problem, the master RNG and the run fingerprint of the options parsed last. Everything is
    // rebuilt on every call: the objects of the run that writes a checkpoint share nothing with the run that reads it.
    template <typename Fn> auto WithGp(RunSpec const& run, ShapeConstraintSet const* shape, Fn&& fn)
    {
        Dataset ds { "./data/Poly-10.csv", /*hasHeader=*/true };
        Operon::Problem problem { gsl::not_null<Dataset*>(&ds) };
        problem.SetTrainingRange(0, 100);
        problem.SetTestRange(100, 200);
        problem.SetTarget("Y");
        problem.ConfigurePrimitiveSet(PrimitiveSet::Arithmetic);
        auto const inputs = std::vector<Operon::Hash>(problem.GetInputs().begin(), problem.GetInputs().end());

        Operon::BalancedTreeCreator creator { &problem.GetPrimitiveSet(), inputs, 0.0, MaxLength };
        Operon::UniformTreeInitializer treeInit { &creator };
        treeInit.ParameterizeDistribution(std::size_t { 1 }, MaxLength);
        treeInit.SetMinDepth(1);
        treeInit.SetMaxDepth(6);
        Operon::NormalCoefficientInitializer coeffInit;
        coeffInit.ParameterizeDistribution(Operon::Scalar { 0 }, Operon::Scalar { 1 });
        Operon::SubtreeCrossover crossover { 0.9, 6, MaxLength };
        Operon::OnePointMutation<std::normal_distribution<Operon::Scalar>> onePoint;
        onePoint.ParameterizeDistribution(Operon::Scalar { 0 }, Operon::Scalar { 1 });
        Operon::MultiMutation mutator;
        mutator.Add(&onePoint, 1.0);

        DTable dtable;
        Operon::GeneticAlgorithmConfig config;
        config.Generations = run.Generations;
        config.Evaluations = std::size_t { 1 } << 40U;
        config.Iterations = 0;
        config.PopulationSize = PopulationSize;
        config.PoolSize = PopulationSize;
        config.Seed = Seed;
        Operon::RandomGenerator cacheRng { Seed };
        Operon::Zobrist cache { cacheRng, static_cast<int>(MaxLength), problem.GetInputs(), ZobristOptions {} };
        if (run.Cache) {
            config.Cache = &cache;
        }

        Operon::LevenbergMarquardtOptimizer<DTable, Operon::OptimizerType::Eigen> optimizer { &dtable, &problem };
        optimizer.SetIterations(0);
        Operon::CoefficientOptimizer const coeffOptimizer { &optimizer };
        Operon::Evaluator<DTable> evaluator { &problem, &dtable };
        evaluator.SetBudget(config.Evaluations);
        Operon::ComparisonCallback const comp = Operon::SingleObjectiveComparison { 0 };
        Operon::TournamentSelector fem { comp };
        Operon::TournamentSelector male { comp };
        Operon::BasicOffspringGenerator generator { &evaluator, &crossover, &mutator, &fem, &male, &coeffOptimizer };
        Operon::KeepBestReinserter reinserter { comp, /*eliteCount=*/1 };
        Operon::GeneticProgrammingAlgorithm gp { config, &problem, &treeInit, &coeffInit, &generator, &reinserter };
        Operon::RandomGenerator random { Seed };
        return fn(gp, problem, random, MakeRunFingerprint(gp, shape));
    }

    auto TempPath(std::string const& name) -> std::string
    {
        return (std::filesystem::temp_directory_path() / ("operon_resume_validation_" + name + ".beve")).string();
    }

    // Writes the checkpoint of a run that completed `run.Generations` generations, as the command line does (with the
    // run fingerprint of the options `extra`), after `tweak` changed it.
    void WriteCheckpoint(RunSpec const& run, std::string const& path, std::vector<std::string> const& extra = {},
        std::function<void(Serialization::Checkpoint&)> const& tweak = {})
    {
        std::vector<std::string> args { "--generations", std::to_string(run.Generations) };
        args.insert(args.end(), extra.begin(), extra.end());
        Parse(args);
        WithGp(run, nullptr, [&](auto& gp, auto&, auto& random, auto const& fingerprint) -> int {
            gp.Run(random, nullptr, 1);
            auto cp = Serialization::CheckpointFromBeve(CaptureCheckpoint(gp, random));
            REQUIRE(cp.has_value());
            cp->Run = fingerprint;
            if (tweak) {
                tweak(*cp);
            }
            REQUIRE(Serialization::SaveCheckpoint(*cp, path));
            return 0;
        });
    }

    // What `operon_gp --resume path <extra>` does with the checkpoint (limit and cache as given on the command line).
    auto Resume(RunSpec run, std::string const& path, std::vector<std::string> extra = {},
        std::function<void(GeneticAlgorithmBase&, Problem&)> const& prepare = {}) -> bool
    {
        extra.insert(extra.end(), { "--resume", path });
        auto const result = Parse(extra);
        run.Generations = GetLimit(result, "generations");
        return WithGp(run, nullptr, [&](auto& gp, auto& problem, auto& random, auto const& fingerprint) -> bool {
            if (prepare) {
                prepare(gp, problem);
            }
            return ResumeFromCheckpoint(gp, random, result, fingerprint);
        });
    }

    auto Message(std::function<void()> const& fn) -> std::string
    {
        try {
            fn();
        } catch (std::runtime_error const& e) {
            return e.what();
        }
        return {};
    }

    auto VarNode(Operon::Hash hash) -> Node { return Node(NodeType::Variable, hash); }

    auto FuncNode(BuiltinOp op, std::uint16_t arity) -> Node
    {
        return Node::Function(static_cast<Operon::Hash>(op), arity);
    }

    auto Individuals(std::vector<Tree> trees) -> Operon::Vector<Individual>
    {
        Operon::Vector<Individual> population;
        for (auto& tree : trees) {
            Individual ind;
            ind.Genotype = std::move(tree);
            if (!ind.Genotype.Nodes().empty()) {
                ind.Genotype.UpdateNodes();
            }
            ind.Fitness = { Operon::Scalar { 1 } };
            population.push_back(std::move(ind));
        }
        return population;
    }

    auto Constraint(ShapeConstraintOp op, std::string variable, std::optional<int> sign,
        std::optional<std::pair<Operon::Scalar, Operon::Scalar>> bound) -> ShapeConstraint
    {
        ShapeConstraint constraint;
        constraint.Op = op;
        constraint.Variable = std::move(variable);
        constraint.Sign = sign;
        constraint.Bound = bound;
        return constraint;
    }

    // x1 + x1 + ... (`terms` variables): 2 * terms - 1 nodes
    auto Sum(Operon::Hash variable, std::size_t terms) -> Tree
    {
        Operon::Vector<Node> nodes { VarNode(variable) };
        for (std::size_t i = 1; i < terms; ++i) {
            nodes.push_back(VarNode(variable));
            nodes.push_back(FuncNode(BuiltinOp::Add, 2));
        }
        return Tree { std::move(nodes) };
    }

} // namespace

// ---- the generation limit ----

TEST_CASE("Resume: a finite --generations below the checkpoint's generation is rejected", "[resume][cli]")
{
    auto const path = TempPath("generations");
    WriteCheckpoint({ .Generations = 3 }, path);

    SECTION("equal is fine") { CHECK(Resume({}, path, { "--generations", "3" })); }
    SECTION("greater is fine") { CHECK(Resume({}, path, { "--generations", "5" })); }
    SECTION("unlimited is fine") { CHECK(Resume({}, path, { "--generations", "unlimited" })); }
    SECTION("lower is rejected, naming both")
    {
        auto const message = Message([&] { Resume({}, path, { "--generations", "2" }); });
        CHECK_THAT(message, ContainsSubstring("generation 3"));
        CHECK_THAT(message, ContainsSubstring("--generations is 2"));
    }
    SECTION("zero is rejected") { CHECK_THROWS_AS(Resume({}, path, { "--generations", "0" }), std::runtime_error); }
    std::filesystem::remove(path);
}

TEST_CASE("CheckGenerationLimit: only a finite limit below the checkpoint's generation fails", "[resume]")
{
    CHECK_NOTHROW(CheckGenerationLimit("p", 0, 0));
    CHECK_NOTHROW(CheckGenerationLimit("p", 7, 7));
    CHECK_NOTHROW(CheckGenerationLimit("p", 7, 8));
    CHECK_NOTHROW(CheckGenerationLimit("p", 7, std::numeric_limits<std::size_t>::max()));
    CHECK_THROWS_AS(CheckGenerationLimit("p", 7, 6), std::runtime_error);
    CHECK_THROWS_AS(CheckGenerationLimit("p", 1, 0), std::runtime_error);
}

// ---- the cache, in both directions ----

TEST_CASE("Resume: a cache-less checkpoint is rejected by a run with a cache, and the reverse", "[resume][cli]")
{
    auto const plain = TempPath("plain");
    auto const cached = TempPath("cached");
    WriteCheckpoint({ .Generations = 3, .Cache = false }, plain);
    WriteCheckpoint({ .Generations = 3, .Cache = true }, cached);

    CHECK(Resume({ .Cache = false }, plain));
    CHECK(Resume({ .Cache = true }, cached));
    CHECK_THAT(Message([&] { Resume({ .Cache = true }, plain); }), ContainsSubstring("has no transposition cache"));
    CHECK_THAT(
        Message([&] { Resume({ .Cache = false }, cached); }), ContainsSubstring("carries a transposition cache"));

    // (the plain checkpoint is still what it was: it resumes without a cache)
    CHECK(Resume({ .Cache = false }, plain));
    std::filesystem::remove(plain);
    std::filesystem::remove(cached);
}

TEST_CASE("CheckCachePresence: both or neither", "[resume]")
{
    CHECK_NOTHROW(CheckCachePresence("p", true, true));
    CHECK_NOTHROW(CheckCachePresence("p", false, false));
    CHECK_THROWS_AS(CheckCachePresence("p", true, false), std::runtime_error);
    CHECK_THROWS_AS(CheckCachePresence("p", false, true), std::runtime_error);
}

// ---- parents against the dataset and the primitive set ----

TEST_CASE("CheckPopulation: parents must fit the dataset, the primitive set and --maxlength", "[resume]")
{
    WithGp({}, nullptr, [&](auto&, Problem& problem, auto&, auto const&) -> int {
        auto const x = problem.GetInputs().front();
        auto const target = problem.TargetVariable().Hash;
        auto const check = [&](std::vector<Tree> trees, std::size_t maxLength = MaxLength) {
            CheckPopulation("ckpt.beve", Individuals(std::move(trees)), problem, maxLength);
        };

        SECTION("valid parents are accepted")
        {
            CHECK_NOTHROW(check({ Sum(x, 1), Sum(x, 5), Sum(x, 10) }));
            CHECK_NOTHROW(check({ Sum(x, 10) })); // 19 nodes
        }
        SECTION("a variable that is not an input")
        {
            auto const message = Message([&] { check({ Sum(x, 3), Sum(Operon::Hash { 0xDEADBEEF }, 3) }); });
            CHECK_THAT(message, ContainsSubstring("ckpt.beve"));
            CHECK_THAT(message, ContainsSubstring("parent 1"));
            CHECK_THAT(message, ContainsSubstring("0xdeadbeef"));
            CHECK_THAT(message, ContainsSubstring("not an input"));
        }
        SECTION("the target is not an input") { CHECK_THROWS_AS(check({ Sum(target, 2) }), std::runtime_error); }
        SECTION("an unknown function")
        {
            Tree const tree { VarNode(x), Node::Function(Operon::Hash { 0xFEEDFACE }, 1) };
            auto const message = Message([&] { check({ tree }); });
            CHECK_THAT(message, ContainsSubstring("0xfeedface"));
            CHECK_THAT(message, ContainsSubstring("not an enabled primitive"));
        }
        SECTION("a function of the library that this run's primitive set does not contain")
        {
            Tree const tree { VarNode(x), FuncNode(BuiltinOp::Sin, 1) };
            CHECK_THROWS_AS(check({ tree }), std::runtime_error);
        }
        SECTION("a function that is registered but disabled")
        {
            CHECK_NOTHROW(check({ Sum(x, 3) }));
            problem.GetPrimitiveSet().SetEnabled(static_cast<Operon::Hash>(BuiltinOp::Add), false);
            CHECK_THROWS_AS(check({ Sum(x, 3) }), std::runtime_error);
            problem.GetPrimitiveSet().SetEnabled(static_cast<Operon::Hash>(BuiltinOp::Add), true);
            problem.GetPrimitiveSet().SetFrequency(static_cast<Operon::Hash>(BuiltinOp::Add), 0);
            CHECK_THROWS_AS(check({ Sum(x, 3) }), std::runtime_error);
        }
        SECTION("a tree longer than --maxlength")
        {
            CHECK_NOTHROW(check({ Sum(x, 11) }, 21)); // 21 nodes are fine under --maxlength 21 ...
            auto const message = Message([&] { check({ Sum(x, 3), Sum(x, 11) }); }); // ... but not under 20
            CHECK_THAT(message, ContainsSubstring("parent 1"));
            CHECK_THAT(message, ContainsSubstring("21 nodes"));
            CHECK_THAT(message, ContainsSubstring("--maxlength 20"));
        }
        SECTION("an empty genotype")
        {
            CHECK_THAT(Message([&] { check({ Sum(x, 2), Tree {} }); }), ContainsSubstring("parent 1 has an empty"));
        }
        return 0;
    });
}

TEST_CASE("Resume: parents that do not fit the dataset or primitive set are rejected before anything is applied",
    "[resume][cli]")
{
    auto const path = TempPath("parents");
    auto const x = WithGp({}, nullptr,
        [](auto&, Problem& problem, auto&, auto const&) -> Operon::Hash { return problem.GetInputs().front(); });
    auto const craft = [&](Tree tree) {
        WriteCheckpoint({}, path, {}, [&](Serialization::Checkpoint& cp) {
            cp.Population[3].Genotype = std::move(tree);
            cp.Population[3].Genotype.UpdateNodes();
        });
    };

    SECTION("untouched, the checkpoint resumes")
    {
        WriteCheckpoint({}, path);
        CHECK(Resume({}, path));
    }
    SECTION("an unknown variable")
    {
        craft(Sum(Operon::Hash { 0xDEADBEEF }, 4));
        CHECK_THAT(Message([&] { Resume({}, path); }), ContainsSubstring("parent 3"));
    }
    SECTION("an unknown function")
    {
        craft(Tree { VarNode(x), Node::Function(Operon::Hash { 0xFEEDFACE }, 1) });
        CHECK_THAT(Message([&] { Resume({}, path); }), ContainsSubstring("not an enabled primitive"));
    }
    SECTION("a function the run has disabled")
    {
        WriteCheckpoint({}, path);
        auto const disable = [](GeneticAlgorithmBase&, Problem& problem) {
            problem.GetPrimitiveSet().SetEnabled(static_cast<Operon::Hash>(BuiltinOp::Mul), false);
            problem.GetPrimitiveSet().SetEnabled(static_cast<Operon::Hash>(BuiltinOp::Add), false);
            problem.GetPrimitiveSet().SetEnabled(static_cast<Operon::Hash>(BuiltinOp::Sub), false);
            problem.GetPrimitiveSet().SetEnabled(static_cast<Operon::Hash>(BuiltinOp::Div), false);
        };
        CHECK_THAT(Message([&] { Resume({}, path, {}, disable); }), ContainsSubstring("not an enabled primitive"));
    }
    SECTION("a tree longer than --maxlength")
    {
        craft(Sum(x, 11));
        CHECK_THAT(Message([&] { Resume({}, path); }), ContainsSubstring("21 nodes"));
    }
    std::filesystem::remove(path);
}

TEST_CASE("Resume: exactly one worker RNG state per offspring slot is required", "[resume][cli]")
{
    auto const path = TempPath("workers");
    SECTION("the states are there")
    {
        WriteCheckpoint({}, path);
        CHECK(Resume({}, path));
    }
    SECTION("none")
    {
        WriteCheckpoint({}, path, {}, [](Serialization::Checkpoint& cp) { cp.WorkerRngStates.clear(); });
        CHECK_THAT(Message([&] { Resume({}, path); }), ContainsSubstring("0 worker RNG states"));
    }
    SECTION("one too few")
    {
        WriteCheckpoint({}, path, {}, [](Serialization::Checkpoint& cp) { cp.WorkerRngStates.pop_back(); });
        CHECK_THROWS_AS(Resume({}, path), std::runtime_error);
    }
    SECTION("one too many")
    {
        WriteCheckpoint(
            {}, path, {}, [](Serialization::Checkpoint& cp) { cp.WorkerRngStates.push_back(cp.WorkerRngStates[0]); });
        CHECK_THROWS_AS(Resume({}, path), std::runtime_error);
    }
    std::filesystem::remove(path);
}

TEST_CASE("Resume: a rejected checkpoint leaves the algorithm object untouched", "[resume][cli]")
{
    auto const path = TempPath("untouched");
    WriteCheckpoint({ .Generations = 3 }, path);
    auto const result = Parse({ "--generations", "2", "--resume", path });
    WithGp({ .Generations = 2 }, nullptr, [&](auto& gp, auto&, auto& random, auto const& fingerprint) -> int {
        auto const generation = gp.Generation();
        auto const rngBefore = random.state();
        CHECK_THROWS_AS(ResumeFromCheckpoint(gp, random, result, fingerprint), std::runtime_error);
        CHECK(gp.Generation() == generation);
        CHECK(random.state() == rngBefore);
        CHECK_FALSE(gp.IsFitted());
        return 0;
    });
    std::filesystem::remove(path);
}

TEST_CASE("Resume: a restored algorithm is fitted and continues", "[resume][cli]")
{
    auto const path = TempPath("fitted");
    WriteCheckpoint({ .Generations = 3 }, path);
    auto const result = Parse({ "--generations", "5", "--resume", path });
    WithGp({ .Generations = 5 }, nullptr, [&](auto& gp, auto&, auto& random, auto const& fingerprint) -> int {
        CHECK_FALSE(gp.IsFitted());
        REQUIRE(ResumeFromCheckpoint(gp, random, result, fingerprint));
        CHECK(gp.IsFitted());
        CHECK(gp.Generation() == 3);
        CHECK_NOTHROW(gp.Run(random, nullptr, 1, /*warmStart=*/true));
        CHECK(gp.Generation() == 5);
        return 0;
    });
    std::filesystem::remove(path);
}

// ---- the run fingerprint: build identity and shape constraints ----

TEST_CASE("BuildIdentityHash: revision, precision, architecture and ISA level each change it", "[resume]")
{
    BuildInfo const base = GetBuildInfo();
    CHECK(BuildIdentityHash(base) == BuildIdentityHash(base));

    auto const changed = [&](auto change) {
        auto info = base;
        change(info);
        return BuildIdentityHash(info);
    };
    CHECK(changed([](BuildInfo& i) { i.Revision += "-other"; }) != BuildIdentityHash(base));
    CHECK(changed([](BuildInfo& i) { i.Precision = i.Precision == "single" ? "double" : "single"; })
        != BuildIdentityHash(base));
    CHECK(changed([](BuildInfo& i) { i.Arch += "-other"; }) != BuildIdentityHash(base));
    CHECK(changed([](BuildInfo& i) { i.IsaLevel += "-other"; }) != BuildIdentityHash(base));
    // facts that do not change what is computed do not matter
    CHECK(changed([](BuildInfo& i) { i.BuildTimestamp += "x"; }) == BuildIdentityHash(base));
    CHECK(changed([](BuildInfo& i) { i.CompilerFlags += " -g"; }) == BuildIdentityHash(base));
}

TEST_CASE("The run fingerprint carries the build identity of this build", "[resume]")
{
    Parse({ "--generations", "3" });
    WithGp({}, nullptr, [&](auto&, auto&, auto&, Serialization::RunFingerprint const& fingerprint) -> int {
        auto const it = std::ranges::find(fingerprint.Groups, std::string { BuildIdentityGroup });
        REQUIRE(it != fingerprint.Groups.end());
        CHECK(fingerprint.Hashes[static_cast<std::size_t>(it - fingerprint.Groups.begin())]
            == BuildIdentityHash(GetBuildInfo()));
        return 0;
    });
}

TEST_CASE("FingerprintMismatchMessage: another build is named as such, other differences are listed", "[resume]")
{
    Serialization::RunFingerprint const saved { { "build.identity", "dataset.content", "option.objective" },
        { 1, 2, 3 } };

    CHECK(FingerprintMismatchMessage("p", saved, saved).empty());

    auto otherBuild = saved;
    otherBuild.Hashes[0] = 99;
    auto const buildMessage = FingerprintMismatchMessage("p", saved, otherBuild);
    CHECK_THAT(buildMessage, ContainsSubstring("written by another build"));
    CHECK_THAT(buildMessage, ContainsSubstring("'p'"));

    // with a different build, other differences are not what the user can act on
    auto both = otherBuild;
    both.Hashes[1] = 98;
    CHECK_THAT(FingerprintMismatchMessage("p", saved, both), ContainsSubstring("another build"));
    CHECK_THAT(FingerprintMismatchMessage("p", saved, both), !ContainsSubstring("dataset.content"));

    auto otherOption = saved;
    otherOption.Hashes[2] = 97;
    auto const optionMessage = FingerprintMismatchMessage("p", saved, otherOption);
    CHECK_THAT(optionMessage, ContainsSubstring("differing: --objective"));
    CHECK_THAT(optionMessage, !ContainsSubstring("another build"));

    Serialization::RunFingerprint const withoutBuild { { "dataset.content", "option.objective" }, { 2, 3 } };
    CHECK_THAT(FingerprintMismatchMessage("p", withoutBuild, saved), ContainsSubstring("another build"));
    CHECK_THAT(FingerprintMismatchMessage("p", saved, withoutBuild), ContainsSubstring("another build"));
}

TEST_CASE("Resume: a checkpoint written by another build is rejected", "[resume][cli]")
{
    auto const path = TempPath("build");
    WriteCheckpoint({}, path, {}, [](Serialization::Checkpoint& cp) {
        auto const it = std::ranges::find(cp.Run->Groups, std::string { BuildIdentityGroup });
        REQUIRE(it != cp.Run->Groups.end());
        cp.Run->Hashes[static_cast<std::size_t>(it - cp.Run->Groups.begin())] ^= 0x1234;
    });
    auto const message = Message([&] { Resume({}, path); });
    CHECK_THAT(message, ContainsSubstring("written by another build"));
    std::filesystem::remove(path);
}

TEST_CASE("ShapeConstraintsHash: the content counts, the order of the domains and the path do not", "[resume]")
{
    ShapeConstraintSet base;
    base.Domains["x"] = { Operon::Scalar { 0.1 }, Operon::Scalar { 15 } };
    base.Domains["y"] = { Operon::Scalar { -1 }, Operon::Scalar { 1 } };
    base.Constraints.push_back(Constraint(ShapeConstraintOp::FirstDerivative, "x", -1, std::nullopt));
    base.Constraints.push_back(Constraint(
        ShapeConstraintOp::Identity, "", std::nullopt, std::pair { Operon::Scalar { 0 }, Operon::Scalar { 1 } }));
    auto const reference = ShapeConstraintsHash(&base);

    CHECK(ShapeConstraintsHash(&base) == reference);
    CHECK(ShapeConstraintsHash(nullptr) != reference);
    ShapeConstraintSet const empty;
    CHECK(ShapeConstraintsHash(nullptr) != ShapeConstraintsHash(&empty));

    auto const hashOf = [&](auto change) {
        auto set = base;
        change(set);
        return ShapeConstraintsHash(&set);
    };
    CHECK(hashOf([](ShapeConstraintSet& s) { s.Domains["x"].second = 16; }) != reference);
    CHECK(hashOf([](ShapeConstraintSet& s) { s.Domains["z"] = { 0, 1 }; }) != reference);
    CHECK(hashOf([](ShapeConstraintSet& s) { s.Constraints[0].Sign = 1; }) != reference);
    CHECK(hashOf([](ShapeConstraintSet& s) { s.Constraints[0].Variable = "y"; }) != reference);
    CHECK(
        hashOf([](ShapeConstraintSet& s) { s.Constraints[0].Op = ShapeConstraintOp::SecondDerivative; }) != reference);
    CHECK(hashOf([](ShapeConstraintSet& s) { s.Constraints[1].Bound->second = 2; }) != reference);
    CHECK(hashOf([](ShapeConstraintSet& s) { std::swap(s.Constraints[0], s.Constraints[1]); }) != reference);
    CHECK(hashOf([](ShapeConstraintSet& s) { s.Constraints.pop_back(); }) != reference);

    // the same domains, inserted in another order
    ShapeConstraintSet reordered;
    reordered.Domains["y"] = { Operon::Scalar { -1 }, Operon::Scalar { 1 } };
    reordered.Domains["x"] = { Operon::Scalar { 0.1 }, Operon::Scalar { 15 } };
    reordered.Constraints = base.Constraints;
    CHECK(ShapeConstraintsHash(&reordered) == reference);
}

TEST_CASE("The run fingerprint depends on the shape constraints, not on the path of their file", "[resume]")
{
    ShapeConstraintSet constraints;
    constraints.Domains["X1"] = { Operon::Scalar { 0 }, Operon::Scalar { 1 } };
    constraints.Constraints.push_back(Constraint(ShapeConstraintOp::FirstDerivative, "X1", 1, std::nullopt));
    auto other = constraints;
    other.Constraints[0].Sign = -1;

    auto const fingerprint = [&](std::vector<std::string> extra, ShapeConstraintSet const* shape) {
        Parse(std::move(extra));
        return WithGp(
            {}, shape, [](auto&, auto&, auto&, auto const& fp) -> Serialization::RunFingerprint { return fp; });
    };
    auto const first = fingerprint({ "--shape-constraints-config", "a.json" }, &constraints);
    auto const samePathOther = fingerprint({ "--shape-constraints-config", "a.json" }, &other);
    auto const otherPathSame = fingerprint({ "--shape-constraints-config", "b.json" }, &constraints);
    auto const none = fingerprint({}, nullptr);

    CHECK(FingerprintDifferences(first, otherPathSame).empty());
    CHECK(FingerprintDifferences(first, samePathOther) == std::vector<std::string> { "shape.constraints" });
    CHECK(FingerprintDifferences(first, none) == std::vector<std::string> { "shape.constraints" });
}

} // namespace Operon::Test
