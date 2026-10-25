// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <limits>
#include <memory>
#include <scn/scan.h>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>

#include "capabilities.hpp"
#include "cli_args.hpp"
#include "cli_error.hpp"
#include "resume_validation.hpp"
#include "util.hpp"

#include "operon/core/node.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/serialization.hpp"
#include "operon/core/types.hpp"
#include "operon/core/version.hpp"
#include "operon/hash/hash.hpp"
#include "operon/hash/zobrist.hpp"

using Operon::BuiltinOp;
using Operon::NodeType;

namespace Operon {

static const Operon::Map<std::string, PrimitiveSetConfig> Primitives { { "add", ToConfig(BuiltinOp::Add) },
    { "mul", ToConfig(BuiltinOp::Mul) }, { "sub", ToConfig(BuiltinOp::Sub) }, { "div", ToConfig(BuiltinOp::Div) },
    { "fmin", ToConfig(BuiltinOp::Fmin) }, { "fmax", ToConfig(BuiltinOp::Fmax) }, { "aq", ToConfig(BuiltinOp::Aq) },
    { "pow", ToConfig(BuiltinOp::Pow) }, { "powabs", ToConfig(BuiltinOp::Powabs) }, { "abs", ToConfig(BuiltinOp::Abs) },
    { "acos", ToConfig(BuiltinOp::Acos) }, { "asin", ToConfig(BuiltinOp::Asin) }, { "atan", ToConfig(BuiltinOp::Atan) },
    { "cbrt", ToConfig(BuiltinOp::Cbrt) }, { "ceil", ToConfig(BuiltinOp::Ceil) }, { "cos", ToConfig(BuiltinOp::Cos) },
    { "cosh", ToConfig(BuiltinOp::Cosh) }, { "exp", ToConfig(BuiltinOp::Exp) }, { "floor", ToConfig(BuiltinOp::Floor) },
    { "log", ToConfig(BuiltinOp::Log) }, { "logabs", ToConfig(BuiltinOp::Logabs) },
    { "log1p", ToConfig(BuiltinOp::Log1p) }, { "sin", ToConfig(BuiltinOp::Sin) }, { "sinh", ToConfig(BuiltinOp::Sinh) },
    { "sqrt", ToConfig(BuiltinOp::Sqrt) }, { "sqrtabs", ToConfig(BuiltinOp::Sqrtabs) },
    { "tan", ToConfig(BuiltinOp::Tan) }, { "tanh", ToConfig(BuiltinOp::Tanh) },
    { "square", ToConfig(BuiltinOp::Square) }, { "dyn", ToConfig(NodeType::Function) },
    { "constant", ToConfig(NodeType::Constant) }, { "variable", ToConfig(NodeType::Variable) } };

auto Split(const std::string& s, char delimiter) -> std::vector<std::string>
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// splits a string into substrings separated by delimiter
// formats a duration as dd:hh:mm:ss.ms
auto FormatDuration(std::chrono::duration<double> d) -> std::string
{
    auto h = std::chrono::duration_cast<std::chrono::hours>(d);
    auto m = std::chrono::duration_cast<std::chrono::minutes>(d - h);
    auto s = std::chrono::duration_cast<std::chrono::seconds>(d - h - m);
    auto l = std::chrono::duration_cast<std::chrono::milliseconds>(d - h - m - s);
    return fmt::format("{:#02d}:{:#02d}:{:#02d}.{:#03d}", h.count(), m.count(), s.count(), l.count());
}

auto FormatBytes(size_t bytes) -> std::string
{
    constexpr std::array<char, 6> sizes { " KMGT" };
    constexpr size_t base { 1024 };
    auto p = static_cast<size_t>(std::floor(std::log2(bytes) / std::log2(base)));
    return fmt::format("{:.2f} {}b", static_cast<double>(bytes) / std::pow(base, p), sizes.at(p));
}

auto ParseRange(std::string const& str) -> std::pair<size_t, size_t>
{
    auto result = scn::scan<size_t, size_t>(str, "{}:{}");
    if (!result) {
        throw std::runtime_error(fmt::format("could not parse range '{}'", str));
    }
    auto [a, b] = result->values();
    return std::make_pair(a, b);
}

auto ParsePrimitiveSetConfig(const std::string& options) -> PrimitiveSetConfig
{
    PrimitiveSetConfig config {};
    for (auto& s : Split(options, ',')) {
        if (auto it = Primitives.find(s); it != Primitives.end()) {
            config |= it->second;
        } else {
            throw std::runtime_error(fmt::format("Unrecognized symbol {}\n", s));
        }
    }
    return config;
}

auto PrintPrimitives(PrimitiveSetConfig config) -> void
{
    PrimitiveSet tmpSet;
    tmpSet.SetConfig(config);
    fmt::print("Built-in primitives:\n");
    fmt::print("{:<8}\t{:<50}\t{:>7}\t\t{:>9}\n", "Symbol", "Description", "Enabled", "Frequency");

    auto printRow = [&](Node const& node) {
        auto hash = node.HashValue;
        auto enabled = tmpSet.Contains(hash) && tmpSet.IsEnabled(hash);
        auto freq = enabled ? tmpSet.Frequency(hash) : 0U;
        fmt::print("{:<8}\t{:<50}\t{:>7}\t\t{:>9}\n", node.Name(), node.Desc(), enabled,
            freq != 0U ? std::to_string(freq) : "-");
    };

    for (size_t i = 0; i < Operon::BuiltinOpCount; ++i) {
        auto op = static_cast<BuiltinOp>(i);
        printRow(Node::Function(static_cast<Operon::Hash>(op), 0)); // arity irrelevant for Name()/Desc()
    }
    for (auto type : { NodeType::Constant, NodeType::Variable, NodeType::Ref, NodeType::Function }) {
        printRow(Node(type));
    }
}

auto InitOptions(std::string const& name, std::string const& desc, int width) -> cxxopts::Options
{
    cxxopts::Options opts(name, desc);
    opts.set_width(width);

    std::string const symbols
        = "add, sub, mul, div, exp, log, square, sqrt, cbrt, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, abs, "
          "aq, ceil, floor, fmin, fmax, log1p, logabs, sqrtabs, pow, powabs";

    opts.add_options()("dataset", "Dataset file name (csv) (required)", cxxopts::value<std::string>())(
        "shuffle", "Shuffle the input data", cxxopts::value<bool>()->default_value("false"))("standardize",
        "Standardize the training partition (zero mean, unit variance)",
        cxxopts::value<bool>()->default_value("false"))("train", "Training range specified as start:end (required)",
        cxxopts::value<std::string>())("test", "Test range specified as start:end", cxxopts::value<std::string>())(
        "target", "Name of the target variable (required)", cxxopts::value<std::string>())(
        "inputs", "Comma-separated list of input variables", cxxopts::value<std::string>())("epsilon",
        "Tolerance for fitness comparison in non-dominated sorting (operon_nsgp only; rejected by the other programs)",
        cxxopts::value<Operon::Scalar>()->default_value("1e-6"))("objective",
        "The error metric used for calculating fitness",
        cxxopts::value<std::string>()->default_value("r2"))("linear-scaling",
        "Apply linear scaling on model predictions", cxxopts::value<bool>()->default_value("true"))("skip-nonfinite",
        "Skip non-finite rows instead of clamping fitness to ErrMax on any non-finite prediction "
        "(SSE/MSE/NMSE/RMSE/MAE objectives only)",
        cxxopts::value<bool>()->default_value("false"))("nonfinite-penalty-weight",
        "Penalty weight applied to the non-finite row fraction when --skip-nonfinite is set (scaled by target variance "
        "for non-normalized metrics)",
        cxxopts::value<double>()->default_value("1.0"))("jit",
        "JIT mode: 'all' = JIT evaluator + optimizer, 'jac' = interpreter evaluator + JIT Jacobian optimizer (requires "
        "HAVE_ASMJIT). Use --jit=jac for jac mode; bare --jit defaults to all.",
        cxxopts::value<std::string>()->default_value("")->implicit_value("all"))("jit-max-length",
        "Skip JIT compilation for trees longer than this (0 = disabled)", cxxopts::value<int>()->default_value("0"))(
        "jit-min-visits", "Compile a tree only after it has been seen this many times (default 1 = always)",
        cxxopts::value<std::size_t>()->default_value("1"))(
        "population-size", "Population size", cxxopts::value<size_t>()->default_value("1000"))("pool-size",
        "Recombination pool size (how many generated offspring per generation)",
        cxxopts::value<size_t>()->default_value("1000"))("seed",
        "Random number seed. If omitted, all three programs draw a random seed (operon_gp and operon_nsgp print it in "
        "the 'seed' output column); operon_enum also treats 0 as random",
        cxxopts::value<Operon::RandomGenerator::result_type>())("generations",
        "Number of generations, or 'unlimited' (also any value >= 2^62)",
        cxxopts::value<std::string>()->default_value("1000"))("evaluations",
        "Budget of committed work in residual + Jacobian evaluations, or 'unlimited' (also any value >= 2^62). A "
        "generation counts only if everything it spent still fits in the budget, so a run can end with less than the "
        "budget used; initialization always counts and may exceed it. Must be at least --population-size",
        cxxopts::value<std::string>()->default_value("1000000"))("iterations", "Local optimization iterations",
        cxxopts::value<size_t>()->default_value("0"))("selection-pressure",
        "Not supported (rejected); in operon_gp/operon_nsgp set the maximum selection pressure with "
        "--offspring-generator os:<max-pressure>[:<comparison-factor>]",
        cxxopts::value<std::string>())("maxlength", "Maximum length", cxxopts::value<size_t>()->default_value("50"))(
        "maxdepth", "Maximum depth", cxxopts::value<size_t>()->default_value("10"))("crossover-probability",
        "The probability to apply crossover", cxxopts::value<Operon::Scalar>()->default_value("1.0"))(
        "crossover-internal-probability", "Crossover bias towards swapping function nodes",
        cxxopts::value<Operon::Scalar>()->default_value("0.9"))("mutation-probability",
        "The probability to apply mutation", cxxopts::value<Operon::Scalar>()->default_value("0.25"))("creator",
        "Tree creator: name:bias:mindepth:maxdepth:maxlength (e.g. --creator ptc2:0.5::20:30). Omitted fields use "
        "--creator-mindepth/--creator-maxdepth/--maxlength. BTC ignores depths to preserve length; PTC2 enforces max "
        "depth but can return shorter trees.",
        cxxopts::value<std::string>()->default_value("btc"))("creator-mindepth",
        "Default minimum tree depth for --creator; honored by grow, ignored by BTC/PTC2",
        cxxopts::value<std::size_t>()->default_value("1"))("creator-maxdepth",
        "Default maximum tree depth for --creator; enforced by grow/PTC2, ignored by BTC",
        cxxopts::value<std::size_t>()->default_value("100"))("female-selector",
        "Female selection operator, with optional parameters separated by : (eg, --selector tournament:5)",
        cxxopts::value<std::string>()->default_value("tournament"))("male-selector",
        "Male selection operator, with optional parameters separated by : (eg, --selector tournament:5)",
        cxxopts::value<std::string>()->default_value("tournament"))("offspring-generator",
        "OffspringGenerator operator, with optional parameters separated by : basic, os[:pressure[:factor]], "
        "brood[:size], poly[:size] (eg --offspring-generator os:10:0.5 or --offspring-generator brood:10)",
        cxxopts::value<std::string>()->default_value("basic"))("reinserter",
        "Reinsertion operator merging offspring in the recombination pool back into the population",
        cxxopts::value<std::string>()->default_value("keep-best"))("mutators",
        "Comma-separated mutation operators to use, with optional weight separated by : (default weight 1.0), e.g. "
        "--mutators onepoint:2,changefunc,replacesubtree. Available: onepoint, multipoint, changevar, changefunc, "
        "replacesubtree, insertsubtree, removechild, removesubtree, discretepoint, shuffle",
        cxxopts::value<std::string>()->default_value(
            "onepoint:1,changevar:1,changefunc:1,replacesubtree:1,insertsubtree:1,removesubtree:1,discretepoint:1"))(
        "elitism",
        "Number of best individuals guaranteed to carry over between generations, regardless of the reinserter's "
        "merge/replace strategy (default: 1 for GP, 0 for NSGA2)",
        cxxopts::value<size_t>())("enable-symbols", "Comma-separated list of enabled symbols (" + symbols + ")",
        cxxopts::value<std::string>())("local-search-probability", "Probability for local search",
        cxxopts::value<Operon::Scalar>()->default_value("1.0"))("lamarckian-probability",
        "Probability that the local search improvements are saved back into the chromosome",
        cxxopts::value<Operon::Scalar>()->default_value("1.0"))("disable-symbols",
        "Comma-separated list of disabled symbols (" + symbols + ")", cxxopts::value<std::string>())("symbolic",
        "Operate in symbolic mode - no coefficient tuning or coefficient mutation",
        cxxopts::value<bool>()->default_value("false"))(
        "show-primitives", "Display the primitive set used by the algorithm")("threads",
        "Number of threads to use for parallelism (default: all hardware threads); operon_gp and operon_nsgp require "
        "at least 1, operon_enum treats 0 as all hardware threads",
        cxxopts::value<size_t>())("timelimit", "Time limit after which the algorithm will terminate",
        cxxopts::value<size_t>()->default_value(std::to_string(std::numeric_limits<size_t>::max())))(
        "transposition-cache",
        "Cache fitness values keyed by Zobrist hash of tree structure; most effective with coefficient optimization "
        "enabled",
        cxxopts::value<bool>()->default_value("false"))("cache-max-age",
        "Expire transposition cache entries older than this many generations (0 = never expire); only effective with "
        "--transposition-cache",
        cxxopts::value<size_t>()->default_value("0"))("cache-visibility",
        "When a transposition cache entry becomes visible: committed (to other offspring slots only from the next "
        "generation, duplicate entries keep the smallest fitness, expiry at the generation boundary; cache hits then "
        "do not depend on slot arrival order, and the run does not depend on the thread count unless a time limit "
        "or a stop request cuts a generation; JIT evaluation is not covered) or immediate (to every slot at once; "
        "results can depend on thread timing); only "
        "effective with --transposition-cache",
        cxxopts::value<std::string>()->default_value("committed"))("pareto-front",
        "Write the non-dominated members of the retained parent population to this JSON file after the run (only "
        "effective with Pareto-based algorithms, e.g. operon_nsgp). The front is computed from the fitness vectors "
        "with the --epsilon tolerance used by the search: epsilon-equivalent individuals are listed once",
        cxxopts::value<std::string>())("model-selection",
        "Pareto front model selection: obj0 (lowest first objective), mdl, bic, aic",
        cxxopts::value<std::string>()->default_value("obj0"))("mdl-likelihood",
        "Likelihood for MDL/BIC/AIC model selection: gaussian or poisson",
        cxxopts::value<std::string>()->default_value("gaussian"))("checkpoint-interval",
        "Save a checkpoint every N generations (0 = disabled)", cxxopts::value<std::size_t>()->default_value("0"))(
        "checkpoint-file", "Path for checkpoint output (BEVE binary format)",
        cxxopts::value<std::string>()->default_value("checkpoint.beve"))("resume",
        "Resume a previous run from this checkpoint file (a file that does not exist starts a fresh run with a "
        "warning; any other problem is an error). The checkpoint carries the parent population, the random number "
        "generator states, a fingerprint of the build, dataset and options, and for a run with --transposition-cache "
        "the cache (entries, insertion generations, generation clock), under a checksum. Resume with the same "
        "dataset and options as the checkpointed run, with a build of the same revision, precision, architecture "
        "and ISA level (the compiler and its flags are not compared). The run then continues like the uninterrupted "
        "one unless --evaluations (or another limit that cuts a generation) binds: the evaluation budget applies "
        "per invocation, so a resumed run gets a fresh --evaluations and re-evaluates every parent once; its charge "
        "is the objective's residual and Jacobian work. Only these may differ: --threads; --generations "
        "(finite: at least the "
        "generation of the checkpoint); --evaluations; --timelimit; what is only reported or written "
        "(--checkpoint-interval, --checkpoint-file, --pareto-front, --model-selection, --mdl-likelihood, "
        "--probes-config, --report-json, --debug); the options that name the data (--dataset, --train, --test, "
        "--target, --inputs, --shape-constraints-config: the data content, resolved ranges, target, inputs and "
        "constraints are compared, not the paths); and --seed, only for a run without --transposition-cache. "
        "--transposition-cache, --cache-visibility and --cache-max-age must agree with the checkpoint (a cache-less "
        "checkpoint cannot be resumed with a cache, nor the reverse). A corrupt file, a checkpoint of another format "
        "version, another build, or a checkpoint that does not fit the dataset and primitive set is rejected. "
        "Fingerprints and cache-bearing checkpoints are only valid on the architecture that wrote them",
        cxxopts::value<std::string>())("probes-config",
        R"(Path to a JSON config file describing per-generation instrumentation probes to run, e.g. {"probes":[{"type":"cache_hit_rate","every":1}],"sink":{"type":"jsonl","path":"metrics.jsonl"}})",
        cxxopts::value<std::string>())("shape-constraints-config",
        R"(Path to a JSON config file describing shape constraints (function/derivative sign or value bounds over a domain box) to reject infeasible individuals during fitness evaluation, e.g. {"domains":{"x":[0.1,15]},"constraints":[{"op":"derivative","variable":"x","order":1,"sign":-1}]})",
        cxxopts::value<std::string>())("shape-enforcement",
        "Shape-constraint enforcement mode: hard-reject, penalty, extra-objective, penalty,extra-objective "
        "(default with constraints: hard-reject)",
        cxxopts::value<std::string>())("shape-penalty-weight",
        "Penalty weight applied to summed shape-constraint violation in penalty mode (finite, non-negative)",
        cxxopts::value<double>()->default_value("1.0"))("shape-unknown-violation",
        "Violation magnitude assigned to each uncertified shape constraint in penalty/extra-objective modes (finite, "
        "non-negative)",
        cxxopts::value<double>()->default_value("1.0"))("shape-worst-value",
        "Fitness value assigned by hard-reject shape enforcement to infeasible individuals (finite)",
        cxxopts::value<double>()->default_value("1.0"))("shape-bound-mode",
        "Shape-constraint bound backend: interval (default), combined or affine, each with an optional subdivision "
        "depth :N (0-20; the bound is the hull over the 2^N sub-boxes of the domain box, 0 or no suffix is the whole "
        "box), e.g. interval:4",
        cxxopts::value<std::string>()->default_value("interval"))("debug", "Debug mode (more information displayed)")(
        "help", "Print help")("version", "Print version and program information")("capabilities",
        "Print build identity, precision, ISA, optional features and registered option names as one JSON object and "
        "exit");
    return opts;
}

namespace {
    auto NormalizedOptionsStorage() -> std::vector<std::pair<std::string, std::string>>&
    {
        static std::vector<std::pair<std::string, std::string>> storage;
        return storage;
    }
} // namespace

auto NormalizedOptions() -> std::vector<std::pair<std::string, std::string>> const&
{
    return NormalizedOptionsStorage();
}

auto RunFingerprintExcludedOptions() -> std::vector<std::string> const&
{
    static std::vector<std::string> const excluded { "threads", "generations", "evaluations", "timelimit",
        "checkpoint-interval", "checkpoint-file", "resume", "pareto-front", "model-selection", "mdl-likelihood",
        "probes-config", "report-json", "debug", "help", "version", "capabilities", "show-primitives", "dataset",
        "train", "test", "target", "inputs", "shape-constraints-config", "transposition-cache", "cache-visibility",
        "cache-max-age", "seed" };
    return excluded;
}

auto MakeRunFingerprint(GeneticAlgorithmBase const& algo, ShapeConstraintSet const* shapeConstraints)
    -> Serialization::RunFingerprint
{
    Serialization::RunFingerprint fp;
    auto const add = [&](std::string name, std::uint64_t hash) {
        fp.Groups.push_back(std::move(name));
        fp.Hashes.push_back(hash);
    };

    auto const& problem = *algo.GetProblem();
    auto const& config = algo.GetConfig();
    add(std::string { BuildIdentityGroup }, BuildIdentityHash(GetBuildInfo()));
    add("shape.constraints", ShapeConstraintsHash(shapeConstraints));

    // dataset contents: the values of every variable (in variable order) as the problem sees them now, i.e. after
    // --standardize, and the names that bind them
    auto const* dataset = problem.GetDataset();
    std::uint64_t content = 0;
    auto names = dataset->VariableNames();
    for (auto const& variable : dataset->VariableHashes()) {
        auto const values = dataset->GetValues(variable);
        content = CombineHashes(content, variable);
        content = CombineHashes(content, HashBytes(values.data(), values.size() * sizeof(Scalar)));
    }
    for (auto const& name : names) {
        content = CombineHashes(content, HashText(name));
    }
    add("dataset.content", content);
    add("dataset.target", CombineHashes(HashText(problem.TargetVariable().Name), problem.TargetVariable().Hash));
    std::uint64_t inputs = 0;
    for (auto const input : problem.GetInputs()) {
        inputs = CombineHashes(inputs, input);
    }
    add("dataset.inputs", inputs);
    add("dataset.train", CombineHashes(problem.TrainingRange().Start(), problem.TrainingRange().End()));
    add("dataset.test", CombineHashes(problem.TestRange().Start(), problem.TestRange().End()));

    // the resolved configuration of the algorithm (what the options turned into)
    add("config.population-size", config.PopulationSize);
    add("config.pool-size", config.PoolSize);
    add("config.iterations", config.Iterations);
    add("config.epsilon", HashBytes(&config.Epsilon, sizeof(config.Epsilon)));
    add("config.crossover-probability", HashBytes(&config.CrossoverProbability, sizeof(config.CrossoverProbability)));
    add("config.mutation-probability", HashBytes(&config.MutationProbability, sizeof(config.MutationProbability)));
    add("config.local-search-probability",
        HashBytes(&config.LocalSearchProbability, sizeof(config.LocalSearchProbability)));
    add("config.lamarckian-probability",
        HashBytes(&config.LamarckianProbability, sizeof(config.LamarckianProbability)));
    add("config.objectives", algo.GetGenerator()->Evaluator()->ObjectiveCount());

    // every other option, by its effective value
    auto const& excluded = RunFingerprintExcludedOptions();
    for (auto const& [name, value] : NormalizedOptions()) {
        if (std::ranges::find(excluded, name) == excluded.end()) {
            add("option." + name, HashText(value));
        }
    }
    return fp;
}

namespace {
    auto VisibilityName(CacheVisibility visibility) -> char const*
    {
        return visibility == CacheVisibility::Immediate ? "immediate" : "committed";
    }

    // Restores the transposition cache that the checkpoint carries into the cache of the resumed run (the presence
    // of a cache on both sides was checked by CheckCachePresence). The two runs have to agree on what the entries
    // mean (visibility, maximum age, the random table the keys come from); otherwise the resumed run would not
    // continue the checkpointed one, so the mismatch is an error.
    auto RestoreCache(Zobrist* cache, std::size_t objectiveCount, Serialization::Checkpoint const& cp,
        std::string const& path) -> void
    {
        if (!cp.Cache || cache == nullptr) {
            return;
        }
        ZobristState state;
        try {
            state = Serialization::ToZobristState(*cp.Cache);
        } catch (std::invalid_argument const& e) {
            throw std::runtime_error(fmt::format("checkpoint '{}' is corrupt: {}", path, e.what()));
        }
        if (state.Visibility != cache->Visibility()) {
            throw std::runtime_error(fmt::format(
                "checkpoint '{}' was written with --cache-visibility={} but this run uses --cache-visibility={}", path,
                VisibilityName(state.Visibility), VisibilityName(cache->Visibility())));
        }
        if (state.MaxAge != cache->MaxAge()) {
            throw std::runtime_error(
                fmt::format("checkpoint '{}' was written with --cache-max-age={} but this run uses --cache-max-age={}",
                    path, state.MaxAge, cache->MaxAge()));
        }
        try {
            cache->Import(state, objectiveCount);
        } catch (std::invalid_argument const& e) {
            throw std::runtime_error(
                fmt::format("cannot restore the transposition cache of checkpoint '{}': {}", path, e.what()));
        }
    }
} // namespace

auto ResumeFromCheckpoint(GeneticAlgorithmBase& algo, RandomGenerator& rng, cxxopts::ParseResult const& result,
    Serialization::RunFingerprint const& fingerprint) -> bool
{
    if (!result.contains("resume")) {
        return false;
    }
    auto const path = result["resume"].as<std::string>();
    std::error_code existsError;
    if (!std::filesystem::exists(path, existsError) && !existsError) {
        // "resume if present": a checkpoint that was never written (the first run of a restartable job) is not an
        // error. Anything else that keeps the file from being used is.
        fmt::print(stderr, "warning: checkpoint '{}' does not exist, starting fresh\n", path);
        return false;
    }
    auto cp = Serialization::LoadCheckpoint(path);
    if (!cp) {
        throw std::runtime_error(fmt::format("cannot resume: checkpoint '{}' could not be read (see above)", path));
    }
    auto const& config = algo.GetConfig();
    auto const popSize = config.PopulationSize;
    if (cp->Population.size() != popSize) {
        throw std::runtime_error(fmt::format(
            "checkpoint population size {} doesn't match --population-size {}", cp->Population.size(), popSize));
    }
    // Before anything of the checkpoint is used: it must come from a run that computes what this one computes.
    if (!cp->Run) {
        throw std::runtime_error(fmt::format(
            "checkpoint '{}' carries no run fingerprint, so it cannot be checked against this run's options", path));
    }
    if (auto const message = FingerprintMismatchMessage(path, *cp->Run, fingerprint); !message.empty()) {
        throw std::runtime_error(message);
    }
    // Everything is checked before any state of the algorithm is replaced.
    CheckGenerationLimit(path, cp->Generation, config.Generations);
    CheckCachePresence(path, cp->Cache.has_value(), algo.GetGenerator()->Cache() != nullptr);
    CheckPopulation(path, cp->Population, *algo.GetProblem(), result["maxlength"].as<std::size_t>());
    auto const workerCount = std::max(config.PopulationSize, config.PoolSize);
    // Exactly one random state per offspring slot: without them the resumed run cannot continue the checkpointed one.
    if (cp->WorkerRngStates.size() != workerCount) {
        throw std::runtime_error(fmt::format(
            "checkpoint '{}' has {} worker RNG states but this run needs {} (max of --population-size and --pool-size)",
            path, cp->WorkerRngStates.size(), workerCount));
    }

    // The only step that can still throw is the import of the cache, and it leaves the cache as it was when it does:
    // it comes first, so a rejected checkpoint changes nothing. What follows only assigns.
    RestoreCache(algo.GetGenerator()->Cache(), algo.GetGenerator()->Evaluator()->ObjectiveCount(), *cp, path);
    rng.set_state(cp->RngState);
    algo.Generation() = cp->Generation;
    auto parents = algo.Parents();
    for (std::size_t i = 0; i < cp->Population.size(); ++i) {
        parents[i] = std::move(cp->Population[i]);
    }
    auto& workers = algo.WorkerRngs();
    workers.clear();
    workers.reserve(cp->WorkerRngStates.size());
    for (auto const& state : cp->WorkerRngStates) {
        RandomGenerator worker(0ULL);
        worker.set_state(state);
        workers.push_back(std::move(worker));
    }
    // The restored population is what a warm Run() continues: the object counts as fitted.
    algo.IsFitted() = true;
    return true;
}

auto MaybeSaveCheckpoint(GeneticAlgorithmBase const& algo, RandomGenerator const& rng,
    cxxopts::ParseResult const& result, Serialization::RunFingerprint const& fingerprint, bool force) -> void
{
    auto const interval = result["checkpoint-interval"].as<std::size_t>();
    auto const gen = algo.Generation();
    if (interval == 0) {
        return;
    }
    if (!force && (gen == 0 || gen % interval != 0)) {
        return;
    }

    // Avoid writing the same generation twice when the run ends exactly on an interval boundary.
    static uint64_t lastSaved = std::numeric_limits<uint64_t>::max();
    if (force && gen == lastSaved) {
        return;
    }

    auto const path = result["checkpoint-file"].as<std::string>();
    Serialization::Checkpoint cp;
    cp.RngState = rng.state();
    cp.Generation = gen;
    cp.Population.assign(algo.Parents().begin(), algo.Parents().end());
    cp.WorkerRngStates.reserve(algo.WorkerRngs().size());
    for (auto const& worker : algo.WorkerRngs()) {
        cp.WorkerRngStates.push_back(worker.state());
    }
    if (auto const* cache = algo.GetGenerator()->Cache()) {
        cp.Cache = Serialization::ToCacheCheckpoint(cache->Export());
    }
    cp.Run = fingerprint;
    if (!Serialization::SaveCheckpoint(cp, path)) {
        fmt::print(stderr,
            "warning: checkpoint save failed at generation {} — run continues but resume may not be possible\n", gen);
    } else {
        lastSaved = gen;
    }
}

auto SetupRanges(cxxopts::ParseResult const& result, Dataset const& dataset, Range& trainingRange, Range& testRange)
    -> void
{
    auto const rows = dataset.Rows<std::size_t>();
    if (!result.contains("train")) {
        trainingRange = Range { 0, 2 * rows / 3 };
    }
    if (!result.contains("test")) {
        if (trainingRange.Start() > 0) {
            testRange = Range { 0, trainingRange.Start() };
        } else if (trainingRange.End() < rows) {
            testRange = Range { trainingRange.End(), rows };
        } else {
            testRange = Range { 0, 1 };
        }
    }
}

namespace {
    // Looks up a variable by name, throwing std::runtime_error with a message
    // prefixed by `label` (e.g. "target variable" or "variable") if not found.
    auto ResolveVariable(Dataset const& dataset, std::string const& name, std::string_view label) -> Variable
    {
        auto res = dataset.GetVariable(name);
        if (!res) {
            throw std::runtime_error(fmt::format("{} {} does not exist in the dataset", label, name));
        }
        return *res;
    }
} // namespace

auto ResolveTarget(Dataset const& dataset, std::string const& targetName) -> Variable
{
    return ResolveVariable(dataset, targetName, "target variable");
}

auto BuildInputs(cxxopts::ParseResult const& result, Dataset const& dataset, Hash targetHash) -> std::vector<Hash>
{
    if (!result.contains("inputs")) {
        auto inputs = dataset.VariableHashes();
        std::erase(inputs, targetHash);
        return inputs;
    }
    std::vector<Hash> inputs;
    for (auto const& tok : Split(result["inputs"].as<std::string>(), ',')) {
        inputs.push_back(ResolveVariable(dataset, tok, "variable").Hash);
    }
    return inputs;
}

namespace {
    // Options registered for every program but acted on by only some of them. Accepting one silently hides a
    // misconfiguration (the run is unaffected), so each is rejected with a pointer to what does work.
    auto CheckOptions(cxxopts::ParseResult const& result, std::string_view program) -> Cli::Result<void>
    {
        using Cli::Error;
        using Cli::ErrorCode;
        if (result.count("selection-pressure") != 0) {
            auto const advice = program == "operon_enum"
                ? std::string {}
                : std::string { "; set the maximum selection pressure with --offspring-generator "
                                "os:<max-pressure>[:<comparison-factor>]" };
            return tl::unexpected(Error { ErrorCode::InvalidArguments, "--selection-pressure",
                fmt::format("is not supported and has no effect{}", advice) });
        }
        if (result.count("epsilon") != 0 && program != "operon_nsgp") {
            return tl::unexpected(Error { ErrorCode::InvalidArguments, "--epsilon",
                fmt::format("is only used by operon_nsgp (comparison tolerance in non-dominated sorting); {} does not "
                            "read it",
                    program) });
        }
        // operon_enum treats an explicit 0 as "all hardware threads" (see EnumerationEngine::Build); the
        // population-based programs construct their executor from the value directly.
        if (result.count("threads") != 0 && program != "operon_enum") {
            if (auto threads = Cli::ParseThreads(result["threads"].as<size_t>()); !threads) {
                return tl::unexpected(threads.error());
            }
        }
        for (auto const* name : { "generations", "evaluations" }) {
            if (auto limit = Cli::ParseLimit(name, result[name].as<std::string>()); !limit) {
                return tl::unexpected(limit.error());
            }
        }
        if (program != "operon_enum") {
            // A population algorithm needs something to evolve: both sizes must be positive.
            for (auto const* name : { "population-size", "pool-size" }) {
                if (result[name].as<size_t>() == 0) {
                    return tl::unexpected(
                        Error { ErrorCode::InvalidArguments, fmt::format("--{}", name), "must be at least 1" });
                }
            }
            // Initializing the population costs at least one evaluation per individual, so a smaller budget cannot
            // even pay for it. (The real cost can be larger, e.g. with local search; initialization may overshoot
            // the budget.)
            auto const evaluations = *Cli::ParseLimit("evaluations", result["evaluations"].as<std::string>());
            auto const population = result["population-size"].as<size_t>();
            if (evaluations < population) {
                return tl::unexpected(Error { ErrorCode::InvalidArguments, "--evaluations",
                    fmt::format("{} is smaller than --population-size {}: initializing the population costs at least "
                                "one evaluation per individual",
                        evaluations, population) });
            }
        }
        return {};
    }
} // namespace

auto GetLimit(cxxopts::ParseResult const& result, std::string const& name) -> size_t
{
    auto const limit = Cli::ParseLimit(name, result[name].as<std::string>());
    ENSURE(limit.has_value(), "ParseOptions validates --generations/--evaluations before any caller reads them");
    return *limit;
}

auto ParseOptions(cxxopts::Options&& opts, int argc, char** argv) -> cxxopts::ParseResult
{
    auto const helpText = opts.help(); // capture before move
    auto const program = opts.program();
    std::vector<std::string> optionNames;
    std::vector<cxxopts::HelpGroupDetails> optionGroups;
    for (auto const& group : opts.groups()) {
        optionGroups.push_back(opts.group_help(group));
        for (auto const& option : optionGroups.back().options) {
            optionNames.insert(optionNames.end(), option.l.begin(), option.l.end());
        }
    }
    cxxopts::ParseResult result;
    try {
        result = std::move(opts).parse(argc, argv);
    } catch (cxxopts::exceptions::parsing const& ex) {
        fmt::print(stderr, "error: {}. rerun with --help to see available options.\n", ex.what());
        std::exit(EXIT_FAILURE);
    }
    {
        // effective value of every registered option, for the run fingerprint of checkpoints
        auto& normalized = NormalizedOptionsStorage();
        normalized.clear();
        for (auto const& group : optionGroups) {
            for (auto const& option : group.options) {
                if (option.l.empty()) {
                    continue;
                }
                auto const& name = option.l.front();
                std::string value = option.has_default ? option.default_value : std::string { "<unset>" };
                for (auto const& argument : result.arguments()) {
                    if (argument.key() == name) {
                        value = argument.value();
                    }
                }
                normalized.emplace_back(name, std::move(value));
            }
        }
    }
    if (result.arguments().empty() || result.contains("help")) {
        fmt::print("{}\n", helpText);
        std::exit(EXIT_SUCCESS);
    }
    if (result.contains("version")) {
        fmt::print("{}\n", Operon::Version());
        std::exit(EXIT_SUCCESS);
    }
    if (result.contains("capabilities")) {
        fmt::print("{}\n", Cli::CapabilitiesJson(program, std::move(optionNames), Operon::GetBuildInfo()));
        std::exit(EXIT_SUCCESS);
    }
    if (auto const checked = CheckOptions(result, program); !checked) {
        std::exit(Cli::Report(checked.error()));
    }
    // --show-primitives is a "just print primitives/recipes and exit" flag (see PrintPrimitives /
    // operon_enum's PrintEnumerationFunctions) - it never touches the dataset/target, so it's exempt
    // from the checks below the same way --help/--version are, rather than forcing every caller to
    // supply a dataset just to display what primitives are available.
    if (result.contains("show-primitives")) {
        return result;
    }
    if (!result.contains("target")) {
        fmt::print(stderr, "error: no target variable was specified.\n");
        std::exit(EXIT_FAILURE);
    }
    if (!result.contains("dataset")) {
        fmt::print(stderr, "error: no dataset was specified.\n");
        std::exit(EXIT_FAILURE);
    }
    return result;
}
} // namespace Operon
