// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
#include "resolved_config.hpp"
#include "capabilities.hpp"
#include "final_record.hpp"
#include "operon/hash/sha256.hpp"
#include "operon/operators/reinserter.hpp"
#include <fstream>
#include <glaze/glaze.hpp>
#include <iterator>

namespace Operon::Cli {
namespace {
    // Configuration is serialized once, never in the generation hot path.
    template <typename T> auto Json(T const& value) -> std::string { return glz::write_json(value).value(); }
    auto Limit(std::size_t value) -> std::string { return value >= (std::uint64_t { 1 } << 62) ? "null" : Json(value); }
    struct Object {
        std::string text { "{" };
        void Add(std::string_view key, std::string const& value)
        {
            if (text.size() > 1) {
                text += ',';
            }
            text += Json(key) + ':' + value;
        }
        template <typename T> void Set(std::string_view key, T const& value) { Add(key, Json(value)); }
        auto Finish() const -> std::string { return text + '}'; }
    };
    auto RangeJson(Range range) -> std::string { return Json(std::array { range.Start(), range.End() }); }
    auto SelectorJson(SelectorBase const& selector) -> std::string
    {
        Object out;
        if (auto const* tournament = dynamic_cast<TournamentSelector const*>(&selector)) {
            out.Set("kind", "tournament");
            out.Set("tournament_size", tournament->GetTournamentSize());
        } else {
            out.Set("kind", dynamic_cast<RandomSelector const*>(&selector) ? "random" : "proportional");
        }
        return out.Finish();
    }
}

auto BuildResolvedConfig(cxxopts::ParseResult const& r, ResolvedObjects const& o) -> std::string
{
    auto const& c = o.Config;
    auto const& p = o.ProblemRef;
    auto str = [&](char const* key) { return r[key].as<std::string>(); };
    auto flag = [&](char const* key) { return r[key].as<bool>(); };
    bool const nsga = o.Program == "operon_nsgp";
    Object doc;
    doc.Set("schema", "operon.resolved_config");
    doc.Set("schema_version", "1.0");
    doc.Set("algorithm", nsga ? "nsga2" : "gp");
    auto const info = GetBuildInfo();
    Object engine;
    engine.Set("program", o.Program);
    engine.Set("revision", info.Revision);
    engine.Set("precision", info.Precision);
    engine.Set("build_type", info.BuildType);
    engine.Set("capabilities", BuildCapabilities(info));
    doc.Add("engine", engine.Finish());
    doc.Set("seed", c.Seed);
    doc.Set("seed_source", r.count("seed") ? "explicit" : "random_device");
    Object threads;
    threads.Set("effective", o.Threads);
    threads.Set("source", r.count("threads") ? "explicit" : "hardware_concurrency");
    doc.Add("threads", threads.Finish());
    Object generator;
    std::string pressure = "null";
    if (auto const* os = dynamic_cast<OffspringSelectionGenerator const*>(&o.Generator)) {
        generator.Set("kind", "os");
        generator.Add("max_selection_pressure", Limit(os->MaxSelectionPressure()));
        generator.Set("comparison_factor", os->ComparisonFactor());
        pressure = Limit(os->MaxSelectionPressure());
    } else if (auto const* brood = dynamic_cast<BroodOffspringGenerator const*>(&o.Generator)) {
        generator.Set("kind", "brood");
        generator.Set("size", brood->BroodSize());
    } else if (auto const* poly = dynamic_cast<PolygenicOffspringGenerator const*>(&o.Generator)) {
        generator.Set("kind", "poly");
        generator.Set("size", poly->PolygenicSize());
    } else {
        generator.Set("kind", "basic");
    }
    Object budgets;
    budgets.Add("generations", Limit(c.Generations));
    budgets.Add("evaluations", "{\"limit\":" + Limit(o.Evaluator.Budget()) + ",\"unit\":\"residual+jacobian\"}");
    budgets.Add("time_limit_s", Limit(c.TimeLimit));
    budgets.Add("pressure_limit", pressure);
    doc.Add("budgets", budgets.Finish());
    doc.Set("population_size", c.PopulationSize);
    doc.Set("pool_size", c.PoolSize);
    Object limits;
    limits.Set("max_length", o.Crossover.MaxLength());
    limits.Set("max_depth", o.Crossover.MaxDepth());
    doc.Add("tree_limits", limits.Finish());
    Object data;
    data.Set("file", str("dataset"));
    data.Set("file_sha256", Sha256FileHex(str("dataset")));
    data.Set("target", p.TargetVariable().Name);
    std::vector<std::string> inputs;
    for (auto hash : p.GetInputs()) {
        inputs.emplace_back(*p.GetDataset()->FindVariableName(hash));
    }
    data.Set("inputs", inputs);
    data.Add("train", RangeJson(p.TrainingRange()));
    data.Add("test", RangeJson(p.TestRange()));
    Object split;
    split.Add("train", RangeJson(p.TrainingRange()));
    split.Add("test", RangeJson(p.TestRange()));
    split.Set("shuffle", flag("shuffle"));
    split.Set("standardize", flag("standardize"));
    data.Add("split", split.Finish());
    doc.Add("data", data.Finish());
    Object objective;
    auto const names = o.Evaluator.ObjectiveNames();
    objective.Set("metric", names.front());
    std::string objectives = "[";
    for (auto const& name : names) {
        if (objectives.size() > 1) {
            objectives += ',';
        }
        objectives += "{\"name\":" + Json(name) + ",\"direction\":\"min\"}";
    }
    objective.Add("objectives", objectives + ']');
    objective.Set("linear_scaling", p.LinearScalingEnabled());
    objective.Set("skip_nonfinite", p.LinearScalingOmitsNonFinite());
    objective.Set("nonfinite_penalty_weight", r["nonfinite-penalty-weight"].as<double>());
    doc.Add("objective", objective.Finish());
    Object optimizer;
    optimizer.Set("kind", "levenberg-marquardt");
    optimizer.Set("iterations", o.Optimizer.Iterations());
    optimizer.Set("local_search_probability", c.LocalSearchProbability);
    optimizer.Set("lamarckian_probability", c.LamarckianProbability);
    doc.Add("optimizer", optimizer.Finish());
    Object creator;
    if (auto const* btc = dynamic_cast<BalancedTreeCreator const*>(&o.Creator)) {
        creator.Set("kind", "btc");
        creator.Set("bias", btc->GetBias());
    } else if (auto const* ptc = dynamic_cast<ProbabilisticTreeCreator const*>(&o.Creator)) {
        creator.Set("kind", "ptc2");
        creator.Set("bias", ptc->GetBias());
    } else {
        creator.Set("kind", "grow");
        creator.Set("bias", 0);
    }
    creator.Set("min_depth", o.CreatorMinDepth);
    creator.Set("max_depth", o.CreatorMaxDepth);
    creator.Set("max_length", o.CreatorMaxLength);
    Object crossover;
    crossover.Set("kind", "subtree");
    crossover.Set("probability", c.CrossoverProbability);
    crossover.Set("internal_probability", o.Crossover.InternalProbability());
    crossover.Set("max_depth", o.Crossover.MaxDepth());
    crossover.Set("max_length", o.Crossover.MaxLength());
    Object mutation;
    mutation.Set("probability", c.MutationProbability);
    mutation.Set("mutators", o.Mutators);
    Object reinserter;
    reinserter.Set("name", o.Reinserter.Name());
    reinserter.Set("elite_count", o.Reinserter.EliteCount());
    Object operators;
    operators.Add("creator", creator.Finish());
    operators.Add("crossover", crossover.Finish());
    operators.Add("mutation", mutation.Finish());
    operators.Add("female_selector", SelectorJson(o.Female));
    operators.Add("male_selector", SelectorJson(o.Male));
    operators.Add("generator", generator.Finish());
    operators.Add("reinserter", reinserter.Finish());
    operators.Set("symbolic", flag("symbolic"));
    if (nsga) {
        operators.Set("epsilon", c.Epsilon);
        operators.Set("sorter", str("sorter"));
        operators.Set("model_selection", str("model-selection"));
        operators.Set("mdl_likelihood", str("mdl-likelihood"));
    }
    doc.Add("operators", operators.Finish());
    Object primitives;
    primitives.Set("base", "arithmetic");
    primitives.Set(
        "enable", r.count("enable-symbols") ? Split(str("enable-symbols"), ',') : std::vector<std::string> {});
    primitives.Set(
        "disable", r.count("disable-symbols") ? Split(str("disable-symbols"), ',') : std::vector<std::string> {});
    std::vector<std::string> enabled;
    for (auto const& node : p.GetPrimitiveSet().EnabledPrimitives()) {
        enabled.emplace_back(node.Name());
    }
    std::ranges::sort(enabled);
    primitives.Set("effective", enabled);
    doc.Add("primitives", primitives.Finish());
    Object cache;
    cache.Set("enabled", c.Cache != nullptr);
    cache.Set("max_age", r["cache-max-age"].as<std::size_t>());
    doc.Add("cache", cache.Finish());
    Object jit;
    jit.Set("mode", str("jit"));
    jit.Set("max_length", r["jit-max-length"].as<int>());
    jit.Set("min_visits", r["jit-min-visits"].as<std::size_t>());
    doc.Add("jit", jit.Finish());
    if (o.Shape) {
        Object shape;
        auto const path = str("shape-constraints-config");
        std::ifstream stream(path);
        if (!stream) {
            throw std::runtime_error("cannot read shape configuration: " + path);
        }
        std::string contents { std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>() };
        glz::generic_i64 parsed;
        if (auto error = glz::read_json(parsed, contents)) {
            throw std::runtime_error(glz::format_error(error, contents));
        }
        shape.Set("enabled", true);
        shape.Set("config", parsed);
        shape.Set("config_sha256", Sha256FileHex(path));
        auto const& policy = o.Shape->Policy;
        constexpr std::array modes { "hard-reject", "penalty", "extra-objective", "penalty,extra-objective" };
        shape.Set("enforcement", modes.at(static_cast<std::size_t>(policy.Enforcement)));
        shape.Set("penalty_weight", policy.PenaltyWeight);
        shape.Set("unknown_violation", policy.UnknownViolation);
        shape.Set("worst_value", o.Shape->WorstValue);
        constexpr std::array backends { "combined", "interval", "affine" };
        Object bound;
        bound.Set("backend", backends.at(static_cast<std::size_t>(o.Shape->Bound.Mode)));
        bound.Set("depth", o.Shape->Bound.BisectionDepth);
        shape.Add("bound", bound.Finish());
        doc.Add("shape", shape.Finish());
    } else {
        doc.Add("shape", "null");
    }
    std::vector<std::string> defaulted;
    auto d = [&](char const* option, char const* path) {
        if (r.count(option) == 0) {
            defaulted.emplace_back(path);
        }
    };
    for (auto const& [option, path] : std::initializer_list<std::pair<char const*, char const*>> { { "seed", "seed" },
             { "threads", "threads.effective" }, { "generations", "budgets.generations" },
             { "evaluations", "budgets.evaluations.limit" }, { "timelimit", "budgets.time_limit_s" },
             { "population-size", "population_size" }, { "pool-size", "pool_size" },
             { "maxlength", "tree_limits.max_length" }, { "maxdepth", "tree_limits.max_depth" },
             { "inputs", "data.inputs" }, { "train", "data.train" }, { "test", "data.test" },
             { "shuffle", "data.split.shuffle" }, { "standardize", "data.split.standardize" },
             { "objective", "objective.metric" }, { "linear-scaling", "objective.linear_scaling" },
             { "skip-nonfinite", "objective.skip_nonfinite" },
             { "nonfinite-penalty-weight", "objective.nonfinite_penalty_weight" },
             { "iterations", "optimizer.iterations" },
             { "local-search-probability", "optimizer.local_search_probability" },
             { "lamarckian-probability", "optimizer.lamarckian_probability" }, { "creator", "operators.creator.kind" },
             { "crossover-probability", "operators.crossover.probability" },
             { "crossover-internal-probability", "operators.crossover.internal_probability" },
             { "mutation-probability", "operators.mutation.probability" },
             { "mutators", "operators.mutation.mutators" }, { "female-selector", "operators.female_selector.kind" },
             { "male-selector", "operators.male_selector.kind" }, { "offspring-generator", "operators.generator.kind" },
             { "reinserter", "operators.reinserter.name" }, { "elitism", "operators.reinserter.elite_count" },
             { "symbolic", "operators.symbolic" }, { "transposition-cache", "cache.enabled" },
             { "cache-max-age", "cache.max_age" }, { "jit", "jit.mode" }, { "jit-max-length", "jit.max_length" },
             { "jit-min-visits", "jit.min_visits" } }) {
        d(option, path);
    }
    auto const ct = Split(str("creator"), ':');
    if (ct.size() < 2 || ct[1].empty()) {
        defaulted.emplace_back("operators.creator.bias");
    }
    if (ct.size() < 3 || ct[2].empty()) {
        d("creator-mindepth", "operators.creator.min_depth");
    }
    if (ct.size() < 4 || ct[3].empty()) {
        d("creator-maxdepth", "operators.creator.max_depth");
    }
    if (ct.size() < 5 || ct[4].empty()) {
        d("maxlength", "operators.creator.max_length");
    }
    for (auto const* sex : { "female", "male" }) {
        auto option = std::string(sex) + "-selector";
        if (Split(str(option.c_str()), ':').size() == 1 && str(option.c_str()) == "tournament") {
            defaulted.emplace_back("operators." + std::string(sex) + "_selector.tournament_size");
        }
    }
    auto const gt = Split(str("offspring-generator"), ':');
    if (gt[0] == "os") {
        if (gt.size() < 2) {
            defaulted.emplace_back("budgets.pressure_limit");
        }
        if (gt.size() < 3) {
            defaulted.emplace_back("operators.generator.comparison_factor");
        }
    } else if ((gt[0] == "brood" || gt[0] == "poly") && gt.size() < 2) {
        defaulted.emplace_back("operators.generator.size");
    }
    if (!r.count("enable-symbols") || !r.count("disable-symbols")) {
        defaulted.emplace_back("primitives");
    }
    if (nsga) {
        d("epsilon", "operators.epsilon");
        d("sorter", "operators.sorter");
        d("model-selection", "operators.model_selection");
        d("mdl-likelihood", "operators.mdl_likelihood");
    }
    if (o.Shape) {
        d("shape-enforcement", "shape.enforcement");
        d("shape-penalty-weight", "shape.penalty_weight");
        d("shape-unknown-violation", "shape.unknown_violation");
        d("shape-worst-value", "shape.worst_value");
        d("shape-bound-mode", "shape.bound");
    }
    doc.Set("defaulted", defaulted);
    std::vector<std::string> ignored;
    for (auto const* key : { "debug", "pareto-front", "model-selection", "mdl-likelihood" }) {
        if (r.count(key) && (std::string_view(key) == "debug" || !nsga)) {
            ignored.emplace_back(key);
        }
    }
    doc.Set("ignored_options", ignored);
    std::vector<std::string> argv;
    for (auto const* arg : o.Argv) {
        argv.emplace_back(arg);
    }
    doc.Set("argv", argv);
    if (r.count("pareto-front") && nsga) {
        Object io;
        io.Set("pareto_front", str("pareto-front"));
        doc.Add("io", io.Finish());
    }
    return doc.Finish();
}

auto ExportResolvedConfig(cxxopts::ParseResult const& result, ResolvedObjects const& objects) -> bool
{
    if (!result.count("print-config") && !result.count("config-out")) {
        return false;
    }
    auto const document = BuildResolvedConfig(result, objects) + '\n';
    if (result.count("config-out")) {
        WriteFileAtomically(result["config-out"].as<std::string>(), document);
    }
    if (result.count("print-config")) {
        fmt::print("{}", document);
        return true;
    }
    return false;
}
} // namespace Operon::Cli
