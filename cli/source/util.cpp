// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include "util.hpp"

#include <memory>
#include <scn/scn.h>

#include "operon/core/node.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/version.hpp"

using Operon::NodeType;

namespace Operon {

static const Operon::Map<std::string, NodeType> Primitives {
    { "add",      NodeType::Add },
    { "mul",      NodeType::Mul },
    { "sub",      NodeType::Sub },
    { "div",      NodeType::Div },
    { "fmin",     NodeType::Fmin },
    { "fmax",     NodeType::Fmax },
    { "aq",       NodeType::Aq },
    { "pow",      NodeType::Pow },
    { "abs",      NodeType::Abs },
    { "acos",     NodeType::Acos },
    { "asin",     NodeType::Asin },
    { "atan",     NodeType::Atan },
    { "cbrt",     NodeType::Cbrt },
    { "ceil",     NodeType::Ceil },
    { "cos",      NodeType::Cos },
    { "cosh",     NodeType::Cosh },
    { "exp",      NodeType::Exp },
    { "floor",    NodeType::Floor },
    { "log",      NodeType::Log },
    { "logabs",   NodeType::Logabs },
    { "log1p",    NodeType::Log1p },
    { "sin",      NodeType::Sin },
    { "sinh",     NodeType::Sinh },
    { "sqrt",     NodeType::Sqrt },
    { "sqrtabs",  NodeType::Sqrtabs },
    { "tan",      NodeType::Tan },
    { "tanh",     NodeType::Tanh },
    { "square",   NodeType::Square },
    { "dyn",      NodeType::Dynamic },
    { "constant", NodeType::Constant },
    { "variable", NodeType::Variable }
};

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
    constexpr std::array<char, 6> sizes{" KMGT"};
    constexpr size_t base{1024};
    auto p = static_cast<size_t>(std::floor(std::log2(bytes) / std::log2(base)));
    return fmt::format("{:.2f} {}b", static_cast<double>(bytes) / std::pow(base, p), sizes.at(p));
}

auto ParseRange(std::string const& str) -> std::pair<size_t, size_t>
{
    size_t a{0};
    size_t b{0};
    (void) scn::scan(str, "{}:{}", a, b);
    return std::make_pair(a, b);
}

auto ParsePrimitiveSetConfig(const std::string& options) -> PrimitiveSetConfig
{
    auto config = static_cast<PrimitiveSetConfig>(0);
    for (auto& s : Split(options, ',')) {
        if (auto it = Primitives.find(s); it != Primitives.end()) {
            config |= it->second;
        } else {
            throw std::runtime_error(fmt::format("Unrecognized symbol {}\n", s));
        }
    }
    return config;
}

auto PrintPrimitives(NodeType config) -> void
{
    PrimitiveSet tmpSet;
    tmpSet.SetConfig(config);
    fmt::print("Built-in primitives:\n");
    fmt::print("{:<8}\t{:<50}\t{:>7}\t\t{:>9}\n", "Symbol", "Description", "Enabled", "Frequency");
    for (size_t i = 0; i < Operon::NodeTypes::Count; ++i) {
        auto type = static_cast<NodeType>(1U << i);
        auto hash = Node(type).HashValue;
        auto enabled = tmpSet.Contains(hash) && tmpSet.IsEnabled(hash);
        auto freq = enabled ? tmpSet.Frequency(hash) : 0U;
        Node node(type);
        fmt::print("{:<8}\t{:<50}\t{:>7}\t\t{:>9}\n", node.Name(), node.Desc(), enabled, freq != 0U ? std::to_string(freq) : "-");
    }
}

auto PrintStats(std::vector<std::tuple<std::string, double, std::string>> const& stats, bool printHeader) -> void
{
    std::vector<size_t> widths;
    auto out = fmt::memory_buffer();
    for (auto const& [name, value, format] : stats) {
        fmt::format_to(std::back_inserter(out), fmt::runtime(fmt::format("{{{}}}", format)), value);
        auto width = std::max(name.size(), fmt::to_string(out).size());
        widths.push_back(width);
        out.clear();
    }
    if (printHeader) {
        for (auto i = 0UL; i < stats.size(); ++i) {
            fmt::print("{} ", fmt::format("{:>{}}", std::get<0>(stats[i]), widths[i]));
        }
        fmt::print("\n");
    }
    for (auto i = 0UL; i < stats.size(); ++i) {
        fmt::format_to(std::back_inserter(out), fmt::runtime(fmt::format("{{{}}}", std::get<2>(stats[i]))), std::get<1>(stats[i]));
        fmt::print("{} ", fmt::format("{:>{}}", fmt::to_string(out), widths[i]));
        out.clear();
    }
    fmt::print("\n");
}

auto InitOptions(std::string const& name, std::string const& desc, int width) -> cxxopts::Options
{
    cxxopts::Options opts(name, desc);
    opts.set_width(width);

    std::string const symbols = "add, sub, mul, div, exp, log, square, sqrt, cbrt, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, abs, aq, ceil, floor, fmin, fmax, log1p, logabs, sqrtabs";

    opts.add_options()
        ("dataset", "Dataset file name (csv) (required)", cxxopts::value<std::string>())
        ("shuffle", "Shuffle the input data", cxxopts::value<bool>()->default_value("false"))
        ("standardize", "Standardize the training partition (zero mean, unit variance)", cxxopts::value<bool>()->default_value("false"))
        ("train", "Training range specified as start:end (required)", cxxopts::value<std::string>())
        ("test", "Test range specified as start:end", cxxopts::value<std::string>())
        ("target", "Name of the target variable (required)", cxxopts::value<std::string>())
        ("inputs", "Comma-separated list of input variables", cxxopts::value<std::string>())
        ("epsilon", "Tolerance for fitness comparison (needed e.g. for eps-dominance)", cxxopts::value<Operon::Scalar>()->default_value("1e-6"))
        ("error-metric", "The error metric used for calculating fitness", cxxopts::value<std::string>()->default_value("r2"))
        ("population-size", "Population size", cxxopts::value<size_t>()->default_value("1000"))
        ("pool-size", "Recombination pool size (how many generated offspring per generation)", cxxopts::value<size_t>()->default_value("1000"))
        ("seed", "Random number seed", cxxopts::value<Operon::RandomGenerator::result_type>()->default_value("0"))
        ("generations", "Number of generations", cxxopts::value<size_t>()->default_value("1000"))
        ("evaluations", "Evaluation budget", cxxopts::value<size_t>()->default_value("1000000"))
        ("iterations", "Local optimization iterations", cxxopts::value<size_t>()->default_value("0"))
        ("selection-pressure", "Selection pressure", cxxopts::value<size_t>()->default_value("100"))
        ("maxlength", "Maximum length", cxxopts::value<size_t>()->default_value("50"))
        ("maxdepth", "Maximum depth", cxxopts::value<size_t>()->default_value("10"))
        ("crossover-probability", "The probability to apply crossover", cxxopts::value<Operon::Scalar>()->default_value("1.0"))
        ("crossover-internal-probability", "Crossover bias towards swapping function nodes", cxxopts::value<Operon::Scalar>()->default_value("0.9"))
        ("mutation-probability", "The probability to apply mutation", cxxopts::value<Operon::Scalar>()->default_value("0.25"))
        ("creator", "Tree creator operator to initialize the population with.", cxxopts::value<std::string>()->default_value("btc"))
        ("creator-mindepth", "Minimum tree depth (applies to the grow tree creator)", cxxopts::value<std::size_t>()->default_value("1"))
        ("creator-maxdepth", "Minimum tree depth (applies to all tree creators)", cxxopts::value<std::size_t>()->default_value("100"))
        ("creator-maxlength", "Maximum tree length (applies to all tree creators)", cxxopts::value<std::size_t>()->default_value("50"))
        ("female-selector", "Female selection operator, with optional parameters separated by : (eg, --selector tournament:5)", cxxopts::value<std::string>()->default_value("tournament"))
        ("male-selector", "Male selection operator, with optional parameters separated by : (eg, --selector tournament:5)", cxxopts::value<std::string>()->default_value("tournament"))
        ("offspring-generator", "OffspringGenerator operator, with optional parameters separated by : (eg --offspring-generator brood:10:10)", cxxopts::value<std::string>()->default_value("basic"))
        ("reinserter", "Reinsertion operator merging offspring in the recombination pool back into the population", cxxopts::value<std::string>()->default_value("keep-best"))
        ("enable-symbols", "Comma-separated list of enabled symbols ("+symbols+")", cxxopts::value<std::string>())
        ("disable-symbols", "Comma-separated list of disabled symbols ("+symbols+")", cxxopts::value<std::string>())
        ("symbolic", "Operate in symbolic mode - no coefficient tuning or coefficient mutation", cxxopts::value<bool>()->default_value("false"))
        ("show-primitives", "Display the primitive set used by the algorithm")
        ("threads", "Number of threads to use for parallelism", cxxopts::value<size_t>()->default_value("0"))
        ("timelimit", "Time limit after which the algorithm will terminate", cxxopts::value<size_t>()->default_value(std::to_string(std::numeric_limits<size_t>::max())))
        ("debug", "Debug mode (more information displayed)")
        ("help", "Print help")
        ("version", "Print version and program information");
    return opts;
}

auto ParseOptions(cxxopts::Options&& opts, int argc, char** argv) -> cxxopts::ParseResult {
    cxxopts::ParseResult result;
    try {
        result = opts.parse(argc, argv);
    } catch (cxxopts::exceptions::parsing const& ex) {
        fmt::print(stderr, "error: {}. rerun with --help to see available options.\n", ex.what());
        std::exit(EXIT_FAILURE);
    }
    if (result.arguments().empty() || result.count("help") > 0) {
        fmt::print("{}\n", opts.help());
        std::exit(EXIT_SUCCESS);
    }
    if (result.count("version") > 0) {
        fmt::print("{}\n", Operon::Version());
        std::exit(EXIT_SUCCESS);
    }
    if (result.count("target") == 0) {
        fmt::print(stderr, "error: no target variable was specified.\n");
        std::exit(EXIT_FAILURE);
    }
    if (result.count("dataset") == 0) {
        fmt::print(stderr, "error: no dataset was specified.\n");
        std::exit(EXIT_FAILURE);
    }
    return result;
}
} // namespace Operon
