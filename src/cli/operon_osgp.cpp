#include <fmt/core.h>
#include <cxxopts.hpp>

//#include "algorithms/sgp.hpp"
#include "algorithms/osgp.hpp"
#include "operators/initialization.hpp"
#include "operators/crossover.hpp"
#include "operators/mutation.hpp"
#include "operators/selection.hpp"

#include "util.hpp"

using namespace Operon;

int main(int argc, char* argv[])
{
    cxxopts::Options opts("operon_cli", "C++ large-scale genetic programming");

    opts.add_options()
        ("dataset",               "Dataset file name (csv) (required)",                                                                 cxxopts::value<std::string>())
        ("train",                 "Training range specified as start:end (required)",                                                   cxxopts::value<std::string>())
        ("test",                  "Test range specified as start:end",                                                                  cxxopts::value<std::string>())
        ("target",                "Name of the target variable (required)",                                                             cxxopts::value<std::string>())
        ("population-size",       "Population size",                                                                                    cxxopts::value<size_t>()->default_value("1000"))
        ("generations",           "Number of generations",                                                                              cxxopts::value<size_t>()->default_value("1000"))
        ("evaluations",           "Evaluation budget",                                                                                  cxxopts::value<size_t>()->default_value("1000000"))
        ("selection-pressure",    "Selection pressure",                                                                                 cxxopts::value<size_t>()->default_value("100"))
        ("iterations",            "Local optimization iterations",                                                                      cxxopts::value<size_t>()->default_value("50"))
        ("maxlength",             "Maximum length",                                                                                     cxxopts::value<size_t>()->default_value("50"))
        ("maxdepth",              "Maximum depth",                                                                                      cxxopts::value<size_t>()->default_value("12"))
        ("crossover-probability", "The probability to apply crossover",                                                                 cxxopts::value<double>()->default_value("1.0"))
        ("mutation-probability",  "The probability to apply mutation",                                                                  cxxopts::value<double>()->default_value("0.25"))
        ("enable-symbols",        "Comma-separated list of enabled symbols (add, sub, mul, div, exp, log, sin, cos, tan, sqrt, cbrt)",  cxxopts::value<std::string>())
        ("disable-symbols",       "Comma-separated list of disabled symbols (add, sub, mul, div, exp, log, sin, cos, tan, sqrt, cbrt)", cxxopts::value<std::string>())
        ("show-grammar",          "Show grammar (primitive set) used by the algorithm")
        ("debug",                 "Debug mode (more information displayed)")
        ("help",                  "Print help")
    ;
    auto result = opts.parse(argc, argv);
    if (result.arguments().empty() || result.count("help") > 0) 
    {
        fmt::print("{}\n", opts.help());
        return 0;
    }
    if (result.count("dataset") == 0)
    {
        throw std::runtime_error(fmt::format("{}\n{}\n", "Error: no dataset given.", opts.help()));
    }
    if (result.count("target") == 0)
    {
        throw std::runtime_error(fmt::format("{}\n{}\n", "Error: no target variable given.", opts.help()));
    }

    // parse and set default values
    //OffspringSelectionGeneticAlgorithmConfig config;
    OffspringSelectionGeneticAlgorithmConfig config;
    config.Generations          = result["generations"].as<size_t>();
    config.PopulationSize       = result["population-size"].as<size_t>();
    config.Evaluations          = result["evaluations"].as<size_t>();
    config.Iterations           = result["iterations"].as<size_t>();
    config.CrossoverProbability = result["crossover-probability"].as<double>();
    config.MutationProbability  = result["mutation-probability"].as<double>();
    auto maxLength              = result["maxlength"].as<size_t>();
    auto maxDepth               = result["maxdepth"].as<size_t>();

    // parse remaining config options
    Range trainingRange;
    Range testRange;
    std::unique_ptr<Dataset> dataset;
    std::string fileName; // data file name
    std::string target;
    //bool debug = false;
    //bool showGrammar = false;
    GrammarConfig grammarConfig = Grammar::Arithmetic;

    try 
    {
        for (auto kv : result.arguments())
        {
            auto& key   = kv.key();
            auto& value = kv.value();
            if (key == "dataset")
            {
                fileName = value;
                dataset = std::make_unique<Dataset>(fileName, true);
            }
            if (key == "train")
            {
                trainingRange = ParseRange(value);
            }
            if (key == "test")
            {
                testRange = ParseRange(value);
            }
            if (key == "target")
            {
                target = value;
            }
            if (key == "population-size")
            {
                config.PopulationSize = kv.as<size_t>();
            }
            if (key == "selection-pressure")
            {
                config.MaxSelectionPressure = kv.as<size_t>();
            }
            if (key == "generations")
            {
                config.Generations = kv.as<size_t>();
            }
            if (key == "evaluations")
            {
                config.Evaluations = kv.as<size_t>();
            }
            if (key == "iterations")
            {
                config.Iterations = kv.as<size_t>();
            }
            if (key == "maxlength")
            {
                maxLength = kv.as<size_t>();
            }
            if (key == "maxdepth")
            {
                maxDepth = kv.as<size_t>();
            }
            if (key == "enable-symbols")
            {
                auto mask = ParseGrammarConfig(value);
                grammarConfig |= mask;
            }
            if (key == "disable-symbols")
            {
                auto mask = ~ParseGrammarConfig(value);
                grammarConfig &= mask;
            }
            //if (key == "debug")
            //{
            //    debug = true;
            //}
            //if (key == "show-grammar")
            //{
            //    showGrammar = true;
            //}
        }
        if (result.count("train") == 0)
        {
            trainingRange = { 0, 2 * dataset->Rows() / 3 }; // by default use 66% of the data as training 
        }
        // validate training range
        if (trainingRange.Start >= dataset->Rows() || trainingRange.End > dataset->Rows())
        {
            throw std::runtime_error(fmt::format("The training range {}:{} exceeds the available data range ({} rows)\n", trainingRange.Start, trainingRange.End, dataset->Rows()));
        }
        if (trainingRange.Start > trainingRange.End)
        {
            throw std::runtime_error(fmt::format("Invalid training range {}:{}\n", trainingRange.Start, trainingRange.End));
        }
        if (result.count("test") == 0)
        {
            // if no test range is specified, we try to infer a reasonable range based on the trainingRange
            if (trainingRange.Start > 0) 
            {
                testRange = { 0, trainingRange.Start};
            }
            else if (trainingRange.End < dataset->Rows())
            {
                testRange = { trainingRange.End, dataset->Rows() };
            }
            else 
            {
                testRange = { 0, 0 };
            }
        }
        auto seed = std::random_device{}();
        Operon::Random::JsfRand<64> random(seed);

        auto creator             = GrowTreeCreator(maxDepth, maxLength);
        auto crossover           = SubtreeCrossover(0.9, maxDepth, maxLength);

        constexpr bool inPlace   = true; // utilize in-place mutation
        auto mutator             = OnePointMutation<inPlace>();

        auto variables = dataset->Variables();
        std::vector<Variable> inputs;
        std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](const auto& var) { return var.Name != target; });

        auto problem       = Problem(*dataset, inputs, target, trainingRange, testRange);
        problem.GetGrammar().SetConfig(grammarConfig);

        const bool maximization  = true;
        const size_t idx         = 0;

        //TournamentSelector<Individual<1>, idx, maximization> selector(2);
        RandomSelector<Individual<1>, idx, maximization> selector;
        //ProportionalSelector<Individual<1>, idx, maximization> selector;
        //OffspringSelectionGeneticAlgorithm(random, problem, config, creator, selector, crossover, mutator);
        OffspringSelectionGeneticAlgorithm(random, problem, config, creator, selector, crossover, mutator);
    }
    catch(std::exception& e) 
    {
        throw std::runtime_error(e.what());
    }

    return 0;
}

