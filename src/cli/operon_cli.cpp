#include <fmt/core.h>
#include <cxxopts.hpp>

#include "algorithms/sgp.hpp"
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
        ("d,dataset",             "Dataset file name (csv) (required)",                                                                            cxxopts::value<std::string>())
        ("r,training",            "Training range specified as start:end (required)",                                                              cxxopts::value<std::string>())
        ("s,test",                "Test range specified as start:end",                                                                             cxxopts::value<std::string>())
        ("t,target",              "Name of the target variable (required)",                                                                        cxxopts::value<std::string>())
        ("z,size",                "Population size",                                                                                               cxxopts::value<size_t>()->default_value("100000"))
        ("g,generations",         "Number of generations",                                                                                         cxxopts::value<size_t>()->default_value("1000"))
        ("e,evaluations",         "Evaluation budget",                                                                                             cxxopts::value<size_t>()->default_value("1000000"))
        ("i,iterations",          "Local optimization iterations",                                                                                 cxxopts::value<size_t>()->default_value("50"))
        ("l,length",              "Maximum length",                                                                                                cxxopts::value<size_t>()->default_value("50"))
        ("p,depth",               "Maximum depth",                                                                                                 cxxopts::value<size_t>()->default_value("12"))
        ("m,enable-symbols",      "Comma-separated list of enabled symbols (add, sub, mul, div, exp, log, sin, cos, tan, sqrt, cbrt)",             cxxopts::value<std::string>())
        ("n,disable-symbols",     "Comma-separated list of disabled symbols (add, sub, mul, div, exp, log, sin, cos, tan, sqrt, cbrt)",            cxxopts::value<std::string>())
        ("a,show-grammar",        "Show grammar (primitive set) used by the algorithm")
        ("h,help",                "Print help")
    ;
    auto results = opts.parse(argc, argv);
    if (results.arguments().empty() || results.count("help") > 0) 
    {
        fmt::print("{}\n", opts.help());
        return 0;
    }

    try 
    {
        if (results.count("dataset") == 0)
        {
            throw std::runtime_error(fmt::format("{}\n{}\n", "Error: no dataset given.", opts.help()));
        }
        if (results.count("target") == 0)
        {
            throw std::runtime_error(fmt::format("{}\n{}\n", "Error: no target variable given.", opts.help()));
        }

        Range trainingRange;
        Range testRange;
        std::unique_ptr<Dataset> dataset;
        std::string fileName; // data file name
        std::string target;
        size_t maxLength = 50;
        size_t maxDepth = 12;
        bool debug = false;
        bool showGrammar = false;

        GeneticAlgorithmConfig config;
        config.CrossoverProbability = 1.0;
        config.MutationProbability = 0.25;
        for (auto kv : results.arguments())
        {
            auto& key   = kv.key();
            auto& value = kv.value();
            if (key == "dataset")
            {
                fileName = value;
                dataset = std::make_unique<Dataset>(fileName, true);
            }
            if (key == "training")
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
            if (key == "size")
            {
                config.PopulationSize = kv.as<size_t>();
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
            if (key == "length")
            {
                maxLength = kv.as<size_t>();
            }
            if (key == "depth")
            {
                maxDepth = kv.as<size_t>();
            }
            if (key == "enable-symbols")
            {
//                uint8_t mask = ParseGrammarOptions(value);
//                config |= mask;
            }
            if (key == "disable-symbols")
            {
//                uint8_t mask = ~ParseGrammarOptions(value);
//                config &= mask;
            }
            if (key == "debug")
            {
                debug = true;
            }
            if (key == "show-grammar")
            {
                showGrammar = true;
            }
        }
        if (results.count("training") == 0)
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
        if (results.count("test") == 0)
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
        Operon::Random::JsfRand<64> random;

        auto creator             = GrowTreeCreator(maxDepth, maxLength);
        auto crossover           = SubtreeCrossover(0.9, maxDepth, maxLength);
        auto mutator             = OnePointMutation();

        auto inputs              = dataset->VariableNames();
        inputs.erase(std::remove_if(inputs.begin(), inputs.end(), [&](const std::string& s) { return s == target; }), inputs.end());

        const auto problem       = Problem(*dataset, inputs, target, trainingRange, testRange);

        const bool maximization  = true;
        const size_t idx         = 0;
        const size_t tSize       = 50;

        fmt::print("generations: {}, population: {}, iterations: {}, evaluations: {}, maxDepth: {}, maxLength: {}\n", config.Generations, config.PopulationSize, config.Iterations, config.Evaluations, maxDepth, maxLength);

        TournamentSelector<Individual<1>, idx, maximization> selector(tSize);
        GeneticAlgorithm(random, problem, config, creator, selector, crossover, mutator);
    }
    catch(std::exception& e) 
    {
        throw std::runtime_error(e.what());
    }

    return 0;
}

