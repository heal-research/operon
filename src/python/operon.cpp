// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "operon.hpp"

#include "core/version.hpp"
#include "parser/infix.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_operon, m)
{
    m.doc() = "Operon Python Module";
    m.attr("__version__") = 0.1;

    // binding code
    py::bind_vector<std::vector<Operon::Variable>>(m, "VariableCollection");
    py::bind_vector<std::vector<Operon::Individual>>(m, "IndividualCollection");

    init_algorithm(m);
    init_creator(m);
    init_crossover(m);
    init_dataset(m);
    init_eval(m);
    init_generator(m);
    init_mutation(m);
    init_non_dominated_sorter(m);
    init_node(m);
    init_problem(m);
    init_pset(m);
    init_reinserter(m);
    init_selection(m);
    init_tree(m);

    // build information
    m.def("Version", &Operon::Version);

    // random numbers
    m.def("UniformInt", &Operon::Random::Uniform<Operon::RandomGenerator, int>);
    m.def("UniformReal", &Operon::Random::Uniform<Operon::RandomGenerator, double>);

    // classes
    py::class_<Operon::Individual>(m, "Individual")
        .def(py::init<>())
        .def(py::init<size_t>())
        .def("__getitem__", py::overload_cast<size_t>(&Operon::Individual::operator[]))
        .def("__getitem__", py::overload_cast<size_t>(&Operon::Individual::operator[], py::const_))
        .def_readwrite("Genotype", &Operon::Individual::Genotype)
        .def("SetFitness", [](Operon::Individual& self, Operon::Scalar f, size_t i) { self[i] = f; })
        .def("GetFitness", [](Operon::Individual& self, size_t i) { return self[i]; });

    py::class_<Operon::Comparison>(m, "Comparison");

    py::class_<Operon::SingleObjectiveComparison, Operon::Comparison>(m, "SingleObjectiveComparison")
        .def(py::init<size_t>())
        .def("__call__", &Operon::SingleObjectiveComparison::operator());

    py::class_<Operon::CrowdedComparison, Operon::Comparison>(m, "CrowdedComparison")
        .def(py::init<>())
        .def("__call__", &Operon::CrowdedComparison::operator());

    py::class_<Operon::Variable>(m, "Variable")
        .def_readwrite("Name", &Operon::Variable::Name)
        .def_readwrite("Hash", &Operon::Variable::Hash)
        .def_readwrite("Index", &Operon::Variable::Index);

    py::class_<Operon::Range>(m, "Range")
        .def(py::init<size_t, size_t>())
        .def(py::init<std::pair<size_t, size_t>>())
        .def_property_readonly("Start", &Operon::Range::Start)
        .def_property_readonly("End", &Operon::Range::End)
        .def_property_readonly("Size", &Operon::Range::Size);

    // random generators
    py::class_<Operon::Random::RomuTrio>(m, "RomuTrio")
        .def(py::init<uint64_t>())
        .def("__call__", &Operon::Random::RomuTrio::operator());

    py::class_<Operon::Random::Sfc64>(m, "Sfc64")
        .def(py::init<uint64_t>())
        .def("__call__", &Operon::Random::Sfc64::operator());

    // tree format
    py::class_<Operon::TreeFormatter>(m, "TreeFormatter")
        .def_static("Format", py::overload_cast<Operon::Tree const&, Operon::Dataset const&, int>(&Operon::TreeFormatter::Format))
        .def_static("Format", py::overload_cast<Operon::Tree const&, std::unordered_map<Operon::Hash, std::string> const&, int>(&Operon::TreeFormatter::Format));

    py::class_<Operon::InfixFormatter>(m, "InfixFormatter")
        .def_static("Format", py::overload_cast<Operon::Tree const&, Operon::Dataset const&, int>(&Operon::InfixFormatter::Format))
        .def_static("Format", py::overload_cast<Operon::Tree const&, std::unordered_map<Operon::Hash, std::string> const&, int>(&Operon::InfixFormatter::Format));

    py::class_<Operon::InfixParser>(m, "InfixParser")
        .def_static("Parse", &Operon::InfixParser::Parse<std::unordered_map<std::string, Operon::Hash>>);

    // genetic algorithm
    py::class_<Operon::GeneticAlgorithmConfig>(m, "GeneticAlgorithmConfig")
        .def_readwrite("Generations", &Operon::GeneticAlgorithmConfig::Generations)
        .def_readwrite("Evaluations", &Operon::GeneticAlgorithmConfig::Evaluations)
        .def_readwrite("Iterations", &Operon::GeneticAlgorithmConfig::Iterations)
        .def_readwrite("PopulationSize", &Operon::GeneticAlgorithmConfig::PopulationSize)
        .def_readwrite("PoolSize", &Operon::GeneticAlgorithmConfig::PoolSize)
        .def_readwrite("CrossoverProbability", &Operon::GeneticAlgorithmConfig::CrossoverProbability)
        .def_readwrite("MutationProbability", &Operon::GeneticAlgorithmConfig::MutationProbability)
        .def_readwrite("Seed", &Operon::GeneticAlgorithmConfig::Seed)
        .def_readwrite("TimeLimit", &Operon::GeneticAlgorithmConfig::TimeLimit)
        .def(py::init([](size_t gen, size_t evals, size_t iter, size_t popsize, size_t poolsize, double pc, double pm, size_t seed, size_t timelimit) {
                    Operon::GeneticAlgorithmConfig config;
                    config.Generations = gen;
                    config.Evaluations = evals;
                    config.Iterations = iter;
                    config.PopulationSize = popsize;
                    config.PoolSize = poolsize;
                    config.CrossoverProbability = pc;
                    config.MutationProbability = pm;
                    config.Seed = seed;
                    config.TimeLimit = timelimit;
                    return config;
        }), py::arg("generations")
          , py::arg("max_evaluations")
          , py::arg("local_iterations")
          , py::arg("population_size")
          , py::arg("pool_size")
          , py::arg("p_crossover")
          , py::arg("p_mutation")
          , py::arg("seed")
          , py::arg("time_limit"));
}
