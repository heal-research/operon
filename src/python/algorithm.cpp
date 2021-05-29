// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "operon.hpp"

#include <pybind11/detail/common.h>

void init_algorithm(py::module_ &m)
{
    py::class_<GeneticProgrammingAlgorithm>(m, "GeneticProgrammingAlgorithm")
        .def(py::init<Operon::Problem const&, Operon::GeneticAlgorithmConfig const&, UniformInitializer&,
                Operon::OffspringGeneratorBase const&, Operon::ReinserterBase const&>())
        .def("Run", py::overload_cast<Operon::RandomGenerator&, std::function<void()>, size_t>(&GeneticProgrammingAlgorithm::Run),
                py::call_guard<py::gil_scoped_release>(), py::arg("rng"), py::arg("callback") = nullptr, py::arg("threads") = 0)
        .def("Reset", &GeneticProgrammingAlgorithm::Reset)
        .def("BestModel", [](GeneticProgrammingAlgorithm const& self, Operon::Comparison const& comparison) {
                    auto min_elem = std::min_element(self.Parents().begin(), self.Parents().end(), [&](auto const& a, auto const& b) { return comparison(a, b);});
                    return *min_elem;
                })
        .def_property_readonly("Generation", &GeneticProgrammingAlgorithm::Generation)
        .def_property_readonly("Parents", static_cast<std::vector<Operon::Individual> const& (GeneticProgrammingAlgorithm::*)() const>(&GeneticProgrammingAlgorithm::Parents))
        .def_property_readonly("Config", &GeneticProgrammingAlgorithm::GetConfig);
}
