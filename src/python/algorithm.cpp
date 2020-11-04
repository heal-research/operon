#include "operon.hpp"

void init_algorithm(py::module_ &m)
{
    py::class_<GeneticProgrammingAlgorithm>(m, "GeneticProgrammingAlgorithm")
        .def(py::init<Operon::Problem const&, Operon::GeneticAlgorithmConfig const&, UniformInitializer&,
                Operon::OffspringGeneratorBase const&, Operon::ReinserterBase const&>())
        .def("Run", &GeneticProgrammingAlgorithm::Run, py::call_guard<py::gil_scoped_release>(), py::arg("rng"), py::arg("callback"), py::arg("threads") = 0)
        .def("BestModel", [](GeneticProgrammingAlgorithm const& self, Operon::Comparison const& comparison) {
                    auto min_elem = std::min_element(self.Parents().begin(), self.Parents().end(), [&](auto const& a, auto const& b) { return comparison(a, b);});
                    return *min_elem;
                })
        .def_property_readonly("Generation", &GeneticProgrammingAlgorithm::Generation);
}
