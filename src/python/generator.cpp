#include "operon.hpp"

void init_generator(py::module_ &m)
{
    // offspring generator base
    py::class_<Operon::OffspringGeneratorBase>(m, "OffspringGeneratorBase")
        .def_property_readonly("Terminate", &Operon::OffspringGeneratorBase::Terminate);

    // basic offspring generator
    py::class_<Operon::BasicOffspringGenerator, Operon::OffspringGeneratorBase>(m, "BasicOffspringGenerator")
        .def(py::init<Operon::EvaluatorBase&, Operon::CrossoverBase&, Operon::MutatorBase&,
                Operon::SelectorBase&, Operon::SelectorBase&>())
        .def("__call__", &Operon::BasicOffspringGenerator::operator(),
            py::arg("rng"),
            py::arg("crossover_probability"),
            py::arg("mutation_probability")
        )
        .def("Prepare", [](Operon::BasicOffspringGenerator& self, std::vector<Operon::Individual> const& individuals) {
            gsl::span<const Operon::Individual> s(individuals.data(), individuals.size());
            self.Prepare(s);
        })
        .def("__call__", [](Operon::BasicOffspringGenerator& self, Operon::RandomGenerator& rng, double pc, double pm, size_t n) {
            std::vector<Operon::Individual> v;
            v.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                if (auto res = self(rng, pc, pm); res.has_value())
                    v.push_back(res.value());
            }
            return v;
        });

    // offspring selection generator
    py::class_<Operon::OffspringSelectionGenerator, Operon::OffspringGeneratorBase>(m, "OffspringSelectionGenerator")
        .def(py::init<Operon::EvaluatorBase&, Operon::CrossoverBase&, Operon::MutatorBase&,
                Operon::SelectorBase&, Operon::SelectorBase&>())
        .def("__call__", &Operon::OffspringSelectionGenerator::operator(),
            py::arg("rng"),
            py::arg("crossover_probability"),
            py::arg("mutation_probability")
        )
        .def("Prepare", [](Operon::OffspringSelectionGenerator& self, std::vector<Operon::Individual> const& individuals) {
            gsl::span<const Operon::Individual> s(individuals.data(), individuals.size());
            self.Prepare(s);
        })
        .def("__call__", [](Operon::OffspringSelectionGenerator& self, Operon::RandomGenerator& rng, double pc, double pm, size_t n) {
            std::vector<Operon::Individual> v;
            v.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                if (auto res = self(rng, pc, pm); res.has_value())
                    v.push_back(res.value());
            }
            return v;
        })
        .def_property("MaxSelectionPressure",
                py::overload_cast<>(&Operon::OffspringSelectionGenerator::MaxSelectionPressure, py::const_), // getter
                py::overload_cast<size_t>(&Operon::OffspringSelectionGenerator::MaxSelectionPressure)        // setter
                )
        .def_property("ComparisonFactor",
                py::overload_cast<>(&Operon::OffspringSelectionGenerator::ComparisonFactor, py::const_), // getter
                py::overload_cast<double>(&Operon::OffspringSelectionGenerator::ComparisonFactor)        // setter
                )
        .def_property_readonly("SelectionPressure", &Operon::OffspringSelectionGenerator::SelectionPressure);

    // brood generator
    py::class_<Operon::BroodOffspringGenerator, Operon::OffspringGeneratorBase>(m, "BroodOffspringGenerator")
        .def(py::init<Operon::EvaluatorBase&, Operon::CrossoverBase&, Operon::MutatorBase&,
                Operon::SelectorBase&, Operon::SelectorBase&>())
        .def("__call__", &Operon::BroodOffspringGenerator::operator(),
            py::arg("rng"),
            py::arg("crossover_probability"),
            py::arg("mutation_probability")
        )
        .def("Prepare", [](Operon::BroodOffspringGenerator& self, std::vector<Operon::Individual> const& individuals) {
            gsl::span<const Operon::Individual> s(individuals.data(), individuals.size());
            self.Prepare(s);
        })
        .def("__call__", [](Operon::BroodOffspringGenerator& self, Operon::RandomGenerator& rng, double pc, double pm, size_t n) {
            std::vector<Operon::Individual> v;
            v.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                if (auto res = self(rng, pc, pm); res.has_value())
                    v.push_back(res.value());
            }
            return v;
        })
        .def_property("BroodSize",
                py::overload_cast<>(&Operon::BroodOffspringGenerator::BroodSize, py::const_), // getter
                py::overload_cast<size_t>(&Operon::BroodOffspringGenerator::BroodSize)        // setter
                );

    // polygenic generator
    py::class_<Operon::PolygenicOffspringGenerator, Operon::OffspringGeneratorBase>(m, "PolygenicOffspringGenerator")
        .def(py::init<Operon::EvaluatorBase&, Operon::CrossoverBase&, Operon::MutatorBase&,
                Operon::SelectorBase&, Operon::SelectorBase&>())
        .def("__call__", &Operon::PolygenicOffspringGenerator::operator(),
            py::arg("rng"),
            py::arg("crossover_probability"),
            py::arg("mutation_probability")
        )
        .def("Prepare", [](Operon::PolygenicOffspringGenerator& self, std::vector<Operon::Individual> const& individuals) {
            gsl::span<const Operon::Individual> s(individuals.data(), individuals.size());
            self.Prepare(s);
        })
        .def("__call__", [](Operon::PolygenicOffspringGenerator& self, Operon::RandomGenerator& rng, double pc, double pm, size_t n) {
            std::vector<Operon::Individual> v;
            v.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                if (auto res = self(rng, pc, pm); res.has_value())
                    v.push_back(res.value());
            }
            return v;
        })
        .def_property("BroodSize",
                py::overload_cast<>(&Operon::PolygenicOffspringGenerator::PolygenicSize, py::const_), // getter
                py::overload_cast<size_t>(&Operon::PolygenicOffspringGenerator::PolygenicSize)        // setter
                );
}

