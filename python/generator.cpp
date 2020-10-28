#include "pyoperon.hpp"

void init_generator(py::module_ &m)
{
    // offspring generator
    py::class_<Operon::OffspringGeneratorBase>(m, "OffspringGeneratorBase")
        .def_property_readonly("Terminate", &Operon::OffspringGeneratorBase::Terminate);
    
    py::class_<Operon::BasicOffspringGenerator, Operon::OffspringGeneratorBase>(m, "BasicOffspringGenerator")
        .def(py::init<Operon::EvaluatorBase&, Operon::CrossoverBase&, Operon::MutatorBase&, 
                Operon::SelectorBase&, Operon::SelectorBase&>())
        .def("__call__", &Operon::BasicOffspringGenerator::operator())
        .def("Prepare", [](Operon::BasicOffspringGenerator& self, std::vector<Operon::Individual> const& individuals) {
            gsl::span<const Operon::Individual> s(individuals.data(), individuals.size());
            self.Prepare(s);
        })
        .def("__call__", [](Operon::BasicOffspringGenerator& self, Operon::RandomGenerator& rng, double pc, double pm,size_t n) {
            std::vector<Operon::Individual> v;
            v.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                if (auto res = self(rng, pc, pm); res.has_value())
                    v.push_back(res.value());
            }
            return v;
        });
}

