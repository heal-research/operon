#include "operon.hpp"

void init_crossover(py::module_ &m)
{
    // crossover
    py::class_<Operon::CrossoverBase>(m, "CrossoverBase");

    py::class_<Operon::SubtreeCrossover, Operon::CrossoverBase>(m, "SubtreeCrossover")
        .def(py::init<double, size_t, size_t>(),
                py::arg("internal_probability"),
                py::arg("depth_limit"),
                py::arg("length_limit"))
        .def("__call__", &Operon::SubtreeCrossover::operator())
        .def(py::pickle(
            [](Operon::SubtreeCrossover const& c) {
                return py::make_tuple(c.InternalProbability(), c.MaxDepth(), c.MaxLength());
            },
            [](py::tuple t) {
                if (t.size() != 3) {
                    throw std::runtime_error("Invalid state!");
                }
                double internalProbability = t[0].cast<double>();
                size_t maxDepth = t[1].cast<size_t>();
                size_t maxLength = t[2].cast<size_t>();
                return Operon::SubtreeCrossover(internalProbability, maxDepth, maxLength);
            }
        ));
}
