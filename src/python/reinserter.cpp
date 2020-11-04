#include "operon.hpp"

void init_reinserter(py::module_ &m)
{
    // reinserter
    py::class_<Operon::ReinserterBase>(m, "ReinserterBase");

    py::class_<Operon::ReplaceWorstReinserter<std::execution::parallel_unsequenced_policy>, Operon::ReinserterBase>(m, "ReplaceWorstReinserter")
        .def(py::init([](size_t i) { 
            return Operon::ReplaceWorstReinserter([i](const auto& a, const auto& b) { return a[i] < b[i]; });
                    }), py::arg("objective_index"))
        .def(py::init<Operon::ComparisonCallback>())
        .def("__call__", &Operon::ReplaceWorstReinserter<std::execution::parallel_unsequenced_policy>::operator());

}
