#include "operon.hpp"
#include <operators/mutation.hpp>

void init_mutation(py::module_ &m)
{
    // mutation
    py::class_<Operon::MutatorBase>(m, "MutatorBase");

    py::class_<Operon::OnePointMutation, Operon::MutatorBase>(m, "OnePointMutation")
        .def(py::init<>())
        .def("__call__", &Operon::OnePointMutation::operator())
        .def(py::pickle(
            [](Operon::OnePointMutation const&) {
                return py::make_tuple(); // no state nothing to pickle

            },
            [](py::tuple) {
                return Operon::OnePointMutation();
            }
        ));

    py::class_<Operon::ChangeVariableMutation, Operon::MutatorBase>(m, "ChangeVariableMutation")
        .def(py::init([](std::vector<Operon::Variable> const& variables) {
                    return Operon::ChangeVariableMutation(gsl::span<const Operon::Variable>(variables.data(), variables.size()));
                }),
            py::arg("variables"))
        .def("__call__", &Operon::ChangeVariableMutation::operator());

    py::class_<Operon::ChangeFunctionMutation, Operon::MutatorBase>(m, "ChangeFunctionMutation")
        .def(py::init<Operon::PrimitiveSet>())
        .def("__call__", &Operon::ChangeFunctionMutation::operator());

    py::class_<Operon::ReplaceSubtreeMutation, Operon::MutatorBase>(m, "ReplaceSubtreeMutation")
        .def(py::init<Operon::CreatorBase&, size_t, size_t>())
        .def("__call__", &Operon::ReplaceSubtreeMutation::operator());

    py::class_<Operon::RemoveSubtreeMutation, Operon::MutatorBase>(m, "RemoveSubtreeMutation")
        .def(py::init<Operon::PrimitiveSet>())
        .def("__call__", &Operon::RemoveSubtreeMutation::operator());

    py::class_<Operon::InsertSubtreeMutation, Operon::MutatorBase>(m, "InsertSubtreeMutation")
        .def(py::init<Operon::CreatorBase&, size_t, size_t>())
        .def("__call__", &Operon::InsertSubtreeMutation::operator());

    py::class_<Operon::MultiMutation, Operon::MutatorBase>(m, "MultiMutation")
        .def(py::init<>())
        .def("__call__", &Operon::MultiMutation::operator())
        .def("Add", &Operon::MultiMutation::Add)
        .def_property_readonly("Count", &Operon::MultiMutation::Count);

}
