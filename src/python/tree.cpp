
#include "operon.hpp"

void init_tree(py::module_ &m)
{
    // tree
    py::class_<Operon::Tree>(m, "Tree")
        .def(py::init<std::initializer_list<Operon::Node>>())
        .def(py::init<Operon::Vector<Operon::Node>>())
        .def(py::init<const Operon::Tree&>())
        //.def(py::init<Operon::Tree&&>())
        .def("UpdateNodes", &Operon::Tree::UpdateNodes)
        .def("Sort", &Operon::Tree::Sort)
        .def("Hash", static_cast<Operon::Tree& (Operon::Tree::*)(Operon::HashFunction, Operon::HashMode)>(&Operon::Tree::Hash))
        .def("Reduce", &Operon::Tree::Reduce)
        //.def("Simplify", &Operon::Tree::Simplify) // not yet implemented
        .def("ChildIndices", &Operon::Tree::ChildIndices)
        .def("SetEnabled", &Operon::Tree::SetEnabled)
        .def("SetCoefficients", &Operon::Tree::SetCoefficients)
        .def("GetCoefficients", &Operon::Tree::GetCoefficients)
        .def("CoefficientsCount", &Operon::Tree::CoefficientsCount)
        .def_property_readonly("Nodes", static_cast<Operon::Vector<Operon::Node>& (Operon::Tree::*)()&>(&Operon::Tree::Nodes))
        .def_property_readonly("Nodes", static_cast<Operon::Vector<Operon::Node> const& (Operon::Tree::*)() const&>(&Operon::Tree::Nodes))
        //.def_property_readonly("Nodes", static_cast<Operon::Vector<Operon::Node>&& (Operon::Tree::*)() &&>(&Operon::Tree::Nodes))
        .def_property_readonly("Length", &Operon::Tree::Length)
        .def_property_readonly("VisitationLength", &Operon::Tree::VisitationLength)
        .def_property_readonly("Depth", static_cast<size_t (Operon::Tree::*)() const>(&Operon::Tree::Depth))
        .def_property_readonly("Empty", &Operon::Tree::Empty)
        .def_property_readonly("HashValue", &Operon::Tree::HashValue)
        .def("__getitem__", py::overload_cast<size_t>(&Operon::Tree::operator[]))
        .def("__getitem__", py::overload_cast<size_t>(&Operon::Tree::operator[], py::const_))
        .def(py::pickle(
            [](Operon::Tree const& tree) {
                return py::make_tuple(tree.Nodes());
            },
            [](py::tuple t) {
                if (t.size() != 1) {
                    throw std::runtime_error("Invalid state!");
                }

                return Operon::Tree(t[0].cast<Operon::Vector<Operon::Node>>()).UpdateNodes();
            }
        ));

}
