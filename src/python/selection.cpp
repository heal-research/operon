#include "operon.hpp"

void init_selection(py::module_ &m)
{
    // selection
    py::class_<Operon::SelectorBase>(m, "SelectorBase");

    py::class_<Operon::TournamentSelector, Operon::SelectorBase>(m, "TournamentSelector")
        .def(py::init([](size_t i){ 
                    return Operon::TournamentSelector([i](const auto& a, const auto& b) { return a[i] < b[i]; });
                    }), py::arg("objective_index"))
        .def(py::init<Operon::ComparisonCallback const&>())
        .def("__call__", &Operon::TournamentSelector::operator())
        .def_property("TournamentSize", &Operon::TournamentSelector::GetTournamentSize, &Operon::TournamentSelector::SetTournamentSize);

    py::class_<Operon::RankTournamentSelector, Operon::SelectorBase>(m, "RankTournamentSelector")
        .def(py::init([](size_t i){
                    return Operon::RankTournamentSelector([i](const auto& a, const auto& b) { return a[i] < b[i]; });
                    }), py::arg("objective_index"))
        .def(py::init<Operon::ComparisonCallback const&>())
        .def("__call__", &Operon::RankTournamentSelector::operator())
        .def("Prepare", &Operon::RankTournamentSelector::Prepare)
        .def_property("TournamentSize", &Operon::RankTournamentSelector::GetTournamentSize, &Operon::RankTournamentSelector::SetTournamentSize);

    py::class_<Operon::ProportionalSelector, Operon::SelectorBase>(m, "ProportionalSelector")
        .def(py::init([](size_t i){
                    return Operon::ProportionalSelector([i](const auto& a, const auto& b) { return a[i] < b[i]; });
                    }), py::arg("objective_index"))
        .def(py::init<Operon::ComparisonCallback const&>())
        .def("__call__", &Operon::ProportionalSelector::operator())
        .def("Prepare", py::overload_cast<const gsl::span<const Operon::Individual>>(&Operon::ProportionalSelector::Prepare, py::const_))
        .def("SetObjIndex", &Operon::ProportionalSelector::SetObjIndex);

    py::class_<Operon::RandomSelector, Operon::SelectorBase>(m, "RandomSelector")
        .def(py::init<>())
        .def("__call__", &Operon::RandomSelector::operator())
        .def("Prepare", py::overload_cast<const gsl::span<const Operon::Individual>>(&Operon::RandomSelector::Prepare, py::const_));

}
