// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "operon.hpp"

namespace py = pybind11;

void init_non_dominated_sorter(py::module_ &m)
{
    py::class_<Operon::NondominatedSorterBase>(m, "NonDominatedSorterBase");

    py::class_<Operon::RankSorter, Operon::NondominatedSorterBase>(m, "RankSorter")
        .def(py::init<>())
        .def("Sort", &Operon::RankSorter::Sort);
}
