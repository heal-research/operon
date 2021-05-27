// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "operon.hpp"
#include "operators/reinserter/keepbest.hpp"
#include "operators/reinserter/replaceworst.hpp"

void init_reinserter(py::module_ &m)
{
    // reinserter
    py::class_<Operon::ReinserterBase>(m, "ReinserterBase");

    py::class_<Operon::ReplaceWorstReinserter, Operon::ReinserterBase>(m, "ReplaceWorstReinserter")
        .def(py::init([](size_t i) { 
            return Operon::ReplaceWorstReinserter([i](const auto& a, const auto& b) { return a[i] < b[i]; });
                    }), py::arg("objective_index"))
        .def(py::init<Operon::ComparisonCallback>())
        .def("__call__", &Operon::ReplaceWorstReinserter::operator());

    py::class_<Operon::KeepBestReinserter, Operon::ReinserterBase>(m, "KeepBestReinserter")
        .def(py::init([](size_t i) {
            return Operon::KeepBestReinserter([i](const auto& a, const auto& b) { return a[i] < b[i]; });
                    }), py::arg("objective_index"))
        .def(py::init<Operon::ComparisonCallback>())
        .def("__call__", &Operon::KeepBestReinserter::operator());
}
