// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "interpreter/dispatch_table.hpp"
#include "operators/evaluator.hpp"
#include "operon.hpp"

namespace py = pybind11;

void init_eval(py::module_ &m)
{
    // free functions
    // we use a lambda to avoid defining a fourth arg for the defaulted C++ function arg
    m.def("Evaluate", [](Operon::Interpreter const& i, Operon::Tree const& t, Operon::Dataset const& d, Operon::Range r) {
        auto result = py::array_t<Operon::Scalar>(static_cast<pybind11::ssize_t>(r.Size()));
        auto buf = result.request();
        auto res = Operon::Span<Operon::Scalar>((Operon::Scalar*)buf.ptr, r.Size());
        i.Evaluate(t, d, r, res, static_cast<Operon::Scalar*>(nullptr));
        return result;
        }, py::arg("interpreter"), py::arg("tree"), py::arg("dataset"), py::arg("range"));

    m.def("CalculateFitness", [](Operon::Interpreter const& i, Operon::Tree const& t, Operon::Dataset const& d, Operon::Range r, std::string const& target, std::string const& metric) {
        auto estimated = i.Evaluate(t, d, r, static_cast<Operon::Scalar*>(nullptr));
        auto values = d.GetValues(target).subspan(r.Start(), r.Size());

        if (metric == "rsquared") return Operon::RSquared<Operon::Scalar>(estimated, values); 
        if (metric == "mse")      return Operon::MeanSquaredError<Operon::Scalar>(estimated, values);
        if (metric == "rmse")     return Operon::RootMeanSquaredError<Operon::Scalar>(estimated, values);
        if (metric == "nmse")     return Operon::NormalizedMeanSquaredError<Operon::Scalar>(estimated, values);
        throw std::runtime_error("Invalid fitness metric"); 

    }, py::arg("interpreter"), py::arg("tree"), py::arg("dataset"), py::arg("range"), py::arg("target"), py::arg("metric") = "rsquared");

    m.def("CalculateFitness", [](Operon::Interpreter const& i, std::vector<Operon::Tree> const& trees, Operon::Dataset const& d, Operon::Range r, std::string const& target, std::string const& metric) {
        std::add_pointer<double(Operon::Span<const Operon::Scalar>, Operon::Span<const Operon::Scalar>)>::type func;
        if (metric == "rsquared")  func = &Operon::RSquared;
        else if (metric == "mse")  func = &Operon::MeanSquaredError;
        else if (metric == "nmse") func = &Operon::NormalizedMeanSquaredError;
        else if (metric == "rmse") func = &Operon::RootMeanSquaredError;
        else throw std::runtime_error("Unsupported error metric");

        auto result = py::array_t<double>(static_cast<pybind11::ssize_t>(trees.size()));
        auto buf = result.request();
        auto values = d.GetValues(target).subspan(r.Start(), r.Size());

        // TODO: make this run in parallel with taskflow
        std::transform(trees.begin(), trees.end(), (double*)buf.ptr, [&](auto const& t) -> double {
            auto estimated = i.Evaluate(t, d, r, (Operon::Scalar*)nullptr);
            return func(estimated, values);
        });

        return result;
    }, py::arg("interpreter"), py::arg("trees"), py::arg("dataset"), py::arg("range"), py::arg("target"), py::arg("metric") = "rsquared");

    m.def("FitLeastSquares", [](py::array_t<float> lhs, py::array_t<float> rhs) {
        auto s1 = MakeConstSpan(lhs);
        auto s2 = MakeConstSpan(rhs);
        auto stats = bivariate::accumulate<float>(s1.data(), s2.data(), s1.size());
        auto a = stats.covariance / stats.variance_x; // scale
        if (!std::isfinite(a)) { a = 1; }
        auto b = stats.mean_y - a * stats.mean_x;     // offset
        return std::make_pair(a, b);
    });

    m.def("FitLeastSquares", [](py::array_t<double> lhs, py::array_t<double> rhs) {
        auto s1 = MakeConstSpan(lhs);
        auto s2 = MakeConstSpan(rhs);
        auto stats = bivariate::accumulate<float>(s1.data(), s2.data(), s1.size());
        auto a = stats.covariance / stats.variance_x; // scale
        if (!std::isfinite(a)) { a = 1; }
        auto b = stats.mean_y - a * stats.mean_x;     // offset
        return std::make_pair(a, b);
    });

    m.def("RSquared", [](py::array_t<float> lhs, py::array_t<float> rhs) {
        return Operon::RSquared<float>(MakeSpan(lhs), MakeSpan(rhs));
    });

    m.def("RSquared", [](py::array_t<double> lhs, py::array_t<double> rhs) {
        return Operon::RSquared<double>(MakeSpan(lhs), MakeSpan(rhs));
    });

    m.def("NormalizedMeanSquaredError", [](py::array_t<float> lhs, py::array_t<float> rhs) {
        return Operon::NormalizedMeanSquaredError<float>(MakeSpan(lhs), MakeSpan(rhs));
    });

    m.def("NormalizedMeanSquaredError", [](py::array_t<double> lhs, py::array_t<double> rhs) {
        return Operon::NormalizedMeanSquaredError<double>(MakeSpan(lhs), MakeSpan(rhs));
    });

    m.def("RootMeanSquaredError", [](py::array_t<float> lhs, py::array_t<float> rhs) {
        return Operon::RootMeanSquaredError<float>(MakeSpan(lhs), MakeSpan(rhs));
    });

    m.def("RootMeanSquaredError", [](py::array_t<double> lhs, py::array_t<double> rhs) {
        return Operon::RootMeanSquaredError<double>(MakeSpan(lhs), MakeSpan(rhs));
    });

    m.def("MeanSquaredError", [](py::array_t<float> lhs, py::array_t<float> rhs) {
        return Operon::MeanSquaredError<float>(MakeSpan(lhs), MakeSpan(rhs));
    });

    m.def("MeanSquaredError", [](py::array_t<double> lhs, py::array_t<double> rhs) {
        return Operon::MeanSquaredError<double>(MakeSpan(lhs), MakeSpan(rhs));
    });

    m.def("MeanAbsoluteError", [](py::array_t<float> lhs, py::array_t<float> rhs) {
        return Operon::MeanAbsoluteError<float>(MakeSpan(lhs), MakeSpan(rhs));
    });

    m.def("MeanAbsoluteError", [](py::array_t<double> lhs, py::array_t<double> rhs) {
        return Operon::MeanAbsoluteError<double>(MakeSpan(lhs), MakeSpan(rhs));
    });

    // dispatch table
    py::class_<Operon::DispatchTable>(m, "DispatchTable")
        .def(py::init<>());

    // interpreter
    py::class_<Operon::Interpreter>(m, "Interpreter")
        .def(py::init<>())
        .def(py::init<Operon::DispatchTable>());

    // evaluator
    py::class_<Operon::EvaluatorBase>(m, "EvaluatorBase")
        .def_property("LocalOptimizationIterations", &Operon::EvaluatorBase::GetLocalOptimizationIterations, &Operon::EvaluatorBase::SetLocalOptimizationIterations)
        .def_property("Budget",&Operon::EvaluatorBase::GetBudget, &Operon::EvaluatorBase::SetBudget)
        .def_property_readonly("FitnessEvaluations", &Operon::EvaluatorBase::FitnessEvaluations)
        .def_property_readonly("LocalEvaluations", &Operon::EvaluatorBase::LocalEvaluations)
        .def_property_readonly("TotalEvaluations", &Operon::EvaluatorBase::TotalEvaluations);

    py::class_<Operon::MeanSquaredErrorEvaluator, Operon::EvaluatorBase>(m, "MeanSquaredErrorEvaluator")
        .def(py::init<Operon::Problem&, Operon::Interpreter&>())
        .def("__call__", &Operon::MeanSquaredErrorEvaluator::operator());

    py::class_<Operon::RootMeanSquaredErrorEvaluator, Operon::EvaluatorBase>(m, "RootMeanSquaredErrorEvaluator")
        .def(py::init<Operon::Problem&, Operon::Interpreter&>())
        .def("__call__", &Operon::RootMeanSquaredErrorEvaluator::operator());

    py::class_<Operon::NormalizedMeanSquaredErrorEvaluator, Operon::EvaluatorBase>(m, "NormalizedMeanSquaredErrorEvaluator")
        .def(py::init<Operon::Problem&, Operon::Interpreter&>())
        .def("__call__", &Operon::NormalizedMeanSquaredErrorEvaluator::operator());

    py::class_<Operon::MeanAbsoluteErrorEvaluator, Operon::EvaluatorBase>(m, "MeanAbsoluteErrorEvaluator")
        .def(py::init<Operon::Problem&, Operon::Interpreter&>())
        .def("__call__", &Operon::MeanAbsoluteErrorEvaluator::operator());

    py::class_<Operon::RSquaredEvaluator, Operon::EvaluatorBase>(m, "RSquaredEvaluator")
        .def(py::init<Operon::Problem&, Operon::Interpreter&>())
        .def("__call__", &Operon::RSquaredEvaluator::operator());

    py::class_<Operon::UserDefinedEvaluator, Operon::EvaluatorBase>(m, "UserDefinedEvaluator")
        .def(py::init<Operon::Problem&, std::function<typename Operon::EvaluatorBase::ReturnType(Operon::RandomGenerator*, Operon::Individual&)> const&>())
        .def("__call__", &Operon::UserDefinedEvaluator::operator());

    py::class_<Operon::LengthEvaluator, Operon::EvaluatorBase>(m, "LengthEvaluator")
        .def(py::init<Operon::Problem&>())
        .def("__call__", &Operon::LengthEvaluator::operator());

    py::class_<Operon::ShapeEvaluator, Operon::EvaluatorBase>(m, "ShapeEvaluator")
        .def(py::init<Operon::Problem&>())
        .def("__call__", &Operon::ShapeEvaluator::operator());

    py::class_<Operon::MultiEvaluator, Operon::EvaluatorBase>(m, "MultiEvaluator")
        .def(py::init<Operon::Problem&>())
        .def("Add", &Operon::MultiEvaluator::Add)
        .def("__call__", &Operon::MultiEvaluator::operator());
}
