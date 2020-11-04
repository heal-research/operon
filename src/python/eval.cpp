
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <gsl/span>
#include <stat/linearscaler.hpp>

#include "core/metrics.hpp"
#include "core/eval.hpp"

#include "operon.hpp"

namespace py = pybind11;

template<typename T> 
gsl::span<T> MakeSpan(py::array_t<T> arr)
{
    py::buffer_info info = arr.request();
    using size_type = gsl::span<const Operon::Scalar>::size_type;
    return gsl::span<T>(static_cast<T*>(info.ptr), static_cast<size_type>(info.size));
}

void init_eval(py::module_ &m)
{
    // free functions
    // we use a lambda to avoid defining a fourth arg for the defaulted C++ function arg
    m.def("Evaluate", [](Operon::Tree const& t, Operon::Dataset const& d, Operon::Range r) {
        auto result = py::array_t<Operon::Scalar>(static_cast<pybind11::ssize_t>(r.Size()));
        auto buf = result.request();
        auto res = gsl::span<Operon::Scalar>((Operon::Scalar*)buf.ptr, r.Size());
        Operon::Evaluate(t, d, r, res, static_cast<Operon::Scalar*>(nullptr));
        return result;
        }, py::arg("tree"), py::arg("dataset"), py::arg("range"));

    m.def("CalculateFitness", [](Operon::Tree const& t, Operon::Dataset const& d, Operon::Range r, std::string const& target, std::string const& metric) {
        auto estimated = Operon::Evaluate(t, d, r, static_cast<Operon::Scalar*>(nullptr));
        auto values = d.GetValues(target).subspan(r.Start(), r.Size());

        if (metric == "rsquared") return Operon::RSquared<Operon::Scalar>(estimated, values); 
        if (metric == "mse")      return Operon::MeanSquaredError<Operon::Scalar>(estimated, values);
        if (metric == "rmse")     return Operon::RootMeanSquaredError<Operon::Scalar>(estimated, values);
        if (metric == "nmse")     return Operon::NormalizedMeanSquaredError<Operon::Scalar>(estimated, values);
        throw std::runtime_error("Invalid fitness metric"); 

    }, py::arg("tree"), py::arg("dataset"), py::arg("range"), py::arg("target"), py::arg("metric") = "rsquared");

    m.def("CalculateFitness", [](std::vector<Operon::Tree> const& trees, Operon::Dataset const& d, Operon::Range r, std::string const& target, std::string const& metric) {
        std::add_pointer<double(gsl::span<const Operon::Scalar>, gsl::span<const Operon::Scalar>)>::type func;
        if (metric == "rsquared")  func = Operon::RSquared;
        else if (metric == "mse")  func = Operon::MeanSquaredError;
        else if (metric == "nmse") func = Operon::NormalizedMeanSquaredError;
        else if (metric == "rmse") func = Operon::RootMeanSquaredError;
        else throw std::runtime_error("Unsupported error metric");

        auto result = py::array_t<double>(static_cast<pybind11::ssize_t>(trees.size()));
        auto buf = result.request();
        auto values = d.GetValues(target).subspan(r.Start(), r.Size());

        std::transform(std::execution::par, trees.begin(), trees.end(), (double*)buf.ptr, [&](auto const& t) -> double {
            auto estimated = Operon::Evaluate(t, d, r, (Operon::Scalar*)nullptr);
            return func(estimated, values);
        });

        return result;
    }, py::arg("trees"), py::arg("dataset"), py::arg("range"), py::arg("target"), py::arg("metric") = "rsquared");

    m.def("FitLeastSquares", [](py::array_t<float> lhs, py::array_t<float> rhs) {
        auto s1 = MakeSpan(lhs);
        auto s2 = MakeSpan(rhs);
        return Operon::LinearScalingCalculator::Calculate(s1.begin(), s1.end(), s2.begin());
    });

    m.def("FitLeastSquares", [](py::array_t<double> lhs, py::array_t<double> rhs) {
        auto s1 = MakeSpan(lhs);
        auto s2 = MakeSpan(rhs);
        return Operon::LinearScalingCalculator::Calculate(s1.begin(), s1.end(), s2.begin());
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

    // evaluator
    py::class_<Operon::EvaluatorBase>(m, "EvaluatorBase")
        .def_property("LocalOptimizationIterations", &Operon::EvaluatorBase::GetLocalOptimizationIterations, &Operon::EvaluatorBase::SetLocalOptimizationIterations)
        .def_property("Budget",&Operon::EvaluatorBase::GetBudget, &Operon::EvaluatorBase::SetBudget)
        .def_property_readonly("FitnessEvaluations", &Operon::EvaluatorBase::FitnessEvaluations)
        .def_property_readonly("LocalEvaluations", &Operon::EvaluatorBase::LocalEvaluations)
        .def_property_readonly("TotalEvaluations", &Operon::EvaluatorBase::TotalEvaluations);

    py::class_<Operon::RSquaredEvaluator, Operon::EvaluatorBase>(m, "RSquaredEvaluator")
        .def(py::init<Operon::Problem&>())
        .def("__call__", &Operon::RSquaredEvaluator::operator());

    py::class_<Operon::NormalizedMeanSquaredErrorEvaluator, Operon::EvaluatorBase>(m, "NormalizedMeanSquaredErrorEvaluator")
        .def(py::init<Operon::Problem&>())
        .def("__call__", &Operon::NormalizedMeanSquaredErrorEvaluator::operator());

}
