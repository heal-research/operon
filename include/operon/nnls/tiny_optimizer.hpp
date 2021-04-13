#ifndef OPERON_NNLS_TINY_OPTIMIZER
#define OPERON_NNLS_TINY_OPTIMIZER

#include <Eigen/Core>
#include "interpreter/interpreter.hpp"

#if defined(CERES_TINY_SOLVER)
#include "tiny_solver.h"
#else
#include <ceres/tiny_solver.h>
#endif

namespace Operon {

// simple functor that wraps everything together and provides residuals
struct ResidualEvaluator {
    ResidualEvaluator(Interpreter const& interpreter, Tree const& tree, Dataset const& dataset, const gsl::span<const Operon::Scalar> targetValues, Range const range)
        : interpreter_(interpreter)
        , tree_(tree)
        , dataset_(dataset)
        , range_(range)
        , target_(targetValues)
        , numParameters_(tree_.get().GetCoefficients().size())
    {
    }

    template <typename T>
    bool operator()(T const* const* parameters, T* residuals) const
    {
        gsl::span<T> result(residuals, target_.size());
        GetInterpreter().Evaluate<T>(tree_.get(), dataset_.get(), range_, result, parameters[0]);
        Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1, Eigen::ColMajor>> resMap(residuals, target_.size());
        Eigen::Map<const Eigen::Array<Operon::Scalar, Eigen::Dynamic, 1, Eigen::ColMajor>> targetMap(target_.data(), target_.size());
        resMap -= targetMap.cast<T>();
        return true;
    }

    size_t NumParameters() const { return numParameters_; }
    size_t NumResiduals() const { return target_.size(); }

    Interpreter const& GetInterpreter() const { return interpreter_.get(); }

private:
    std::reference_wrapper<Interpreter const> interpreter_;
    std::reference_wrapper<Tree const> tree_;
    std::reference_wrapper<Dataset const> dataset_;
    Range range_;
    gsl::span<const Operon::Scalar> target_;
    size_t numParameters_; // cache the number of parameters in the tree
};

// this cost function is adapted to work with both solvers from Ceres: the normal one and the tiny solver
// for this, a number of template parameters are necessary:
// - the CostFunctor is the actual functor for computing the residuals
// - the JetT type represents a dual number, the user can specify the type for the Scalar part (float, double) and the Stride (Ceres-specific)
// - the StorageOrder specifies the format of the jacobian (row-major for the big Ceres solver, column-major for the tiny solver)

template <typename CostFunctor, typename JetT, int StorageOrder = Eigen::RowMajor>
struct TinyCostFunction {
    using Scalar = typename JetT::Scalar;
    static constexpr int Stride = JetT::DIMENSION;
    static constexpr int Storage = StorageOrder;

    enum {
        NUM_RESIDUALS = Eigen::Dynamic,
        NUM_PARAMETERS = Eigen::Dynamic,
    };

    TinyCostFunction(CostFunctor const& functor)
        : functor_(functor)
    {
    }

    bool Evaluate(Scalar const* parameters, Scalar* residuals, Scalar* jacobian) const
    {
        if (jacobian == nullptr) {
            return functor_(&parameters, residuals);
        }

        auto numParameters = NumParameters();
        auto numResiduals = NumResiduals();

        // Allocate scratch space for the strided evaluation.
        Operon::Vector<JetT> input_jets(numParameters);
        Operon::Vector<JetT> output_jets(numResiduals);

        auto ptr = &input_jets[0];

        for (int j = 0; j < numParameters; ++j) {
            input_jets[j].a = parameters[j];
        }

        int current_derivative_section = 0;
        int current_derivative_section_cursor = 0;

        Eigen::Map<Eigen::Matrix<Scalar, -1, -1, StorageOrder>> jMap(jacobian, numResiduals, numParameters);

        // Evaluate all of the strides. Each stride is a chunk of the derivative to
        // evaluate, typically some size proportional to the size of the SIMD
        // registers of the CPU.
        int num_strides = static_cast<int>(
            std::ceil(static_cast<float>(numParameters) / static_cast<float>(Stride)));

        for (int pass = 0; pass < num_strides; ++pass) {
            // Set most of the jet components to zero, except for
            // non-constant #Stride parameters.
            const int initial_derivative_section = current_derivative_section;
            const int initial_derivative_section_cursor = current_derivative_section_cursor;

            int active_parameter_count = 0;
            for (int j = 0; j < numParameters; ++j) {
                input_jets[j].v.setZero();
                if (active_parameter_count < Stride && j >= current_derivative_section_cursor) {
                    input_jets[j].v[active_parameter_count] = 1.0;
                    ++active_parameter_count;
                }
            }

            if (!functor_(&ptr, &output_jets[0])) {
                return false;
            }

            active_parameter_count = 0;
            current_derivative_section = initial_derivative_section;
            current_derivative_section_cursor = initial_derivative_section_cursor;

            // Copy the pieces of the jacobians into their final place.
            for (int j = current_derivative_section_cursor; j < numParameters; ++j) {
                if (active_parameter_count < Stride) {
                    for (int k = 0; k < numResiduals; ++k) {
                        jMap(k, j) = output_jets[k].v[active_parameter_count];
                    }
                    ++active_parameter_count;
                    ++current_derivative_section_cursor;
                }
            }

            // Only copy the residuals over once (even though we compute them on every loop).
            if (pass == num_strides - 1) {
                std::transform(output_jets.begin(), output_jets.end(), residuals, [](auto const& jet) { return jet.a; });
            }
        }
        return true;
    }

    // required by tiny solver
    bool operator()(Scalar const* parameters, Scalar* residuals, Scalar* jacobian) const
    {
        return Evaluate(parameters, residuals, jacobian);
    }

    int NumResiduals() const { return functor_.NumResiduals(); }
    int NumParameters() const { return functor_.NumParameters(); }

private:
    CostFunctor functor_;
};
}

#endif
