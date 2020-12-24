#ifndef OPERON_NNLS_TINY_OPTIMIZER
#define OPERON_NNLS_TINY_OPTIMIZER

#include <Eigen/Core>

#include "core/eval.hpp"
#include "tiny_solver.h"

namespace Operon {
// this cost function is adapted to work with both solvers from Ceres: the normal one and the tiny solver
// for this, a number of template parameters are necessary:
// - the CostFunctor is the actual functor for computing the residuals
// - the JetT type represents a dual number, the user can specify the type for the Scalar part (float, double) and the Stride (Ceres-specific)
// - the StorageOrder specifies the format of the jacobian (row-major for the big Ceres solver, column-major for the tiny solver)

template <typename CostFunctor, typename JetT, int StorageOrder = Eigen::RowMajor>
struct TinyCostFunction {
    using Scalar = typename JetT::Scalar;
    const int Stride = JetT::DIMENSION;

    enum {
        NUM_RESIDUALS = Eigen::Dynamic,
        NUM_PARAMETERS = Eigen::Dynamic,
    };

    TinyCostFunction(const Tree& tree, const Dataset& dataset, const gsl::span<const Operon::Scalar> targetValues, const Range range)
        : functor_(tree, dataset, targetValues, range)
    {
        numResiduals_ = targetValues.size();
        numParameters_ = tree.GetCoefficients().size();
    }

    bool Evaluate(Scalar const* parameters, Scalar* residuals, Scalar* jacobian) const
    {
        if (jacobian == nullptr) {
            return functor_(&parameters, residuals);
        }

        // Allocate scratch space for the strided evaluation.
        Operon::Vector<JetT> input_jets(numParameters_);
        Operon::Vector<JetT> output_jets(numResiduals_);

        auto ptr = &input_jets[0];

        for (int j = 0; j < numParameters_; ++j) {
            input_jets[j].a = parameters[j];
        }

        int current_derivative_section = 0;
        int current_derivative_section_cursor = 0;

        Eigen::Map<Eigen::Matrix<Scalar, -1, -1, StorageOrder>> jMap(jacobian, numResiduals_, numParameters_);

        // Evaluate all of the strides. Each stride is a chunk of the derivative to
        // evaluate, typically some size proportional to the size of the SIMD
        // registers of the CPU.
        int num_strides = static_cast<int>(
            std::ceil(static_cast<float>(numParameters_) / static_cast<float>(Stride)));

        for (int pass = 0; pass < num_strides; ++pass) {
            // Set most of the jet components to zero, except for
            // non-constant #Stride parameters.
            const int initial_derivative_section = current_derivative_section;
            const int initial_derivative_section_cursor = current_derivative_section_cursor;

            int active_parameter_count = 0;
            for (int j = 0; j < numParameters_; ++j) {
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
            for (int j = current_derivative_section_cursor; j < numParameters_; ++j) {
                if (active_parameter_count < Stride) {
                    for (int k = 0; k < numResiduals_; ++k) {
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

    int NumResiduals() const { return numResiduals_; }
    int NumParameters() const { return numParameters_; }

private:
    CostFunctor functor_;
    int numResiduals_;
    int numParameters_;
};

enum class OptimizerType : int { TINY, CERES };
enum class DerivativeMethod : int { NUMERIC, AUTODIFF };

struct OptimizerSummary {
    double InitialCost;
    double FinalCost;
    int Iterations;
};

// simple optimizer base class
struct OptimizerBase {
    virtual OptimizerSummary Optimize(Tree& tree, Dataset const& dataset, const gsl::span<const Operon::Scalar> targetValues, Range const range, size_t iterations, bool writeCoefficients, bool report) const = 0;
    virtual ~OptimizerBase() { }
};

template <DerivativeMethod = DerivativeMethod::AUTODIFF, OptimizerType T = OptimizerType::TINY>
struct Optimizer : public OptimizerBase {
    virtual OptimizerSummary Optimize(Tree& tree, Dataset const& dataset, const gsl::span<const Operon::Scalar> targetValues, Range const range, size_t iterations = 50, bool writeCoefficients = true, bool = false /* not used by tiny solver */) const override
    {
        Operon::TinyCostFunction<ResidualEvaluator, Dual, Eigen::ColMajor> cf(tree, dataset, targetValues, range);
        ceres::TinySolver<decltype(cf)> solver;
        solver.options.max_num_iterations = static_cast<int>(iterations);

        auto x0 = tree.GetCoefficients();
        decltype(solver)::Parameters params = Eigen::Map<Eigen::Matrix<Operon::Scalar, Eigen::Dynamic, 1>>(x0.data(), x0.size()).cast<typename decltype(cf)::Scalar>();
        solver.Solve(cf, &params);

        if (writeCoefficients) {
            tree.SetCoefficients({ params.data(), x0.size() });
        }

        OptimizerSummary summary;
        summary.InitialCost = solver.summary.initial_cost;
        summary.FinalCost = solver.summary.final_cost;
        summary.Iterations = solver.summary.iterations;
        return summary;
    }
};
}

#endif
