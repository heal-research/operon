#include "nnls/nnls.hpp"

namespace Operon {

OptimizerSummary Optimize(Tree& tree, Dataset const& dataset, const gsl::span<const Operon::Scalar> targetValues, Range const range, size_t iterations, bool writeCoefficients, bool report) {
#if defined(CERES_TINY_SOLVER) || !defined(HAVE_CERES)
    Optimizer<DerivativeMethod::AUTODIFF, OptimizerType::TINY> optimizer;
#else
    Optimizer<DerivativeMethod::AUTODIFF, OptimizerType::CERES> optimizer;
#endif
    return optimizer.Optimize(tree, dataset, targetValues, range, iterations, writeCoefficients, report);
}

}
