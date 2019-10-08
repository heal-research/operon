#ifndef METRICS_HPP
#define METRICS_HPP

#include "core/common.hpp"
#include "core/stats.hpp"

namespace Operon {

operon::scalar_t NormalizedMeanSquaredError(gsl::span<const operon::scalar_t> x, gsl::span<const operon::scalar_t> y);
operon::scalar_t MeanSquaredError(gsl::span<const operon::scalar_t> x, gsl::span<const operon::scalar_t> y);
operon::scalar_t RootMeanSquaredError(gsl::span<const operon::scalar_t> x, gsl::span<const operon::scalar_t> y);
operon::scalar_t RSquared(gsl::span<const operon::scalar_t> x, gsl::span<const operon::scalar_t> y);
} // namespace
#endif
