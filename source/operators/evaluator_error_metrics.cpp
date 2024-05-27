#include "operon/operators/evaluator.hpp"
#include "operon/error_metrics/error_metrics.hpp"

namespace Operon {
    auto ErrorMetric::operator()(Operon::Span<Operon::Scalar const> x, Operon::Span<Operon::Scalar const> y) const -> double {
        switch (type_) {
        case ErrorType::SSE: return SumOfSquaredErrors(x, y);
        case ErrorType::MSE: return MeanSquaredError(x, y);
        case ErrorType::NMSE: return NormalizedMeanSquaredError(x, y);
        case ErrorType::RMSE: return RootMeanSquaredError(x, y);
        case ErrorType::MAE: return MeanAbsoluteError(x, y);
        case ErrorType::R2: return -R2Score(x, y);
        case ErrorType::C2: return -SquaredCorrelation(x, y);
        default: { throw std::runtime_error("unknown error type"); }
        }
    }

    auto ErrorMetric::operator()(Operon::Span<Operon::Scalar const> x, Operon::Span<Operon::Scalar const> y, Operon::Span<Operon::Scalar const> w) const -> double {
        switch (type_) {
        case ErrorType::SSE: return SumOfSquaredErrors(x, y, w);
        case ErrorType::MSE: return MeanSquaredError(x, y, w);
        case ErrorType::NMSE: return NormalizedMeanSquaredError(x, y, w);
        case ErrorType::RMSE: return RootMeanSquaredError(x, y, w);
        case ErrorType::MAE: return MeanAbsoluteError(x, y, w);
        case ErrorType::R2: return -R2Score(x, y, w);
        case ErrorType::C2: return -SquaredCorrelation(x, y, w);
        default: { throw std::runtime_error("unknown error type"); }
        }
    }

    auto ErrorMetric::operator()(Iterator beg1, Iterator end1, Iterator beg2) const -> double {
        switch (type_) {
        case ErrorType::SSE: return SumOfSquaredErrors(beg1, end1, beg2);
        case ErrorType::MSE: return MeanSquaredError(beg1, end1, beg2);
        case ErrorType::NMSE: return NormalizedMeanSquaredError(beg1, end1, beg2);
        case ErrorType::RMSE: return RootMeanSquaredError(beg1, end1, beg2);
        case ErrorType::MAE: return MeanAbsoluteError(beg1, end1, beg2);
        case ErrorType::R2: return -R2Score(beg1, end1, beg2);
        case ErrorType::C2: return -SquaredCorrelation(beg1, end1, beg2);
        default: { throw std::runtime_error("unknown error type"); }
        }
    }

    auto ErrorMetric::operator()(Iterator beg1, Iterator end1, Iterator beg2, Iterator beg3) const -> double {
        switch (type_) {
        case ErrorType::SSE: return SumOfSquaredErrors(beg1, end1, beg2, beg3);
        case ErrorType::MSE: return MeanSquaredError(beg1, end1, beg2, beg3);
        case ErrorType::NMSE: return NormalizedMeanSquaredError(beg1, end1, beg2, beg3);
        case ErrorType::RMSE: return RootMeanSquaredError(beg1, end1, beg2, beg3);
        case ErrorType::MAE: return MeanAbsoluteError(beg1, end1, beg2, beg3);
        case ErrorType::R2: return -R2Score(beg1, end1, beg2, beg3);
        case ErrorType::C2: return -SquaredCorrelation(beg1, end1, beg2, beg3);
        default: { throw std::runtime_error("unknown error type"); }
        }
    }
}  // namespace Operon