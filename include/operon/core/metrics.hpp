#ifndef METRICS_HPP
#define METRICS_HPP

#include <execution>

#include "core/common.hpp"
#include "core/stats.hpp"

namespace Operon
{
    template<typename InputIt1, typename InputIt2, typename T = typename InputIt1::value_type>
    T NormalizedMeanSquaredError(InputIt1 xBegin, InputIt1 xEnd, InputIt2 yBegin)
    {
        MeanVarianceCalculator<T> varCalc;
        MeanVarianceCalculator<T> mseCalc;

        for(; xBegin != xEnd; ++xBegin, ++yBegin)
        {
            auto e = *xBegin;
            auto t = *yBegin;
            if (!std::isnan(t)) 
            {
                varCalc.Add(t);                
            }
            auto err = e - t;
            mseCalc.Add(err * err);
        }
        double var = varCalc.PopulationVariance();
        double mse = mseCalc.Mean();

        return var > 0.0 ? mse / var : 0.0;
    }

    template<typename InputIt1, typename InputIt2, typename T = typename InputIt1::value_type, typename ExecutionPolicy = std::execution::sequenced_policy>
    T MeanSquaredError(InputIt1 xBegin, InputIt1 xEnd, InputIt2 yBegin)
    {
        ExecutionPolicy policy;
        return std::transform_reduce(
                policy,
                xBegin,
                xEnd,
                yBegin,
                T{0}, 
                std::plus<T>{},
                [](auto a, auto b) { return (a-b) * (a-b); }
                ) / std::distance(xBegin, xEnd);
    }

    template<typename InputIt1, typename InputIt2, typename T = typename InputIt1::value_type>
    T RootMeanSquaredError(InputIt1 xBegin, InputIt1 xEnd, InputIt2 yBegin)
    {
        return std::sqrt(MeanSquaredError(xBegin, xEnd, yBegin));
    }

    template<typename InputIt1, typename InputIt2, typename T = typename InputIt1::value_type>
    T RSquared(InputIt1 xBegin, InputIt1 xEnd, InputIt2 yBegin)
    {
        T r = PearsonsRCalculator<T>::Calculate(xBegin, xEnd, yBegin);
        return r * r;
    }
}
#endif

