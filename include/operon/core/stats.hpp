#ifndef STATISTICS_HPP
#define STATISTICS_HPP

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

#include "core/common.hpp"

namespace Operon {

template <typename T>
constexpr T eps = std::numeric_limits<T>::epsilon();

template <typename T>
class MeanVarianceCalculator {
public:
    MeanVarianceCalculator() { Reset(); }
    void Add(T x)
    {
        ++n;
        if (n == 1) {
            oldMean = newMean = x;
            oldVar = 0;
        } else {
            newMean = oldMean + (x - oldMean) / n;
            newVar = oldVar + (x - oldMean) * (x - newMean);
            // set up for next iteration
            oldMean = newMean;
            oldVar = newVar;
        }
    }

    template <typename InputIt, typename U = typename InputIt::value_type>
    void Add(InputIt xBegin, InputIt xEnd)
    {
        static_assert(std::is_same_v<T, U>);
        for (; xBegin != xEnd; ++xBegin) {
            Add(*xBegin);
        }
    }

    void Reset()
    {
        newMean = 0;
        oldMean = 0;
        newVar = 0;
        oldVar = 0;
        n = 0;
    }
    T Mean() const { return n > 0 ? newMean : 0.0; }
    T Variance() const { return n > 1 ? newVar / (n - 1) : 0.0; }
    T PopulationVariance() const { return n > 1 ? newVar / n : 0.0; }
    T Stddev() const { return std::sqrt(Variance()); }
    T PopulationStddev() const { return std::sqrt(PopulationVariance()); }

    std::pair<T, T> SampleMeanVariance() const { return { Mean(), Variance() }; }
    std::pair<T, T> PopulationMeanVariance() const { return { Mean(), PopulationVariance() }; }

    template <typename InputIt, typename U = typename InputIt::value_type>
    static std::pair<T, T> CalculateSampleMeanVariance(InputIt xBegin, InputIt xEnd)
    {
        static_assert(std::is_same_v<T, U>);
        MeanVarianceCalculator<T> calc;
        calc.Add(xBegin, xEnd);
        return calc.SampleMeanVariance();
    }

    template <typename InputIt, typename U = typename InputIt::value_type>
    static std::pair<T, T> CalculatePopulationMeanVariance(InputIt xBegin, InputIt xEnd)
    {
        static_assert(std::is_same_v<T, U>);
        MeanVarianceCalculator<T> calc;
        calc.Add(xBegin, xEnd);
        return calc.PopulationMeanVariance();
    }

private:
    T newMean;
    T oldMean;
    T newVar;
    T oldVar;
    size_t n = 0; // number of elements
};

template <typename T>
class CovarianceCalculator {
public:
    CovarianceCalculator()
        : n(0)
        , xMean(0)
        , yMean(0)
        , cn(0)
    {
    }

    T Covariance() { return n > T(0) ? cn / n : T(0); }
    void Reset()
    {
        n = 0;
        xMean = 0;
        yMean = 0;
        cn = 0;
    }
    void Add(T x, T y)
    {
        ++n;
        xMean = xMean + (x - xMean) / n;
        T delta = y - yMean;
        yMean = yMean + delta / n;
        cn = cn + delta * (x - xMean);
    }

    template <typename InputIt1, typename InputIt2, typename U = typename InputIt1::value_type>
    static T Calculate(InputIt1 xBegin, InputIt1 xEnd, InputIt2 yBegin)
    {
        static_assert(std::is_same_v<T, U>);
        CovarianceCalculator<T> calc;
        for (; xBegin != xEnd; ++xBegin, ++yBegin) {
            calc.Add(*xBegin, *yBegin);
        }
        return calc.Covariance();
    }

private:
    size_t n;
    T xMean;
    T yMean;
    T cn;
};

template <typename T>
class PearsonsRCalculator {
public:
    PearsonsRCalculator() { Reset(); }
    void Add(T x, T y)
    {
        sxCalculator.Add(x);
        syCalculator.Add(y);
        covCalculator.Add(x, y);
    }

    void Reset()
    {
        sxCalculator.Reset();
        syCalculator.Reset();
        covCalculator.Reset();
    }

    template <typename InputIt1, typename InputIt2, typename U = typename InputIt1::value_type>
    static T Calculate(InputIt1 xBegin, InputIt1 xEnd, InputIt2 yBegin)
    {
        static_assert(std::is_same_v<T, U>);
        PearsonsRCalculator<T> calc;
        for (; xBegin != xEnd; ++xBegin, ++yBegin) {
            calc.Add(*xBegin, *yBegin);
        }
        return calc.R();
    }
    T R()
    {
        T xvar = sxCalculator.PopulationVariance();
        T yvar = syCalculator.PopulationVariance();
        if (xvar < eps<T> || yvar < eps<T>) {
            return T { 0 };
        }
        return covCalculator.Covariance() / std::sqrt(xvar * yvar);
    }

private:
    MeanVarianceCalculator<T> sxCalculator;
    MeanVarianceCalculator<T> syCalculator;
    CovarianceCalculator<T> covCalculator;
};

// linear scaling parameter calculator
// the reasons for scaling are explained in: http://www2.cs.uidaho.edu/~cs472_572/f11/scaledsymbolicRegression.pdf
template <typename T>
class LinearScalingCalculator {
public:
    LinearScalingCalculator() { Reset(); }
    void Reset()
    {
        tCalculator.Reset();
        ovCalculator.Reset();
        otCalculator.Reset();

        alpha = 0;
        beta = 0;
    }
    void Add(T original, T target)
    {
        tCalculator.Add(target);
        ovCalculator.Add(original);
        otCalculator.Add(original, target);

        //if (ovCalculator.Variance() < eps<T>) beta = 1;
        auto variance = ovCalculator.Variance();
        beta = variance < eps<T> ? 1 : (otCalculator.Covariance() / variance);
        //else beta = otCalculator.Covariance() / ovCalculator.Variance();
        alpha = tCalculator.Mean() - beta * ovCalculator.Mean();
    }
    T Beta() const { return beta; }
    T Alpha() const { return alpha; }

    template <typename InputIt1, typename InputIt2, typename U = typename InputIt1::value_type>
    static std::pair<T, T> Calculate(InputIt1 xBegin, InputIt1 xEnd, InputIt2 yBegin)
    {
        static_assert(std::is_same_v<T, U>);
        LinearScalingCalculator<T> calc;
        for (; xBegin != xEnd; ++xBegin, ++yBegin) {
            calc.Add(*xBegin, *yBegin);
        }
        return { calc.Alpha(), calc.Beta() };
    }

private:
    T alpha; // additive constant
    T beta; // multiplicative factor

    MeanVarianceCalculator<T> tCalculator; // target values
    MeanVarianceCalculator<T> ovCalculator; // original values
    CovarianceCalculator<T> otCalculator; // original-target covariance calculator
};
}
#endif

