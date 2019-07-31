#ifndef STATISTICS_HPP
#define STATISTICS_HPP

#include <vector>
#include <algorithm>
#include <memory>
#include <cmath>
#include <numeric>

namespace Operon {

    constexpr double eps = std::numeric_limits<double>::epsilon();

    class MeanVarianceCalculator {
        public:
            MeanVarianceCalculator() { Reset(); }
            void Add(double x) 
            {
                ++n;
                if (n == 1) 
                {
                    oldMean = newMean = x;
                    oldVar  = 0;
                }
                else {
                    newMean = oldMean + (x - oldMean) / n;
                    newVar  = oldVar + (x - oldMean) * (x - newMean);
                    // set up for next iteration
                    oldMean = newMean;
                    oldVar  = newVar;
                }
            }
            template<typename InputIt> void Add(InputIt xBegin, InputIt xEnd)
            {
                for(; xBegin < xEnd; ++xBegin)
                {
                    Add(*xBegin);
                }
            }
            void   Reset()                    { n = 0;                                  }
            double Mean() const               { return n > 0 ? newMean : 0.0;           }
            double Variance() const           { return  n > 1 ? newVar / (n - 1) : 0.0; }
            double PopulationVariance() const { return n > 1 ? newVar / n : 0.0;        }
            double Stddev() const             { return std::sqrt(Variance());           }
            double PopulationStddev() const   { return std::sqrt(PopulationVariance()); }

            template<typename InputIt>
                static std::pair<double, double> CalculateSampleMeanVariance(InputIt xBegin, InputIt xEnd)
                {
                    MeanVarianceCalculator calc;
                    calc.Add(xBegin, xEnd);
                    return { calc.Mean(), calc.Variance() };
                }

            template<typename InputIt>
                static std::pair<double, double> CalculatePopulationMeanVariance(InputIt xBegin, InputIt xEnd)
                {
                    MeanVarianceCalculator calc;
                    calc.Add(xBegin, xEnd);
                    return { calc.Mean(), calc.PopulationVariance() };
                }

        private:
            double newMean = 0, oldMean = 0, newVar = 0, oldVar = 0;
            int n = 0; // number of elements
    };

    class CovarianceCalculator {
        public:
            CovarianceCalculator() { Reset(); }

            double Covariance() { return n > 0 ? cn / n : 0.0;  }
            void   Reset()      { n = xMean = yMean = cn = 0.0; }
            void   Add(double x, double y)
            {
                ++n;
                xMean        = xMean + (x - xMean) / n;
                double delta = y - yMean;
                yMean        = yMean + delta / n;
                cn           = cn + delta * (x - xMean);
            }
            template<typename InputIt1, typename InputIt2>
                static double Calculate(InputIt1 xBegin, InputIt1 xEnd, InputIt2 yBegin)
                {
                    CovarianceCalculator calc;
                    for(; xBegin < xEnd; ++xBegin, ++yBegin)
                    {
                        calc.Add(*xBegin, *yBegin);
                    }
                    return calc.Covariance();
                }
            static double Calculate(const std::vector<double>& first, const std::vector<double>& second)
            {
                return Calculate(first.begin(), first.end(), second.begin());
            }
        private:
            double xMean, yMean, cn;
            int n;
    };

    class PearsonsRCalculator {
        public:
            PearsonsRCalculator() { Reset(); } 
            void Add(double x, double y) 
            {
                sxCalculator.Add(x);
                syCalculator.Add(y);
                covCalculator.Add(x,y);
            }
            void Reset() 
            {
                sxCalculator.Reset();
                syCalculator.Reset();
                covCalculator.Reset();
            }
            template<typename InputIt1, typename InputIt2>
                static double Calculate(InputIt1 xBegin, InputIt1 xEnd, InputIt2 yBegin)
                {
                    PearsonsRCalculator calc;
                    // the two vectors should have the same size
                    for(; xBegin < xEnd; ++xBegin, ++yBegin)
                    {
                        calc.Add(*xBegin, *yBegin);
                    }
                    return calc.R();
                }
            static double Calculate(const std::vector<double>& first, const std::vector<double>& second) 
            {
                return Calculate(first.begin(), first.end(), second.begin());
            }
            double R() 
            { 
                double xvar = sxCalculator.PopulationVariance();
                double yvar = syCalculator.PopulationVariance();
                if( xvar < eps || yvar < eps)  { return 0.0; }
                return covCalculator.Covariance() / std::sqrt(xvar * yvar);
            }
        private:
            MeanVarianceCalculator sxCalculator;
            MeanVarianceCalculator syCalculator;
            CovarianceCalculator   covCalculator;
    };

    // linear scaling parameter calculator
    // the reasons for scaling are explained in: http://www2.cs.uidaho.edu/~cs472_572/f11/scaledsymbolicRegression.pdf
    class LinearScalingCalculator {
        public: 
            LinearScalingCalculator() { Reset(); }
            void Reset() {
                tCalculator.Reset();
                ovCalculator.Reset();
                otCalculator.Reset();
            }
            void Add(double original, double target) {
                tCalculator.Add(target);
                ovCalculator.Add(original);
                otCalculator.Add(original, target);

                if (ovCalculator.Variance() < eps) beta = 1;
                else beta = otCalculator.Covariance() / ovCalculator.Variance();
                alpha = tCalculator.Mean() - beta * ovCalculator.Mean(); 
            }
            double Beta() const { return beta; }
            double Alpha() const { return alpha; }
        private:
            double alpha; // additive constant 
            double beta; // multiplicative factor

            MeanVarianceCalculator tCalculator;
            MeanVarianceCalculator ovCalculator; // original values
            CovarianceCalculator   otCalculator; // original - target covariance calculator
    };

    // error measures
    template<typename InputIt1, typename InputIt2, typename T = typename InputIt1::value_type> inline double MeanSquaredError(InputIt1 xBegin, InputIt1 xEnd, InputIt2 yBegin) 
    {
        auto size = std::distance(xBegin, xEnd);
        auto mse  = std::inner_product(xBegin, xEnd, yBegin, T(0.), std::plus<T>(), [](T a, T b) { return (a - b) * (a - b); }) / size;
        return mse;
    }

    template<typename InputIt1, typename InputIt2, typename T = typename InputIt1::value_type> inline double NormalizedMeanSquaredError(InputIt1 xBegin, InputIt1 xEnd, InputIt2 yBegin)
    {
        MeanVarianceCalculator varCalc;
        MeanVarianceCalculator mseCalc;

        for(; xBegin != xEnd; ++xBegin, ++yBegin)
        {
            auto e = *xBegin;
            auto t = *yBegin;
            if (!std::isnan(t)) 
            {
                varCalc.Add(t);                
            }
            double err;
            err = e - t;
            mseCalc.Add(err * err);
        }
        double var = varCalc.PopulationVariance();
        double mse = mseCalc.Mean();

        return var > 0 ? mse / var : 0.0;
    }

    template<typename InputIt1, typename InputIt2, typename T = typename InputIt1::value_type> inline double RSquared(InputIt1 xBegin, InputIt1 xEnd, InputIt2 yBegin)
    {
        PearsonsRCalculator calc;
        for (; xBegin != xEnd; ++xBegin, ++yBegin)
        {
            calc.Add(*xBegin, *yBegin);
        }
        auto r = calc.R();
        return r * r;
    }

    template<typename InputIt1, typename InputIt2, typename T = typename InputIt1::value_type> inline double RootMeanSquaredError(InputIt1 xBegin, InputIt1 xEnd, InputIt2 yBegin)
    {
        return std::sqrt(MeanSquaredError(xBegin, xEnd, yBegin));
    }

    template<typename T> inline double MeanSquaredError(const std::vector<T>& estimated, const std::vector<T>& original)
    {
        return MeanSquaredError(estimated.begin(), estimated.end(), original.begin());
    }

    template<typename T> inline double NormalizedMeanSquaredError(const std::vector<T>& estimated, const std::vector<T>& original)
    {
        return NormalizedMeanSquaredError(estimated.begin(), estimated.end(), original.begin());
    }

    template<typename T> inline double RootMeanSquaredError(const std::vector<T>& estimated, const std::vector<T>& target)
    {
        return RootMeanSquaredError(estimated.begin(), estimated.end(), target.begin()); 
    }

    template<typename T> inline double RSquared(const std::vector<T>& estimated, const std::vector<T>& original)
    {
        return RSquared(estimated.begin(), estimated.end(), original.begin());
    }
}
#endif

