#ifndef EVALUATE_HPP
#define EVALUATE_HPP

#include <ceres/ceres.h>
#include "tree.hpp"
#include "dataset.hpp"

#define FOR(i) for(size_t i = 0; i < BATCHSIZE; ++i)

namespace Operon {
    constexpr size_t BATCHSIZE  = 64;
    constexpr int    JET_STRIDE = 4;
    using Dual = ceres::Jet<double, JET_STRIDE>;

    // When auto-vectorizing without __restrict,
    // gcc and clang check for overlap (with a bunch of integer code)
    // before running the vectorized loop

    // vector operations
    template<typename T> inline void add(T* __restrict a,    T const* __restrict b) noexcept { FOR(i) a[i] += b[i];                          }
    template<typename T> inline void sub(T* __restrict a,    T const* __restrict b) noexcept { FOR(i) a[i] -= b[i];                          }
    template<typename T> inline void mul(T* __restrict a,    T const* __restrict b) noexcept { FOR(i) a[i] *= b[i];                          }
    template<typename T> inline void div(T* __restrict a,    T const* __restrict b) noexcept { FOR(i) a[i] /= b[i];                          }
    template<typename T> inline void inv(T* __restrict a,    T const* __restrict b) noexcept { FOR(i) a[i] = 1. / b[i];                      }
    template<typename T> inline void neg(T* __restrict a,    T const* __restrict b) noexcept { FOR(i) a[i] = -b[i];                          }
    template<typename T> inline void exp(T* __restrict a,    T const* __restrict b) noexcept { FOR(i) a[i] = ceres::exp(b[i]);               }
    template<typename T> inline void log(T* __restrict a,    T const* __restrict b) noexcept { FOR(i) a[i] = ceres::log(b[i]);               }
    template<typename T> inline void sin(T* __restrict a,    T const* __restrict b) noexcept { FOR(i) a[i] = ceres::sin(b[i]);               }
    template<typename T> inline void cos(T* __restrict a,    T const* __restrict b) noexcept { FOR(i) a[i] = ceres::cos(b[i]);               }
    template<typename T> inline void tan(T* __restrict a,    T const* __restrict b) noexcept { FOR(i) a[i] = ceres::tan(b[i]);               }
    template<typename T> inline void sqrt(T* __restrict a,   T const* __restrict b) noexcept { FOR(i) a[i] = ceres::sqrt(b[i]);              }
    template<typename T> inline void cbrt(T* __restrict a,   T const* __restrict b) noexcept { FOR(i) a[i] = ceres::cbrt(b[i]);              }
    template<typename T> inline void square(T* __restrict a, T const* __restrict b) noexcept { FOR(i) a[i] = ceres::pow(b[i], 2.);           }
    template<typename T> inline void abs(T* __restrict a,    T const* __restrict b) noexcept { FOR(i) a[i] = ceres::abs(b[i]);               }
    template<typename T> inline void aq(T* __restrict a,     T const* __restrict b) noexcept { FOR(i) a[i] /= ceres::sqrt(b[i] * b[i] + 1.); }

    template<typename T> inline bool isnan(T a) noexcept                                     { return ceres::IsNaN(a);                       }
    template<typename T> inline bool isinf(T a) noexcept                                     { return ceres::IsInfinite(a);                  }
    template<typename T> inline bool isfinite(T a) noexcept                                  { return ceres::IsFinite(a);                    }

    // vector - scalar operations
    template<typename T> inline void load(T* __restrict a, T const b) noexcept               { std::fill_n(a, BATCHSIZE, b);                 }
    template<typename T> inline void load(T* __restrict a, T const* __restrict b) noexcept   { std::copy_n(b, BATCHSIZE, a);                 }

    template<typename T> inline std::pair<T, T> MinMax(T* values, size_t count) noexcept
    {
        auto min = T(0), max = T(0);
        for (size_t i = 0; i < count; ++i)
        {
            auto& v = values[i];
            if (!isfinite(v)) continue;
            if (min > v) min = v;
            if (max < v) max = v;
        }
        return { min, max };
    }

    template<typename T> inline void LimitToRange(T* values, T min, T max, size_t count) noexcept
    {
        for (size_t i = 0; i < count; ++i)
        {
            auto& v = values[i];
            if (!isfinite(v)) { v = (min + max) / 2.0; }
            if (v < min) { v = min; }
            if (v > max) { v = max; }
        }
    }

    template<typename T>
    std::vector<T> Evaluate(const Tree& tree, const Dataset& dataset, const Range range, T const* const parameters = nullptr)
    {
        std::vector<T> result(range.Size());
        Evaluate(tree, dataset, range, parameters, result.data());
        return result;
    }

    template<typename T>
    void Evaluate(const Tree& tree, const Dataset& dataset, const Range range, T const* const parameters, T* result)
    {
        auto& nodes = tree.Nodes();
        std::vector<T> buffer(nodes.size() * BATCHSIZE); // intermediate results buffer

        // fill in buf for constants and non-terminal nodes
        size_t idx    = 0;
        for (size_t i = 0; i < nodes.size(); ++i)
        {
            auto buf = buffer.data() + i * BATCHSIZE;
            auto& s = nodes[i];
            if (s.IsConstant() || s.IsVariable())
            {
                auto v = parameters == nullptr ? T(s.Value) : parameters[idx++];
                load(buf, v);
            }
        }

        size_t numRows = range.Size();
        for (size_t row = 0; row < numRows; row += BATCHSIZE)
        {
            auto remainingRows = std::min(BATCHSIZE, numRows - row);
            for (size_t i = 0; i < nodes.size(); ++i)
            {
                auto& s = nodes[i];
                auto buf = buffer.data() + i * BATCHSIZE;
                switch (s.Type)
                {
                    case NodeType::Variable:
                        {
                            auto& values = dataset.GetValues(s.CalculatedHashValue);
                            auto start = values.begin() + range.Start + row;
                            //if constexpr (std::is_same_v<T, Dual>)
                            //{
                            //    // for dual numbers we need to assign values manually
                            //    std::transform(start, start + remainingRows, buf, [](double v) { return T(v); });
                            //}
                            //else
                            //{
                            //    // otherwise we just do a memcpy
                            //    std::copy_n(start, remainingRows, buf);
                            //}
                            std::transform(start, start + remainingRows, [&](double v) { return T(s.Value * v); });
                            break;
                        }
                    case NodeType::Add:
                        {
                            auto c = i - 1;             // first child index
                            load(buf, buf - BATCHSIZE); // load child buffer
                            for (size_t k = 1, j = c - 1 - nodes[c].Length; k < s.Arity; ++k, j -= 1 + nodes[j].Length)
                            {
                                add(buf, buffer.data() + j * BATCHSIZE);
                            }
                            break;
                        }
                    case NodeType::Mul:
                        {
                            auto c = i - 1;             // first child index
                            load(buf, buf - BATCHSIZE); // load child buffer
                            for (size_t k = 1, j = c - 1 - nodes[c].Length; k < s.Arity; ++k, j -= 1 + nodes[j].Length)
                            {
                                mul(buf, buffer.data() + j * BATCHSIZE);
                            }
                            break;
                        }
                    case NodeType::Log:
                        {
                            log(buf, buf - BATCHSIZE);
                            break;
                        }
                    case NodeType::Exp:
                        {
                            exp(buf, buf - BATCHSIZE);
                            break;
                        }
                    case NodeType::Sin:
                        {
                            sin(buf, buf - BATCHSIZE);
                            break;
                        }
                    case NodeType::Sqrt:
                        {
                            sqrt(buf, buf - BATCHSIZE);
                            break;
                        }
                    case NodeType::Cbrt:
                        {
                            cbrt(buf, buf - BATCHSIZE);
                            break;
                        }
                    default:
                        {
                            break;
                        }
                }
            }
            // the final result is found in the last section of the buffer corresponding to the root node
            std::copy_n(buffer.end() - BATCHSIZE, remainingRows, result + row); 
        }
        // replace nan and inf values 
        auto [min, max] = MinMax(result, numRows);
        LimitToRange(result, min, max, numRows);
    }

    struct ParameterizedEvaluation
    {
        ParameterizedEvaluation(const Tree& tree, const Dataset& dataset, const std::vector<double>& targetValues, const Range range) 

            : tree_ref(tree)
            , dataset_ref(dataset)
            , target_ref(targetValues)
            , range(range) { }

        template<typename T> bool operator()(T const* const* parameters, T* residuals) const
        {
            Evaluate(tree_ref, dataset_ref, range, parameters[0], residuals);
            std::transform(residuals, residuals + range.Size(), target_ref.cbegin() + range.Start, residuals, [](const T& a, const double b) { return a - b; });
            return true;
        }

        private:
        const Tree            & tree_ref;
        const Dataset             & dataset_ref;
        const std::vector<double> & target_ref;
        const Range               range;
    };

    // returns an array of optimized parameters
    template <bool autodiff = true>
    std::vector<double> Optimize(Tree& tree, const Dataset& dataset, const std::vector<double>& targetValues, const Range range, size_t iterations = 50, bool report = false)
    {
        using ceres::DynamicCostFunction;
        using ceres::DynamicNumericDiffCostFunction;
        using ceres::DynamicAutoDiffCostFunction;
        using ceres::CauchyLoss;
        using ceres::Problem;
        using ceres::Solver;
        using ceres::Solve;

        auto coef = tree.GetCoefficients();
        if (coef.empty())
        {
            return coef;
        }
        if (report)
        {
            fmt::print("x_0: ");
            for(auto c : coef) 
                fmt::print("{} ", c);
            fmt::print("\n");
        }

        auto eval = new ParameterizedEvaluation(tree, dataset, targetValues, range);
        DynamicCostFunction* costFunction;
        if constexpr (autodiff) { costFunction = new DynamicAutoDiffCostFunction<ParameterizedEvaluation, JET_STRIDE>(eval); }
        else                    { costFunction = new DynamicNumericDiffCostFunction(eval);                                   }
        costFunction->AddParameterBlock(coef.size());
        costFunction->SetNumResiduals(range.Size());
        //auto lossFunction = new CauchyLoss(0.5); // see http://ceres-solver.org/nnls_tutorial.html#robust-curve-fitting      

        Problem problem;
        problem.AddResidualBlock(costFunction, nullptr, coef.data());

        Solver::Options options;
        options.max_num_iterations = iterations;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = report;
        options.num_threads = 1;
        Solver::Summary summary;
        Solve(options, &problem, &summary);

        if (report)
        {
            fmt::print("{}\n", summary.BriefReport());
            fmt::print("x_final: ");
            for(auto c : coef) 
                fmt::print("{} ", c);
            fmt::print("\n");
        }
        tree.SetCoefficients(coef);
        return coef;
    }

    // set up some convenience methods using perfect forwarding
    template<typename... Args> auto OptimizeAutodiff(Args&&... args)  
    {
        return Optimize<true>(std::forward<Args>(args)...);
    }

    template<typename... Args> auto OptimizeNumeric(Args&&... args)  
    {
        return Optimize<false>(std::forward<Args>(args)...);
    }
}

#undef FOR
#endif

