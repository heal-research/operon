#ifndef OPERON_CLI_REPORTER_HPP
#define OPERON_CLI_REPORTER_HPP

#include "operon/operators/evaluator.hpp"
#include <fmt/format.h>
#include <operon/algorithms/ga_base.hpp>

#include <string>
#include <taskflow/taskflow.hpp>

namespace Operon {

template<typename DTable>
class Reporter {
public:
    enum class ModelCriterion : std::uint8_t { MeanSquaredError, MinimumDescriptionLength };

private:
    gsl::not_null<DTable const*> dtable_;
    gsl::not_null<EvaluatorBase const*> evaluator_;
    mutable Operon::Individual best_;
    mutable ModelCriterion crit_{ModelCriterion::MeanSquaredError};
    mutable Operon::Scalar sigma_{1.0};

    char sep_ = ' ';
    char end_ = '\n';

public:
    explicit Reporter(gsl::not_null<DTable const*> dtable, gsl::not_null<EvaluatorBase const*> evaluator, char sep = ' ', char end = '\n')
        : dtable_(dtable), evaluator_(evaluator), sep_{sep}, end_{end} {}

    static auto PrintStats(std::vector<std::tuple<std::string, double, std::string>> const& stats, bool printHeader, char sep = ' ', char end = '\n') -> void {
        std::vector<size_t> widths;
        auto out = fmt::memory_buffer();
        for (auto const& [name, value, format] : stats) {
            fmt::format_to(std::back_inserter(out), fmt::runtime(fmt::format("{{{}}}", format)), value);
            auto width = std::max(name.size(), fmt::to_string(out).size());
            widths.push_back(width);
            out.clear();
        }
        if (printHeader) {
            for (auto i = 0UL; i < stats.size(); ++i) {
                fmt::print("{}{}", fmt::format("{:>{}}", std::get<0>(stats[i]), widths[i]), i < stats.size()-1 ? sep : ' ');
            }
            fmt::print("\n");
        }
        for (auto i = 0UL; i < stats.size(); ++i) {
            fmt::format_to(std::back_inserter(out), fmt::runtime(fmt::format("{{{}}}", std::get<2>(stats[i]))), std::get<1>(stats[i]));
            fmt::print("{}{}", fmt::format("{:>{}}", fmt::to_string(out), widths[i]), i < stats.size()-1 ? sep : ' ');
            out.clear();
        }
        fmt::print("{}", end);
    }

    auto SetModelCriterion(ModelCriterion crit) const { crit_ = crit; }
    auto SetSigma(Operon::Scalar sigma) const { sigma_ = sigma; }

    auto GetBest() const -> Operon::Individual const& { return best_; }

    auto operator()(tf::Executor& executor, Operon::GeneticAlgorithmBase const& gp) const -> void {
        auto const config = gp.GetConfig();
        auto const pop = gp.Parents();
        auto const off = gp.Offspring();

        constexpr auto idx{0};

        if (crit_ == ModelCriterion::MeanSquaredError) {
            const auto minElem = std::min_element(pop.begin(), pop.end(), [&](auto const& lhs, auto const& rhs) { return lhs[idx] < rhs[idx]; });
            best_ = *minElem;
        } else {
            auto const* problem = evaluator_->GetProblem();
            Operon::MinimumDescriptionLengthEvaluator<DTable, Operon::GaussianLikelihood<Operon::Scalar>> mdlEval{problem, dtable_.get()};
            mdlEval.SetSigma({ sigma_ });
            Operon::RandomGenerator rng{1234};

            Operon::Scalar bestMdl{ std::numeric_limits<Operon::Scalar>::max() };
            for (auto const& ind : gp.ParetoFront()) {
                if (auto mdl = mdlEval(rng, ind); mdl[0] < bestMdl) {
                    best_ = ind;
                    bestMdl = mdl[0];
                }
            }
        }

        ENSURE(best_.Size() > 0);

        tf::Taskflow tf;
        tf.name("report results");

        auto const* problem = gp.GetProblem();
        auto trainingRange  = problem->TrainingRange();
        auto testRange      = problem->TestRange();
        auto targetTrain    = problem->TargetValues(trainingRange);
        auto targetTest     = problem->TargetValues(testRange);

        Operon::Vector<Operon::Scalar> estimatedTrain;
        Operon::Vector<Operon::Scalar> estimatedTest;

        auto const* dataset = problem->GetDataset();

        using Interpreter = Operon::Interpreter<Operon::Scalar, DTable>;

        auto evaluate = tf.emplace([&](tf::Subflow& sf) {
            sf.emplace([&]() {
                Interpreter interpreter{dtable_, dataset, &best_.Genotype};
                estimatedTrain = interpreter.Evaluate(best_.Genotype.GetCoefficients(), trainingRange);
                ENSURE(trainingRange.Size() > 0 && estimatedTrain.size() == trainingRange.Size());
            }).name("eval train");

            sf.emplace([&]() {
                Interpreter interpreter{dtable_, dataset, &best_.Genotype};
                estimatedTest = interpreter.Evaluate(best_.Genotype.GetCoefficients(), testRange);
                ENSURE(testRange.Size() > 0 && estimatedTest.size() == testRange.Size());
            }).name("eval test");
        });

        // scale values
        Operon::Scalar a{1.0};
        Operon::Scalar b{0.0};
        auto linearScaling = tf.emplace([&]() {
            auto [a_, b_] = Operon::FitLeastSquares(estimatedTrain, targetTrain);
            a = static_cast<Operon::Scalar>(a_);
            b = static_cast<Operon::Scalar>(b_);
            // add scaling terms to the tree
            auto& nodes = best_.Genotype.Nodes();
            auto const sz = nodes.size();
            if (std::abs(a - Operon::Scalar{1}) > std::numeric_limits<Operon::Scalar>::epsilon()) {
                nodes.emplace_back(Operon::Node::Constant(a));
                nodes.emplace_back(Operon::NodeType::Mul);
            }
            if (std::abs(b) > std::numeric_limits<Operon::Scalar>::epsilon()) {
                nodes.emplace_back(Operon::Node::Constant(b));
                nodes.emplace_back(Operon::NodeType::Add);
            }
            if (nodes.size() > sz) {
                best_.Genotype.UpdateNodes();
            }
        }).name("linear scaling");

        double r2Train{};
        double r2Test{};
        double nmseTrain{};
        double nmseTest{};
        double maeTrain{};
        double maeTest{};

        auto scale = tf.emplace([&](tf::Subflow& sf) {
            sf.emplace([&]() {
                ENSURE(estimatedTrain.size() == trainingRange.Size());
                Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1>> estimated(estimatedTrain.data(), std::ssize(estimatedTrain));
                estimated = estimated * a + b;
            }).name("scale train");

            sf.emplace([&]() {
                ENSURE(estimatedTest.size() == testRange.Size());
                Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1>> estimated(estimatedTest.data(), std::ssize(estimatedTest));
                estimated = estimated * a + b;
            }).name("scale test");
        });

        auto calcStats = tf.emplace([&]() {
            ENSURE(!best_.Genotype.Empty());
            // negate the R2 because this is an internal fitness measure (minimization) which we here repurpose
            r2Train = -Operon::R2{}(estimatedTrain, targetTrain);
            r2Test = -Operon::R2{}(estimatedTest, targetTest);

            nmseTrain = Operon::NMSE{}(estimatedTrain, targetTrain);
            nmseTest = Operon::NMSE{}(estimatedTest, targetTest);

            maeTrain = Operon::MAE{}(estimatedTrain, targetTrain);
            maeTest = Operon::MAE{}(estimatedTest, targetTest);
        }).name("calc stats");

        double avgLength = 0;
        double avgQuality = 0;
        double totalMemory = 0;

        auto getSize = [](Operon::Individual const& ind) { return sizeof(ind) + sizeof(ind.Genotype) + (sizeof(Operon::Node) * ind.Genotype.Nodes().capacity()); };
        auto calculateLength = tf.transform_reduce(pop.begin(), pop.end(), avgLength, std::plus{}, [](auto const& ind) { return ind.Genotype.Length(); }).name("calc length");
        auto calculateQuality = tf.transform_reduce(pop.begin(), pop.end(), avgQuality, std::plus{}, [idx=idx](auto const& ind) { return ind[idx]; }).name("calc quality");
        auto calculatePopMemory = tf.transform_reduce(pop.begin(), pop.end(), totalMemory, std::plus{}, [&](auto const& ind) { return getSize(ind); }).name("calc parent mem");
        auto calculateOffMemory = tf.transform_reduce(off.begin(), off.end(), totalMemory, std::plus{}, [&](auto const& ind) { return getSize(ind); }).name("calc child mem");

        // define task graph
        evaluate.precede(linearScaling);
        linearScaling.precede(scale);
        calcStats.succeed(scale);
        calcStats.precede(calculateLength, calculateQuality, calculatePopMemory, calculateOffMemory);
        // taskflow.dump(std::cout);

        executor.corun(tf);
        // executor.wait_for_all();

        avgLength /= static_cast<double>(pop.size());
        avgQuality /= static_cast<double>(pop.size());

        using T = std::tuple<std::string, double, std::string>;
        auto const* format = ":>#8.3g"; // see https://fmt.dev/latest/syntax.html
        auto cacheHits = Zobrist::GetInstance()->Hits();
        auto cacheTotal = Zobrist::GetInstance()->Total();

        auto [resEval, jacEval, callCount, cfTime ] = evaluator_->Stats();
        std::array stats {
            T{ "iteration", gp.Generation(), ":>" },
            T{ "r2_tr", r2Train, format },
            T{ "r2_te", r2Test, format },
            T{ "mae_tr", maeTrain, format },
            T{ "mae_te", maeTest, format },
            T{ "nmse_tr", nmseTrain, format },
            T{ "nmse_te", nmseTest, format },
            T{ "best_fit", best_[idx], format },
            T{ "avg_fit", avgQuality, format },
            T{ "best_len", best_.Genotype.Length(), format },
            T{ "avg_len", avgLength, format },
            T{ "eval_cnt", callCount, ":>" },
            T{ "res_eval", resEval, ":>" },
            T{ "jac_eval", jacEval, ":>" },
            T{ "opt_time", cfTime, ":>" },
            T{ "cache_hits", cacheHits, ":>" },
            T{ "cache_total", cacheTotal, ":>" },
            T{ "seed", config.Seed, ":>10" },
            T{ "elapsed", gp.Elapsed(), ":>"},
        };
        PrintStats({ stats.begin(), stats.end() }, gp.Generation() == 0, sep_, end_);
    }
};
} // namespace Operon

#endif
