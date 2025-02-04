#ifndef OPERON_CLI_REPORTER_HPP
#define OPERON_CLI_REPORTER_HPP

#include <fmt/format.h>
#include <operon/algorithms/ga_base.hpp>

#include <string>
#include <taskflow/taskflow.hpp>

namespace Operon {



template<typename Evaluator>
class Reporter {
    gsl::not_null<Evaluator const*> evaluator_;
    mutable Operon::Individual best_;

public:
    explicit Reporter(gsl::not_null<Evaluator const*> evaluator)
        : evaluator_(evaluator) {}

    static auto PrintStats(std::vector<std::tuple<std::string, double, std::string>> const& stats, bool printHeader) -> void {
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
                fmt::print("{} ", fmt::format("{:>{}}", std::get<0>(stats[i]), widths[i]));
            }
            fmt::print("\n");
        }
        for (auto i = 0UL; i < stats.size(); ++i) {
            fmt::format_to(std::back_inserter(out), fmt::runtime(fmt::format("{{{}}}", std::get<2>(stats[i]))), std::get<1>(stats[i]));
            fmt::print("{} ", fmt::format("{:>{}}", fmt::to_string(out), widths[i]));
            out.clear();
        }
        fmt::print("\n");
    }

    auto GetBest() const -> Operon::Individual const& { return best_; }

    auto operator()(tf::Executor& executor, Operon::GeneticAlgorithmBase const& gp) const -> void {
        auto const config = gp.GetConfig();
        auto const pop = gp.Parents();
        auto const off = gp.Offspring();

        constexpr auto idx{0};
        auto getBest = [&](Operon::Span<Operon::Individual const> pop) -> Operon::Individual {
            const auto minElem = std::min_element(pop.begin(), pop.end(), [&](auto const& lhs, auto const& rhs) { return lhs[idx] < rhs[idx]; });
            return *minElem;
        };

        best_ = getBest(pop);
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

        auto dtable = evaluator_->GetDispatchTable();
        using Interpreter = typename Evaluator::TInterpreter;

        auto evaluate = tf.emplace([&](tf::Subflow& sf) {
            auto evalTrain = sf.emplace([&]() {
                Interpreter interpreter{dtable, dataset, &best_.Genotype};
                estimatedTrain = interpreter.Evaluate(best_.Genotype.GetCoefficients(), trainingRange);
                ENSURE(trainingRange.Size() > 0 && estimatedTrain.size() == trainingRange.Size());
            }).name("eval train");

            auto evalTest = sf.emplace([&]() {
                Interpreter interpreter{dtable, dataset, &best_.Genotype};
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
            auto scaleTrain = sf.emplace([&]() {
                ENSURE(estimatedTrain.size() == trainingRange.Size());
                Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1>> estimated(estimatedTrain.data(), std::ssize(estimatedTrain));
                estimated = estimated * a + b;
            }).name("scale train");

            auto scaleTest = sf.emplace([&]() {
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

        auto getSize = [](Operon::Individual const& ind) { return sizeof(ind) + sizeof(ind.Genotype) + sizeof(Operon::Node) * ind.Genotype.Nodes().capacity(); };
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
            T{ "seed", config.Seed, ":>10" },
            T{ "elapsed", gp.Elapsed(), ":>"},
        };
        PrintStats({ stats.begin(), stats.end() }, gp.Generation() == 0);
    }
};
} // namespace Operon

#endif
