// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
#ifndef OPERON_CLI_REPORTER_HPP
#define OPERON_CLI_REPORTER_HPP

#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unistd.h>
#include <fmt/format.h>
#include <operon/algorithms/ga_base.hpp>
#include <operon/algorithms/phase_timer.hpp>
#include <operon/operators/linear_scaling.hpp>
#include <functional>
#include <string>
#include <cstdint>
#include <taskflow/taskflow.hpp>
#include <glaze/glaze.hpp>

// Versioned machine-readable report. This is deliberately Operon-specific;
// tree_node_count is not SRBench's model-complexity definition.
namespace Operon {
struct MachineMetrics {
    double r2_train{}; double r2_test{}; double mae_train{}; double mae_test{};
    double nmse_train{}; double nmse_test{}; double mse_train{}; double mse_test{};
    double best_fitness{}; double average_fitness{}; double average_tree_node_count{};
    double elapsed_seconds{}; std::uint64_t evaluator_calls{};
    std::uint64_t result_evaluations{}; std::uint64_t jacobian_evaluations{}; double optimizer_seconds{};
};
struct MachineReport {
    std::string report_kind{"operon_gp"}; int schema_version{1}; std::string symbolic_model{};
    std::uint64_t tree_node_count{}; std::uint64_t seed{}; MachineMetrics metrics{};
};
}
template <> struct glz::meta<Operon::MachineMetrics> {
    using T = Operon::MachineMetrics;
    static constexpr auto value = glz::object("r2_train", &T::r2_train, "r2_test", &T::r2_test,
        "mae_train", &T::mae_train, "mae_test", &T::mae_test, "nmse_train", &T::nmse_train, "nmse_test", &T::nmse_test,
        "mse_train", &T::mse_train, "mse_test", &T::mse_test, "best_fitness", &T::best_fitness,
        "average_fitness", &T::average_fitness, "average_tree_node_count", &T::average_tree_node_count,
        "elapsed_seconds", &T::elapsed_seconds, "evaluator_calls", &T::evaluator_calls,
        "result_evaluations", &T::result_evaluations, "jacobian_evaluations", &T::jacobian_evaluations,
        "optimizer_seconds", &T::optimizer_seconds);
};
template <> struct glz::meta<Operon::MachineReport> {
    using T = Operon::MachineReport;
    static constexpr auto value = glz::object("report_kind", &T::report_kind, "schema_version", &T::schema_version,
        "symbolic_model", &T::symbolic_model, "tree_node_count", &T::tree_node_count, "seed", &T::seed, "metrics", &T::metrics);
};

#include <functional>
#include <string>
#include <taskflow/taskflow.hpp>

namespace Operon {

inline auto WriteMachineReport(MachineReport const& report, std::filesystem::path const& path) -> void {
    auto finite = [](double value) { return std::isfinite(value); };
    auto const& m = report.metrics;
    if (!finite(m.r2_train) || !finite(m.r2_test) || !finite(m.mae_train) || !finite(m.mae_test) ||
        !finite(m.nmse_train) || !finite(m.nmse_test) || !finite(m.mse_train) || !finite(m.mse_test) ||
        !finite(m.best_fitness) || !finite(m.average_fitness) || !finite(m.average_tree_node_count) ||
        !finite(m.elapsed_seconds) || !finite(m.optimizer_seconds)) {
        throw std::runtime_error("machine report contains non-finite metric");
    }
    auto encoded = glz::write_json(report);
    if (!encoded) { throw std::runtime_error("machine report JSON serialization failed"); }
    auto tmp = path;
    tmp += ".tmp-" + std::to_string(static_cast<unsigned long long>(::getpid()));
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) { throw std::runtime_error("could not open machine report temporary file"); }
    out << *encoded << '\n';
    out.close();
    if (!out) { std::filesystem::remove(tmp); throw std::runtime_error("could not write machine report"); }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) { std::filesystem::remove(tmp); throw std::runtime_error("could not publish machine report: " + ec.message()); }
}

using ModelSelectorFn = std::function<Individual(Span<Individual const>)>;


template<typename Evaluator>
class Reporter {
    gsl::not_null<Evaluator const*> evaluator_;
    EvaluatorBase const* statsSource_{nullptr}; // non-owning; must outlive Reporter (null → use evaluator_)
    mutable Operon::Individual best_;
    ModelSelectorFn selector_;

public:
    explicit Reporter(gsl::not_null<Evaluator const*> evaluator, ModelSelectorFn selector = nullptr,
                      EvaluatorBase const* statsSource = nullptr)
        : evaluator_(evaluator), statsSource_(statsSource), selector_(std::move(selector)) {}

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

        best_ = selector_ ? selector_(pop) : getBest(pop);
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
            sf.emplace([&]() {
                Interpreter interpreter{dtable, dataset, &best_.Genotype};
                estimatedTrain = interpreter.Evaluate(best_.Genotype.GetCoefficients(), trainingRange);
                ENSURE(trainingRange.Size() > 0 && estimatedTrain.size() == trainingRange.Size());
            }).name("eval train");

            sf.emplace([&]() {
                Interpreter interpreter{dtable, dataset, &best_.Genotype};
                estimatedTest = interpreter.Evaluate(best_.Genotype.GetCoefficients(), testRange);
                ENSURE(testRange.Size() > 0 && estimatedTest.size() == testRange.Size());
            }).name("eval test");
        });

        // scale values
        auto linearScaling = tf.emplace([&]() {
            auto const scaling = Operon::FitLinearScaling(best_.Genotype, *problem, *dtable, trainingRange);
            if (!scaling) { return; }

            best_.Genotype = scaling->Materialize(std::move(best_.Genotype));
            scaling->ApplyInPlace(Operon::Span<Operon::Scalar>{estimatedTrain});
            scaling->ApplyInPlace(Operon::Span<Operon::Scalar>{estimatedTest});
        }).name("linear scaling");

        double r2Train{};
        double r2Test{};
        double mseTrain{};
        double mseTest{};
        double nmseTrain{};
        double nmseTest{};
        double maeTrain{};
        double maeTest{};

        auto calcStats = tf.emplace([&]() {
            ENSURE(!best_.Genotype.Empty());
            // negate the R2 because this is an internal fitness measure (minimization) which we here repurpose
            r2Train = -Operon::R2{}(estimatedTrain, targetTrain);
            r2Test = -Operon::R2{}(estimatedTest, targetTest);
            mseTrain = Operon::MSE{}(estimatedTrain, targetTrain);
            mseTest = Operon::MSE{}(estimatedTest, targetTest);
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
        calcStats.succeed(linearScaling);
        calcStats.precede(calculateLength, calculateQuality, calculatePopMemory, calculateOffMemory);
        // taskflow.dump(std::cout);

        executor.corun(tf);
        // executor.wait_for_all();

        avgLength /= static_cast<double>(pop.size());
        avgQuality /= static_cast<double>(pop.size());

        using T = std::tuple<std::string, double, std::string>;
        auto const* format = ":>#8.3g"; // see https://fmt.dev/latest/syntax.html

        auto [resEval, jacEval, callCount, cfTime ] = statsSource_ ? statsSource_->Stats() : evaluator_->Stats();
        std::array stats {
            T{ "iteration", gp.Generation(), ":>" },
            T{ "r2_tr", r2Train, format },
            T{ "r2_te", r2Test, format },
            T{ "mae_tr", maeTrain, format },
            T{ "mae_te", maeTest, format },
            T{ "nmse_tr", nmseTrain, format },
            T{ "nmse_te", nmseTest, format },
            T{ "mse_tr", mseTrain, format },
            T{ "mse_te", mseTest, format },
            T{ "best_fit", best_[idx], format },
            T{ "avg_fit", avgQuality, format },
            T{ "best_len", best_.Genotype.Length(), format },
            T{ "avg_len", avgLength, format },
            T{ "eval_cnt", callCount, ":>" },
            T{ "res_eval", resEval, ":>" },
            T{ "jac_eval", jacEval, ":>" },
            T{ "opt_time", cfTime / 1e6, format },
            T{ "seed", config.Seed, ":>10" },
            T{ "sort_ms", [&]{ auto const& t = gp.Timings(); auto it = t.find(std::string{SortTaskName}); return it != t.end() ? it->second * 1e3 : 0.0; }(), format },
            T{ "elapsed", gp.Elapsed(), ":>"},
        };
        latest_.seed = config.Seed;
        latest_.tree_node_count = best_.Genotype.Length();
        latest_.symbolic_model.clear();
        latest_.metrics = MachineMetrics{r2Train, r2Test, maeTrain, maeTest, nmseTrain, nmseTest,
            mseTrain, mseTest, best_[idx], avgQuality, avgLength, gp.Elapsed(),
            static_cast<std::uint64_t>(callCount), static_cast<std::uint64_t>(resEval),
            static_cast<std::uint64_t>(jacEval), cfTime / 1e6};
        PrintStats({ stats.begin(), stats.end() }, gp.Generation() == 0);
    }

    auto SetSymbolicModel(std::string model) const -> void { latest_.symbolic_model = std::move(model); }
    auto GetMachineReport() const -> MachineReport const& { return latest_; }

private:
    mutable MachineReport latest_{};
};
} // namespace Operon

#endif
