// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <numeric>
#include <random>
#include <string_view>
#include <thread>
#include <vector>

#include "operon/algorithms/config.hpp"
#include "operon/algorithms/gp.hpp"
#include "operon/algorithms/nsga2.hpp"
#include "operon/algorithms/stop_reason.hpp"
#include "operon/algorithms/stoppable.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/types.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/generator.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/operators/local_search.hpp"
#include "operon/operators/mutation.hpp"
#include "operon/operators/non_dominated_sorter.hpp"
#include "operon/operators/reinserter.hpp"
#include "operon/operators/selector.hpp"
#include "operon/optimizer/optimizer.hpp"

#include "stop_fixture.hpp"

namespace Operon::Test {
namespace {

    constexpr auto Generation = Operon::StopReason::GenerationLimit;
    constexpr auto Evaluation = Operon::StopReason::EvaluationLimit;
    constexpr auto Pressure = Operon::StopReason::PressureLimit;
    constexpr auto Time = Operon::StopReason::TimeLimit;
    constexpr auto Callback = Operon::StopReason::Callback;
    constexpr auto Signal = Operon::StopReason::Signal;
    constexpr auto EngineError = Operon::StopReason::EngineError;

    // A search driver that only exposes the stop mask.
    class MaskOnly : public Operon::StoppableAlgorithm {
    public:
        using StoppableAlgorithm::RecordStop;
    };

    auto Mask(auto... reasons) -> Operon::StopMask { return (Operon::MaskOf(reasons) | ...); }

} // namespace

TEST_CASE("Stop reasons: names, masks and precedence", "[stop]")
{
    CHECK(Operon::ToString(Generation) == "generation_limit");
    CHECK(Operon::ToString(Evaluation) == "evaluation_limit");
    CHECK(Operon::ToString(Pressure) == "pressure_limit");
    CHECK(Operon::ToString(Time) == "time_limit");
    CHECK(Operon::ToString(Callback) == "callback");
    CHECK(Operon::ToString(Signal) == "signal");
    CHECK(Operon::ToString(EngineError) == "engine_error");

    // distinct bits
    Operon::StopMask all {};
    for (auto const reason : Operon::StopPrecedence) {
        CHECK((all & Operon::MaskOf(reason)) == 0);
        all |= Operon::MaskOf(reason);
    }
    CHECK(std::popcount(all) == 7);

    CHECK_FALSE(Operon::PrimaryStopReason(0).has_value());
    // engine_error > signal > callback > evaluation_limit > pressure_limit > generation_limit > time_limit
    CHECK(Operon::PrimaryStopReason(all) == EngineError);
    CHECK(Operon::PrimaryStopReason(all & ~Mask(EngineError)) == Signal);
    CHECK(Operon::PrimaryStopReason(Mask(Callback, Evaluation, Pressure, Generation, Time)) == Callback);
    CHECK(Operon::PrimaryStopReason(Mask(Evaluation, Pressure, Generation, Time)) == Evaluation);
    CHECK(Operon::PrimaryStopReason(Mask(Pressure, Generation, Time)) == Pressure);
    CHECK(Operon::PrimaryStopReason(Mask(Generation, Time)) == Generation);
    CHECK(Operon::PrimaryStopReason(Mask(Time)) == Time);
}

TEST_CASE("StoppableAlgorithm records stop sources", "[stop]")
{
    MaskOnly algo;
    CHECK_FALSE(algo.StopRequested());
    CHECK(algo.StopReasons() == 0);

    SECTION("RequestStop() is a callback request")
    {
        algo.RequestStop();
        CHECK(algo.StopRequested());
        CHECK(algo.StopReasons() == Mask(Callback));
    }

    SECTION("sources are kept apart and accumulate")
    {
        algo.RequestStop(Signal);
        CHECK(algo.StopRequested());
        CHECK(algo.StopReasons() == Mask(Signal));
        algo.RequestStop(Callback);
        algo.RequestStop(EngineError);
        CHECK(algo.StopReasons() == Mask(Signal, Callback, EngineError));
    }

    SECTION("a limit is not a request, and a limit passed as source becomes a callback request")
    {
        algo.RecordStop(Mask(Generation, Time));
        CHECK_FALSE(algo.StopRequested());
        CHECK(algo.StopReasons() == Mask(Generation, Time));
        algo.RequestStop(Evaluation);
        CHECK(algo.StopRequested());
        CHECK(algo.StopReasons() == Mask(Generation, Time, Callback));
    }

    SECTION("copies carry the mask")
    {
        algo.RecordStop(Mask(Pressure));
        algo.RequestStop(Signal);
        MaskOnly copy { algo };
        CHECK(copy.StopReasons() == Mask(Pressure, Signal));
        MaskOnly assigned;
        assigned = algo;
        CHECK(assigned.StopReasons() == Mask(Pressure, Signal));
    }
}

TEST_CASE("The stop mask loses no update under concurrent fetch_or", "[stop]")
{
    constexpr std::size_t rounds = 2000;
    for (std::size_t round = 0; round < rounds; ++round) {
        MaskOnly algo;
        std::atomic<bool> go { false };
        std::vector<std::jthread> writers;
        constexpr std::array<Operon::StopReason, 4> reasons { Generation, Time, Callback, Signal };
        for (auto const reason : reasons) {
            writers.emplace_back([&, reason] {
                while (!go.load()) {
                    std::this_thread::yield();
                }
                if (reason == Callback || reason == Signal) {
                    algo.RequestStop(reason);
                } else {
                    algo.RecordStop(Operon::MaskOf(reason));
                }
            });
        }
        go = true;
        writers.clear(); // joins
        REQUIRE(algo.StopReasons() == Mask(Generation, Time, Callback, Signal));
    }
}

TEST_CASE("Concurrent readers only ever see the mask grow", "[stop]")
{
    MaskOnly algo;
    std::atomic<bool> done { false };
    std::atomic<std::size_t> violations { 0 };
    std::vector<std::jthread> readers;
    for (int i = 0; i < 3; ++i) {
        readers.emplace_back([&] {
            Operon::StopMask last {};
            bool requested = false;
            while (!done.load()) {
                auto const now = algo.StopReasons();
                if ((now & last) != last) {
                    ++violations;
                }
                last = now;
                auto const nowRequested = algo.StopRequested();
                if (requested && !nowRequested) {
                    ++violations;
                }
                requested = nowRequested;
            }
        });
    }
    for (auto const reason : Operon::StopPrecedence) {
        algo.RecordStop(Operon::MaskOf(reason));
        std::this_thread::yield();
    }
    done = true;
    readers.clear();
    CHECK(violations.load() == 0);
}

TEST_CASE("GP records why it stopped", "[stop][algorithms]")
{
    StopFixture fix;
    Operon::RandomGenerator rng { 5 };

    SECTION("generation limit")
    {
        fix.Config.Generations = 2;
        auto gp = fix.MakeGp();
        gp->Run(rng, nullptr, /*threads=*/1);
        CHECK(gp->Generation() == 2);
        CHECK(gp->StopReasons() == Mask(Generation));
        CHECK(gp->TerminationReasons() == Mask(Generation));
        CHECK_FALSE(gp->StopRequested());
        CHECK_FALSE(gp->IncompleteGeneration().has_value());
        CHECK(gp->GenerationsCompleted() == 2);
        auto const progress = gp->Progress();
        CHECK(progress.PoolSize == StopFixture::PoolSize);
        CHECK(progress.Completed == StopFixture::PoolSize);
        CHECK(progress.Attempted == StopFixture::PoolSize); // the basic generator never rejects
    }

    SECTION("evaluation budget")
    {
        // initialization costs PopSize evaluations; the budget runs out in the middle of the first generation
        fix.Evaluator.SetBudget(StopFixture::PopSize + (StopFixture::PoolSize / 2));
        auto gp = fix.MakeGp();
        gp->Run(rng, nullptr, /*threads=*/1);
        CHECK(gp->StopReasons() == Mask(Evaluation));
        CHECK(gp->TerminationReasons() == Mask(Evaluation));
        auto const incomplete = gp->IncompleteGeneration();
        REQUIRE(incomplete.has_value());
        CHECK(incomplete->PoolSize == StopFixture::PoolSize);
        CHECK(incomplete->Completed == StopFixture::PoolSize / 2);
        CHECK(incomplete->Attempted >= incomplete->Completed);
        CHECK(gp->GenerationsCompleted() == 0);
    }

    SECTION("selection pressure: a maximum of zero is exceeded by the first candidate")
    {
        // The pressure is candidates evaluated since the generator's Prepare() per parent, so it is 0 until the
        // first generation starts and > 0 after the very first candidate.
        fix.OffspringSelection.MaxSelectionPressure(0);
        auto gp = fix.MakeGp(&fix.OffspringSelection);
        gp->Run(rng, nullptr, /*threads=*/1);
        CHECK(gp->StopReasons() == Mask(Pressure));
        CHECK(gp->TerminationReasons() == Mask(Pressure));
        CHECK(fix.OffspringSelection.PressureExhausted());
        CHECK_FALSE(fix.OffspringSelection.BudgetExhausted());
        CHECK(fix.OffspringSelection.Terminate());
        auto const incomplete = gp->IncompleteGeneration();
        REQUIRE(incomplete.has_value());
        CHECK(incomplete->Attempted == 1);
        CHECK(incomplete->Completed <= 1);
        CHECK(gp->GenerationsCompleted() == 0);
    }

    SECTION("selection pressure: exhausted in the middle of a generation")
    {
        // With a huge comparison factor almost no child is accepted, so more than one candidate per parent is
        // needed to fill the pool and the generation is cut short.
        fix.Config.Generations = 50;
        fix.OffspringSelection.MaxSelectionPressure(1);
        fix.OffspringSelection.ComparisonFactor(1000.0);
        auto gp = fix.MakeGp(&fix.OffspringSelection);
        gp->Run(rng, nullptr, /*threads=*/1);
        CHECK(gp->StopReasons() == Mask(Pressure));
        CHECK(gp->TerminationReasons() == Mask(Pressure));
        auto const incomplete = gp->IncompleteGeneration();
        REQUIRE(incomplete.has_value());
        CHECK(incomplete->Completed < incomplete->PoolSize);
        CHECK(incomplete->Attempted > incomplete->Completed);
        CHECK(fix.OffspringSelection.SelectionPressure() > 1.0);
    }

    SECTION("time limit")
    {
        fix.Config.TimeLimit = 0;
        auto gp = fix.MakeGp();
        // GP reports elapsed time in whole milliseconds: let one pass over the loop take longer than that
        gp->Run(
            rng,
            [] {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                return false;
            },
            /*threads=*/1);
        CHECK(gp->StopReasons() == Mask(Time));
        CHECK(gp->TerminationReasons() == Mask(Time));
        CHECK(gp->Elapsed() > 0.0);
    }

    SECTION("callback")
    {
        auto gp = fix.MakeGp();
        gp->Run(rng, [] { return true; }, /*threads=*/1);
        CHECK(gp->StopReasons() == Mask(Callback));
        CHECK(gp->StopRequested());
        CHECK(gp->TerminationReasons() == Mask(Callback));
    }

    SECTION("signal requested from outside the callback")
    {
        auto gp = fix.MakeGp();
        auto* const raw = gp.get();
        gp->Run(
            rng,
            [raw] {
                raw->RequestStop(Signal);
                return false;
            },
            /*threads=*/1);
        CHECK(gp->StopReasons() == Mask(Signal));
        CHECK(gp->TerminationReasons() == Mask(Signal));
    }

    SECTION("an engine error request")
    {
        auto gp = fix.MakeGp();
        auto* const raw = gp.get();
        gp->Run(
            rng,
            [raw] {
                raw->RequestStop(EngineError);
                return false;
            },
            /*threads=*/1);
        CHECK(gp->StopReasons() == Mask(EngineError));
    }
}

TEST_CASE("Simultaneous stop conditions are all recorded", "[stop][algorithms]")
{
    StopFixture fix;
    Operon::RandomGenerator rng { 5 };
    fix.Config.Generations = 0; // reached right after initialization
    fix.Evaluator.SetBudget(1); // used up by initialization
    fix.Config.TimeLimit = 0;
    auto gp = fix.MakeGp();

    gp->Run(
        rng,
        [] {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            return true; // and the callback asks to stop
        },
        /*threads=*/1);

    auto const expected = Mask(Generation, Evaluation, Callback, Time);
    CHECK(gp->StopReasons() == expected);
    CHECK(gp->TerminationReasons() == expected);
    CHECK(Operon::PrimaryStopReason(gp->TerminationReasons()) == Callback);

    // A later Run() starts from a clean mask.
    fix.Config.Generations = 0;
    fix.Evaluator.SetBudget(std::size_t { 1 } << 40U);
    fix.Config.TimeLimit = ~std::size_t { 0 };
    auto fresh = fix.MakeGp();
    fresh->Run(rng, nullptr, /*threads=*/1);
    CHECK(fresh->StopReasons() == Mask(Generation));
}

TEST_CASE("Concurrent stop polling at four threads records exactly the true conditions", "[stop][algorithms]")
{
    constexpr std::size_t threads = 4;
    for (int round = 0; round < 25; ++round) {
        StopFixture fix;
        Operon::RandomGenerator rng { static_cast<std::size_t>(round) };
        // the budget runs out in the middle of the first generation, while four workers poll for it
        fix.Evaluator.SetBudget(StopFixture::PopSize + (StopFixture::PoolSize / 2));
        auto gp = fix.MakeGp();
        gp->Run(rng, nullptr, threads);

        REQUIRE(gp->StopReasons() == Mask(Evaluation));
        REQUIRE(gp->TerminationReasons() == Mask(Evaluation));
        auto const incomplete = gp->IncompleteGeneration();
        REQUIRE(incomplete.has_value());
        CHECK(incomplete->Completed < incomplete->PoolSize);
        CHECK(incomplete->Attempted >= incomplete->Completed);
        // the budget is checked before each attempt, so a few attempts per worker may overshoot it, no more
        CHECK(fix.Evaluator.TotalEvaluations() >= fix.Evaluator.Budget());
        CHECK(fix.Evaluator.TotalEvaluations() <= fix.Evaluator.Budget() + threads);
    }
}

TEST_CASE("NSGA2 records why it stopped", "[stop][algorithms]")
{
    StopFixture fix;
    Operon::RandomGenerator rng { 9 };
    Operon::CrowdedComparison comp;
    Operon::TournamentSelector fem { comp };
    Operon::TournamentSelector male { comp };
    Operon::BasicOffspringGenerator generator { &fix.Evaluator, &fix.Crossover, &fix.Mutator, &fem, &male };
    Operon::KeepBestReinserter reinserter { comp };
    Operon::DeductiveSorter sorter;
    auto make = [&] {
        return std::make_unique<Operon::NSGA2>(
            fix.Config, &fix.Problem, &fix.TreeInit, &fix.CoeffInit, &generator, &reinserter, &sorter);
    };

    SECTION("generation limit")
    {
        fix.Config.Generations = 2;
        auto nsga = make();
        nsga->Run(rng, nullptr, /*threads=*/1);
        CHECK(nsga->StopReasons() == Mask(Generation));
        CHECK(nsga->GenerationsCompleted() == 2);
        CHECK_FALSE(nsga->IncompleteGeneration().has_value());
    }

    SECTION("evaluation budget: the generation that ran out of budget is not counted")
    {
        fix.Evaluator.SetBudget(StopFixture::PopSize + (StopFixture::PoolSize / 2));
        auto nsga = make();
        nsga->Run(rng, nullptr, /*threads=*/1);
        CHECK(nsga->StopReasons() == Mask(Evaluation));
        CHECK(nsga->TerminationReasons() == Mask(Evaluation));
        CHECK(nsga->Generation() == 0);
        CHECK(nsga->GenerationsCompleted() == 0);
        auto const incomplete = nsga->IncompleteGeneration();
        REQUIRE(incomplete.has_value());
        CHECK(incomplete->Completed == StopFixture::PoolSize / 2);
        CHECK(incomplete->PoolSize == StopFixture::PoolSize);
    }

    SECTION("callback and signal")
    {
        auto nsga = make();
        auto* const raw = nsga.get();
        nsga->Run(
            rng,
            [raw] {
                raw->RequestStop(Signal);
                return true;
            },
            /*threads=*/1);
        CHECK(nsga->StopReasons() == Mask(Signal, Callback));
        CHECK(Operon::PrimaryStopReason(nsga->TerminationReasons()) == Signal);
    }
}

} // namespace Operon::Test
