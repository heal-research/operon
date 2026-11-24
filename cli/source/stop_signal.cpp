// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "stop_signal.hpp"

#include <atomic>
#include <csignal>

namespace Operon::Cli::StopSignals {

namespace {
    // Everything the handler touches is a lock-free atomic, which is what makes it async-signal-safe.
    std::atomic<int> received { 0 }; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
    std::atomic<Operon::StoppableAlgorithm*> target {
        nullptr
    }; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
    static_assert(std::atomic<int>::is_always_lock_free);
    static_assert(std::atomic<Operon::StoppableAlgorithm*>::is_always_lock_free);

    using Handler = void (*)(int);
    Handler previousInt { SIG_DFL }; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
    Handler previousTerm { SIG_DFL }; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

    void OnSignal(int signal)
    {
        auto expected = 0;
        received.compare_exchange_strong(expected, signal);
        if (auto* algorithm = target.load()) {
            algorithm->RequestStop(Operon::StopReason::Signal);
        }
        // One graceful stop only: a second signal gets the default action (terminate).
        std::signal(signal, SIG_DFL);
    }
} // namespace

auto Install() -> void
{
    previousInt = std::signal(SIGINT, OnSignal);
    previousTerm = std::signal(SIGTERM, OnSignal);
}

auto Uninstall() -> void
{
    std::signal(SIGINT, previousInt);
    std::signal(SIGTERM, previousTerm);
    previousInt = SIG_DFL;
    previousTerm = SIG_DFL;
    target.store(nullptr);
    received.store(0);
}

auto SetTarget(Operon::StoppableAlgorithm* algorithm) -> void { target.store(algorithm); }

auto Received() -> int { return received.load(); }

auto Forward(Operon::StoppableAlgorithm& algorithm) -> void
{
    if (received.load() != 0) {
        algorithm.RequestStop(Operon::StopReason::Signal);
    }
}

} // namespace Operon::Cli::StopSignals
