// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#pragma once

#include <chrono>
#include <mutex>
#include <string>
#include <vector>

#include <taskflow/core/observer.hpp>

#include "operon/core/types.hpp"

namespace Operon {

class PhaseTimer final : public tf::ObserverInterface {
    using Clock = std::chrono::steady_clock;

    std::vector<Clock::time_point> entry_;  // per-worker; each worker writes only its own slot
    mutable std::mutex mtx_;
    Operon::Map<std::string, double> totals_;  // phase name -> cumulative seconds

public:
    void set_up(size_t numWorkers) override {
        entry_.resize(numWorkers);
    }

    void on_entry(tf::WorkerView w, tf::TaskView tv) override {
        if (tv.name().empty()) { return; }
        entry_[w.id()] = Clock::now();
    }

    void on_exit(tf::WorkerView w, tf::TaskView tv) override {
        if (tv.name().empty()) { return; }
        auto const dt = std::chrono::duration<double>(Clock::now() - entry_[w.id()]).count();
        std::scoped_lock lock{mtx_};
        totals_[std::string(tv.name())] += dt;
    }

    [[nodiscard]] auto Timings() const -> Operon::Map<std::string, double> {
        std::scoped_lock lock{mtx_};
        return totals_;
    }
};

} // namespace Operon
