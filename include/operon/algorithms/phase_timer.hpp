// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#pragma once

#include <chrono>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <taskflow/core/observer.hpp>

#include "operon/core/types.hpp"

namespace Operon {

// Task name constants — used in algorithm implementations and reporters to
// avoid magic-string coupling between the task .name() call and any lookup site.
inline constexpr std::string_view kSortTaskName = "non-dominated sort";

namespace detail {
// Transparent hash for std::string keys: avoids constructing a std::string
// on every on_exit call after the first encounter of each task name.
struct StringHash {
    using is_transparent = void;
    using is_avalanching = void;
    auto operator()(std::string_view sv) const noexcept -> uint64_t {
        return ankerl::unordered_dense::hash<std::string_view>{}(sv);
    }
};
} // namespace detail

class PhaseTimer final : public tf::ObserverInterface {
    using Clock    = std::chrono::steady_clock;
    using Totals   = Operon::Map<std::string, double, detail::StringHash, std::equal_to<>>;

    std::vector<Clock::time_point> entry_;  // per-worker; each worker writes only its own slot
    mutable std::mutex mtx_;
    Totals totals_;  // phase name -> cumulative seconds

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
        totals_[tv.name()] += dt;  // transparent lookup: no std::string construction after first insert
    }

    // Returns a snapshot of accumulated timings as a plain map.
    // Note: a new PhaseTimer is created per Run() call, so totals_ always
    // reflects a single run. Reset() on the algorithm clears phaseTimes_ on
    // the base but does not affect any in-flight observer.
    [[nodiscard]] auto Timings() const -> Operon::Map<std::string, double> {
        std::scoped_lock lock{mtx_};
        return { totals_.begin(), totals_.end() };
    }
};

} // namespace Operon
