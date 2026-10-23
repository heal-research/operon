// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_CLI_STOP_SIGNAL_HPP
#define OPERON_CLI_STOP_SIGNAL_HPP

#include "operon/algorithms/stoppable.hpp"

namespace Operon::Cli::StopSignals {

// Process-wide SIGINT/SIGTERM handling for the long-running CLIs: the first signal asks the running algorithm to
// stop (StopReason::Signal) instead of killing the process, so the program can finish, write its final record and
// exit with SignalExitCode(). The default disposition is restored by the handler itself, so a second signal
// terminates the process as usual.

// Installs the handlers. Call once, early. Nothing changes for a process that never calls it.
auto Install() -> void;

// Restores the previous handlers and forgets any recorded signal and target.
auto Uninstall() -> void;

// The algorithm the handler forwards to while it runs (non-owning; clear it with nullptr before the algorithm is
// destroyed). A signal that arrives while no target is set is only recorded; call Forward() once there is one.
auto SetTarget(Operon::StoppableAlgorithm* target) -> void;

// The number of the first signal received since Install(), or 0.
[[nodiscard]] auto Received() -> int;

// If a signal has been received, asks `algorithm` to stop with StopReason::Signal (idempotent).
auto Forward(Operon::StoppableAlgorithm& algorithm) -> void;

// Conventional shell status for termination by `signal`: 128 + the signal number.
[[nodiscard]] constexpr auto SignalExitCode(int signal) -> int { return 128 + signal; }

} // namespace Operon::Cli::StopSignals

#endif
