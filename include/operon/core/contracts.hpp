// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_CONTRACTS_HPP
#define OPERON_CONTRACTS_HPP

#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/ranges.h>

// NOLINTBEGIN(*)
#define EXPECT(cond) \
    if(!(cond)) \
    { \
        fmt::print("Precondition {} failed at {}: {}\n", fmt::format(fmt::fg(fmt::terminal_color::green), "{}", #cond), __FILE__, __LINE__); \
        std::terminate(); \
    } 

#define ENSURE(cond) \
    if(!(cond)) \
    { \
        fmt::print("Precondition {} failed at {}: {}\n", fmt::format(fmt::fg(fmt::terminal_color::green), "{}", #cond), __FILE__, __LINE__); \
        std::terminate(); \
    }
// NOLINTEND(*)
#endif

