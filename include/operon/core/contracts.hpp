// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_CONTRACTS_HPP
#define OPERON_CONTRACTS_HPP

#include <fmt/color.h>
#include <fmt/core.h>
#include <gsl/assert>

#define EXPECT(cond) \
    if(GSL_UNLIKELY(!(cond))) \
    { \
        fmt::print("Precondition {} failed at {}: {}\n", fmt::format(fmt::fg(fmt::terminal_color::green), "{}", #cond), __FILE__, __LINE__); \
        gsl::details::terminate(); \
    } 

#define ENSURE(cond) \
    if(GSL_UNLIKELY(!(cond))) \
    { \
        fmt::print("Precondition {} failed at {}: {}\n", fmt::format(fmt::fg(fmt::terminal_color::green), "{}", #cond), __FILE__, __LINE__); \
        gsl::details::terminate(); \
    }

#endif

