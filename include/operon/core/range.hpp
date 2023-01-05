// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_RANGE_HPP
#define OPERON_RANGE_HPP

#include <tuple>
#include "contracts.hpp"

namespace Operon {
class Range {
    public:
        [[nodiscard]] inline auto Start() const noexcept -> std::size_t { return range_.first; }
        [[nodiscard]] inline auto End() const noexcept -> std::size_t { return range_.second; }
        [[nodiscard]] inline auto Size() const noexcept -> std::size_t { return range_.second - range_.first; }
        [[nodiscard]] auto Bounds() const noexcept -> std::pair<std::size_t, std::size_t> { return range_; }

        Range() = default;
        Range(std::size_t start, std::size_t end)
            : range_(CheckRange(start, end))
        {
        }

        explicit Range(std::pair<std::size_t, std::size_t> range)
            : range_(CheckRange(range.first, range.second))
        {
        }

        auto operator=(std::pair<std::size_t, std::size_t> p) -> Range&
        {
            auto [start, end] = p;
            range_ = CheckRange(start, end);
            return *this;
        }

    private:
        static auto CheckRange(std::size_t start, std::size_t end) -> std::pair<std::size_t, std::size_t>
        {
            EXPECT(start <= end);
            return { start, end };
        }
        std::pair<std::size_t, std::size_t> range_;
};
} // namespace Operon

#endif
