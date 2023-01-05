// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_PROJECTION_HPP
#define OPERON_PROJECTION_HPP

#include <functional>
#include <iterator>
#include <type_traits>

namespace Operon {

namespace detail {
    struct Identity {
        template <typename T>
        constexpr auto operator()(T&& v) const noexcept -> decltype(std::forward<T>(v))
        {
            return std::forward<T>(v);
        }
    };
} // namespace detail

template<typename InputIt, typename Func = detail::Identity>
struct ProjectionIterator {
    using T = typename std::iterator_traits<InputIt>::value_type;
    using R = std::invoke_result_t<Func, T>;

    // projection iterator traits
    using value_type = std::remove_reference_t<R>; // NOLINT
    using pointer = void; //NOLINT
    using reference = value_type&; // NOLINT
    using difference_type = typename std::iterator_traits<InputIt>::difference_type; // NOLINT
    using iterator_category = typename std::iterator_traits<InputIt>::iterator_category; // NOLINT

    explicit ProjectionIterator(InputIt it, Func const& f) : it_(it), pr_(f) { }
    explicit ProjectionIterator(InputIt it, Func&& f) : it_(it), pr_(std::move(f)) { }

    inline auto operator*() const noexcept -> value_type { return std::invoke(pr_, std::forward<typename std::iterator_traits<InputIt>::reference>(*it_)); }

    inline auto operator==(ProjectionIterator const& rhs) const noexcept -> bool
    {
        return it_ == rhs.it_;
    }

    inline auto operator!=(ProjectionIterator const& rhs) const noexcept -> bool
    {
        return !(*this == rhs);
    }

    inline auto operator<(ProjectionIterator const& rhs) const noexcept -> bool
    {
        return it_ < rhs.it_;
    }

    inline auto operator>(ProjectionIterator const& rhs) const noexcept -> bool
    {
        return rhs < *this;
    }

    inline auto operator<=(ProjectionIterator const& rhs) const noexcept -> bool
    {
        return !(*this > rhs);
    }

    inline auto operator>=(ProjectionIterator const& rhs) const noexcept -> bool
    {
        return !(rhs < *this);
    }

    inline auto operator+(ProjectionIterator const& rhs) const noexcept -> difference_type
    {
        return it_ + rhs.it_;
    }

    inline auto operator-(ProjectionIterator const& rhs) const noexcept -> difference_type
    {
        return it_ - rhs.it_;
    }

    inline auto operator++() noexcept -> ProjectionIterator&
    {
        ++it_;
        return *this;
    }

    inline auto operator++(int) noexcept -> ProjectionIterator
    {
        auto ret = *this;
        ++*this;
        return ret;
    }

    inline auto operator--() noexcept -> ProjectionIterator&
    {
        --it_;
        return *this;
    }

    inline auto operator--(int) noexcept -> ProjectionIterator
    {
        auto ret = *this;
        --*this;
        return ret;
    }

    inline auto operator+=(int n) noexcept -> ProjectionIterator&
    {
        it_ += n;
        return *this;
    }

    inline auto operator+(int n) const noexcept -> ProjectionIterator
    {
        auto ret = *this;
        ret += n;
        return ret;
    }

    inline auto operator-=(int n) noexcept -> ProjectionIterator&
    {
        it_ -= n;
        return *this;
    }

    inline auto operator-(int n) const noexcept -> ProjectionIterator
    {
        auto ret = *this;
        ret -= n;
        return ret;
    }

    private:
    InputIt it_;
    Func pr_;
};

template<typename Container, typename Func = detail::Identity>
struct Projection {
    using InputIt = ProjectionIterator<typename Container::const_iterator, Func>;

    explicit Projection(Container const& c, Func const& f)
        : beg_(c.begin(), f), end_(c.end(), f)
    {
    }

    auto begin() const -> InputIt { return beg_; } // NOLINT
    auto end() const -> InputIt { return end_; } // NOLINT
    bool empty() const noexcept { return beg_ == end_; } // NOLINT

    private:
        InputIt beg_;
        InputIt end_;
};

} // namespace Operon

#endif
