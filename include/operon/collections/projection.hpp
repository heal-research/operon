// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_PROJECTION_HPP
#define OPERON_PROJECTION_HPP

#include <functional>
#include <iterator>
#include <type_traits>

namespace Operon {

namespace detail {
    struct identity {
        template <typename T>
        constexpr auto operator()(T&& v) const noexcept -> decltype(std::forward<T>(v))
        {
            return std::forward<T>(v);
        }
    };
};

template<typename InputIt, typename Func = detail::identity>
struct ProjectionIterator {
    using T = typename std::iterator_traits<InputIt>::value_type;
    using R = std::result_of_t<Func(T)>;

    // projection iterator traits
    using value_type = std::remove_reference_t<R>;
    using pointer = void;
    using reference = value_type&;
    using difference_type = typename std::iterator_traits<InputIt>::difference_type;
    using iterator_category = typename std::iterator_traits<InputIt>::iterator_category;

    explicit ProjectionIterator(InputIt it, Func const& f) : it_(it), pr_(f) { }
    explicit ProjectionIterator(InputIt it, Func&& f) : it_(it), pr_(std::move(f)) { }

    inline value_type operator*() const noexcept { return std::invoke(pr_, std::forward<typename std::iterator_traits<InputIt>::reference>(*it_)); }

    inline bool operator==(ProjectionIterator const& rhs) const noexcept
    {
        return it_ == rhs.it_;
    }

    inline bool operator!=(ProjectionIterator const& rhs) const noexcept
    {
        return !(*this == rhs);
    }

    inline bool operator<(ProjectionIterator const& rhs) const noexcept
    {
        return it_ < rhs.it_;
    }

    inline bool operator>(ProjectionIterator const& rhs) const noexcept
    {
        return rhs < *this;
    }

    inline bool operator<=(ProjectionIterator const& rhs) const noexcept
    {
        return !(*this > rhs);
    }

    inline bool operator>=(ProjectionIterator const& rhs) const noexcept
    {
        return !(rhs < *this);
    }

    inline difference_type operator+(ProjectionIterator const& rhs) const noexcept
    {
        return it_ + rhs.it_;
    }

    inline difference_type operator-(ProjectionIterator const& rhs) const noexcept
    {
        return it_ - rhs.it_;
    }

    inline ProjectionIterator& operator++() noexcept
    {
        ++it_;
        return *this;
    }

    inline ProjectionIterator operator++(int) noexcept
    {
        auto ret = *this;
        ++*this;
        return ret;
    }

    inline ProjectionIterator& operator--() noexcept
    {
        --it_;
        return *this;
    }

    inline ProjectionIterator operator--(int) noexcept
    {
        auto ret = *this;
        --*this;
        return ret;
    }

    inline ProjectionIterator& operator+=(int n) noexcept
    {
        it_ += n;
        return *this;
    }

    inline ProjectionIterator operator+(int n) noexcept
    {
        auto ret = *this;
        ret += n;
        return ret;
    }

    inline ProjectionIterator& operator-=(int n) noexcept
    {
        it_ -= n;
        return *this;
    }

    inline ProjectionIterator operator-(int n) noexcept
    {
        auto ret = *this;
        ret -= n;
        return ret;
    }

    private:
    InputIt it_;
    Func pr_;
};

template<typename Container, typename Func = detail::identity>
struct Projection {
    using InputIt = ProjectionIterator<typename Container::const_iterator, Func>;

    explicit Projection(Container const& c, Func const& f)
        : beg_(c.begin(), f), end_(c.end(), f)
    {
    }

    InputIt begin() const { return beg_; }
    InputIt end() const { return end_; }

    bool empty() const noexcept { return beg_ == end_; }

    private:
        InputIt beg_;
        InputIt end_;
};

} // namespace Operon

#endif
