// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research
#ifndef OPERON_SOLVER_SGD_HPP
#define OPERON_SOLVER_SGD_HPP

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <fmt/core.h>
#include <iostream>
#include <ostream>
#include <utility>

namespace Operon {

namespace UpdateRule {
    template <std::floating_point T, typename U = Eigen::Array<T, -1, 1>>
    struct LearningRateUpdateRule {
        // apply the learning rate to the gradient
        virtual auto Update(Eigen::Ref<U const> const& gradient) const -> U = 0;
        virtual auto Update(Eigen::Ref<U const> const& gradient, Eigen::Ref<U> result) const -> void = 0;

        virtual auto Print(std::ostream&) const -> std::ostream& = 0;

        friend auto operator<<(std::ostream& os, LearningRateUpdateRule<T> const& rule) -> std::ostream&
        {
            os << rule.Name() << "\n";
            return rule.Print(os);
        }

        explicit LearningRateUpdateRule(std::string name)
            : name_(std::move(name))
        {
        }

        [[nodiscard]] auto Name() const -> std::string const& { return name_; }

        virtual ~LearningRateUpdateRule() = default;

    private:
        std::string name_;
    };

    template <std::floating_point T, typename U = Eigen::Array<T, -1, 1>>
    class Constant : public LearningRateUpdateRule<T> {
        T r_;

    public:
        Constant() = default;
        explicit Constant(Eigen::Index /*dim*/, T r = 1e-1)
            : LearningRateUpdateRule<T, U>("constant")
            , r_(r)
        {
        }

        auto Update(Eigen::Ref<U const> const& gradient) const -> U final
        {
            U result(gradient.size());
            Update(gradient, result);
            return result;
        }

        auto Update(Eigen::Ref<U const> const& gradient, Eigen::Ref<U> result) const -> void final
        {
            EXPECT(result.size() == gradient.size());
            result = r_ * gradient;
        }

        virtual auto Print(std::ostream& os) const -> std::ostream& final
        {
            os << "constant learning rate update rule\n";
            os << "learning rate: " << r_ << "\n";
            return os;
        }
    };

    template <std::floating_point T, typename U = Eigen::Array<T, -1, 1>>
    class Momentum : public LearningRateUpdateRule<T, U> {
        T r_; // learning rate
        T b_; // beta
        mutable U m_; // first moment

    public:
        explicit Momentum(Eigen::Index dim, T r = 0.01, T b = 0.9)
            : LearningRateUpdateRule<T, U>("momentum")
            , r_ { r }
            , b_ { b }
            , m_ { U::Zero(dim) }
        {
        }

        auto Update(Eigen::Ref<U const> const& gradient) const -> U final
        {
            U result(gradient.size());
            Update(gradient, result);
            return result;
        }

        auto Update(Eigen::Ref<U const> const& gradient, Eigen::Ref<U> result) const -> void final
        {
            EXPECT(result.size() == gradient.size());
            m_ = m_ * b_ + gradient;
            result = r_ * m_;
        }

        virtual auto Print(std::ostream& os) const -> std::ostream& final
        {
            os << "learning rate: " << r_ << "\n"
               << "beta         : " << b_ << "\n"
               << "first moment : " << m_.transpose() << "\n";
            return os;
        }
    };

    template <std::floating_point T, typename U = Eigen::Array<T, -1, 1>>
    class RmsProp : public LearningRateUpdateRule<T, U> {
        T r_; // learning rate
        T b_; // beta
        T e_; // epsilon
        mutable U m_; // second moment

    public:
        explicit RmsProp(Eigen::Index dim, T r = 0.01, T b = 0.9, T e = 1e-6)
            : LearningRateUpdateRule<T, U>("rmsprop")
            , r_ { r }
            , b_ { b }
            , e_ { e }
            , m_ { U::Zero(dim) }
        {
        }

        auto Update(Eigen::Ref<U const> const& gradient) const -> U final
        {
            U result(gradient.size());
            Update(gradient, result);
            return result;
        }

        auto Update(Eigen::Ref<U const> const& gradient, Eigen::Ref<U> result) const -> void final
        {
            EXPECT(result.size() == gradient.size());
            m_ = b_ * m_ + (T { 1 } - b_) * gradient.square();
            result = r_ / (m_.sqrt() + e_) * gradient;
        }

        virtual auto Print(std::ostream& os) const -> std::ostream& final
        {
            os << "learning rate: " << r_ << "\n"
               << "beta         : " << b_ << "\n"
               << "epsilon      : " << e_ << "\n"
               << "moment       : " << m_.transpose() << "\n";
            return os;
        }
    };

    template <std::floating_point T, typename U = Eigen::Array<T, -1, 1>>
    class AdaDelta : public LearningRateUpdateRule<T, U> {
        T b_; // beta
        T e_; // epsilon
        mutable U m_; // moment gradient
        mutable U s_; // moment delta
        mutable U d_; // previous delta

    public:
        explicit AdaDelta(Eigen::Index dim, T b = 0.9, T e = 1e-6)
            : LearningRateUpdateRule<T, U>("adadelta")
            , b_ { b }
            , e_ { e }
            , m_ { U::Zero(dim) }
            , s_ { U::Zero(dim) }
            , d_ { U::Zero(dim) }
        {
        }

        auto Update(Eigen::Ref<U const> const& gradient) const -> U final
        {
            U result(gradient.size());
            Update(gradient, result);
            return result;
        }

        auto Update(Eigen::Ref<U const> const& gradient, Eigen::Ref<U> result) const -> void final
        {
            EXPECT(result.size() == gradient.size());
            m_ = b_ * m_ + (T { 1 } - b_) * gradient.square();
            s_ = b_ * s_ + (T { 1 } - b_) * d_.square();
            d_ = ((s_ + e_) / (m_ + e_)).sqrt() * gradient;
            result = d_;
        }

        virtual auto Print(std::ostream& os) const -> std::ostream& final
        {
            os << "beta         : " << b_ << "\n"
               << "epsilon      : " << e_ << "\n"
               << "moment       : " << m_.transpose() << "\n"
               << "moment delta : " << s_.transpose() << "\n"
               << "prev delta   : " << d_.transpose() << "\n";
            return os;
        }
    };

    template <std::floating_point T, typename U = Eigen::Array<T, -1, 1>>
    class AdaMax : public LearningRateUpdateRule<T, U> {
        T r_; // base learning rate
        T b1_; // exponential decay rate first moment
        T b2_; // exponential decay rate second moment
        mutable U m_; // first moment
        mutable U v_; // second moment

    public:
        explicit AdaMax(Eigen::Index dim, T r = 0.01, T b1 = 0.9, T b2 = 0.999)
            : LearningRateUpdateRule<T, U>("adamax")
            , r_ { r }
            , b1_ { b1 }
            , b2_ { b2 }
            , m_ { U::Zero(dim) }
            , v_ { U::Zero(dim) }
        {
        }

        auto Update(Eigen::Ref<U const> const& gradient) const -> U final
        {
            U result(gradient.size());
            Update(gradient, result);
            return result;
        }

        auto Update(Eigen::Ref<U const> const& gradient, Eigen::Ref<U> result) const -> void final
        {
            EXPECT(result.size() == gradient.size());
            m_ -= (T { 1 } - b1_) * (m_ - gradient);
            v_ = (b2_ * v_).cwiseMax(gradient.abs());
            result = r_ * m_ / v_;
        }

        virtual auto Print(std::ostream& os) const -> std::ostream& final
        {
            os << "learning rate: " << r_ << "\n"
               << "b1           : " << b1_ << "\n"
               << "b2           : " << b2_ << "\n"
               << "m1           : " << m_.transpose() << "\n"
               << "m2           : " << v_.transpose() << "\n";
            return os;
        }
    };

    template <std::floating_point T, typename U = Eigen::Array<T, -1, 1>>
    class Adam : public LearningRateUpdateRule<T, U> {
        T r_; // base learning rate
        T e_; // epsilon
        T b1_; // exponential decay rate first moment
        T b2_; // exponential decay rate second moment
        mutable U m_; // first moment
        mutable U v_; // second moment

        bool debias_ { false };
        mutable uint64_t t_ { 0 }; // time step

    public:
        Adam() = default;
        explicit Adam(Eigen::Index dim, T r = 0.01, T e = 1e-8, T b1 = 0.9, T b2 = 0.999, bool debias = false)
            : LearningRateUpdateRule<T, U>("adam")
            , r_ { r }
            , e_ { e }
            , b1_ { b1 }
            , b2_ { b2 }
            , m_ { U::Zero(dim) }
            , v_ { U::Zero(dim) }
            , debias_ { debias }
        {
        }

        auto Update(Eigen::Ref<U const> const& gradient) const -> U final
        {
            U result(gradient.size());
            Update(gradient, result);
            return result;
        }

        auto Update(Eigen::Ref<U const> const& gradient, Eigen::Ref<U> result) const -> void final
        {
            EXPECT(result.size() == gradient.size());

            ++t_;

            // update moments
            m_ -= (T { 1 } - b1_) * (m_ - gradient);
            v_ -= (T { 1 } - b2_) * (v_ - gradient.square());

            if (debias_) {
                m_ /= (T { 1 } - std::pow(b1_, t_));
                v_ /= (T { 1 } - std::pow(b2_, t_));
            }

            result = r_ * m_ / (v_.sqrt() + e_);
        }

        virtual auto Print(std::ostream& os) const -> std::ostream& final
        {
            os << "adam\n";
            os << "lrate: " << r_ << "\n";
            os << "eps:   " << e_ << "\n";
            os << "m1:    " << m_.transpose() << "\n";
            os << "m2:    " << v_.transpose() << "\n";
            os << "b1:    " << b1_ << "\n";
            os << "b2:    " << b2_ << "\n";
            return os;
        }
    };

    template <std::floating_point T, typename U = Eigen::Array<T, -1, 1>>
    class YamAdam : public LearningRateUpdateRule<T, U> {
        T e_ { 1e-6 }; // epsilon
        mutable U m_ { 0.0 }; // first moment
        mutable U v_ { 0.0 }; // second moment
        mutable U s_ { 0.0 }; // second moment of delta
        mutable U d_ { 0.0 }; // current delta
        mutable U b_ { 0.0 }; // exponential moving average coefficient
        mutable U dp_ { 0.0 }; // previous delta

    public:
        explicit YamAdam(Eigen::Index dim, T e = 1e-6)
            : LearningRateUpdateRule<T, U>("yamadam")
            , e_ { e }
            , m_ { U::Zero(dim) }
            , v_ { U::Zero(dim) }
            , s_ { U::Zero(dim) }
            , d_ { U::Zero(dim) }
            , b_ { U::Zero(dim) }
            , dp_ { U::Zero(dim) }
        {
        }

        auto Update(Eigen::Ref<U const> const& gradient) const -> U final
        {
            U result(gradient.size());
            Update(gradient, result);
            return result;
        }

        auto Update(Eigen::Ref<U const> const& gradient, Eigen::Ref<U> result) const -> void final
        {
            EXPECT(result.size() == gradient.size());
            dp_ = d_;
            m_ = b_ * m_ + (T { 1 } - b_) * gradient;
            v_ = b_ * v_ + (T { 1 } - b_) * (gradient - m_).square();
            s_ = b_ * s_ + (T { 1 } - b_) * d_.square();
            d_ = ((s_ + e_) / (v_ + e_)).sqrt() * m_;
            b_ = (T { 1 } + (d_.abs() + e_) / (dp_.abs() + e_)).inverse().exp() - e_;
            result = d_;
        }

        virtual auto Print(std::ostream& os) const -> std::ostream& final
        {
            os << "eps:   " << e_ << "\n";
            os << "m1:    " << m_.transpose() << "\n";
            os << "m2:    " << v_.transpose() << "\n";
            os << "md:    " << s_.transpose() << "\n";
            os << "d:     " << d_.transpose() << "\n";
            os << "b:     " << b_.transpose() << "\n";
            os << "dp:     " << dp_.transpose() << "\n";
            return os;
        }
    };

    template <std::floating_point T, typename U = Eigen::Array<T, -1, 1>>
    class AmsGrad : public LearningRateUpdateRule<T, U> {
        T r_; // learning rate
        T e_; // epsilon
        T b1_; // exponential decay rate first moment
        T b2_; // exponential decay rate second moment
        mutable U m_; // first moment
        mutable U v_; // second moment

    public:
        explicit AmsGrad(Eigen::Index dim, T r = 0.01, T e = 1e-6, T b1 = 0.9, T b2 = 0.999)
            : LearningRateUpdateRule<T, U>("amsgrad")
            , r_ { r }
            , e_ { e }
            , b1_ { b1 }
            , b2_ { b2 }
            , m_ { U::Zero(dim) }
            , v_ { U::Zero(dim) }
        {
        }

        auto Update(Eigen::Ref<U const> const& gradient) const -> U final
        {
            U result(gradient.size());
            Update(gradient, result);
            return result;
        }

        auto Update(Eigen::Ref<U const> const& gradient, Eigen::Ref<U> result) const -> void final
        {
            EXPECT(result.size() == gradient.size());
            m_ = b1_ * m_ + (T{1} - b1_) * gradient;
            v_ = (b2_ * v_ + (T{1} - b2_)).cwiseMax(v_) * gradient.square();
            result = r_ * m_ / (v_.sqrt() + e_);
        }

        virtual auto Print(std::ostream& os) const -> std::ostream& final
        {
            os << "lrate: " << r_ << "\n"
               << "eps  : " << e_ << "\n"
               << "m1   : " << m_.transpose() << "\n"
               << "m2   : " << v_.transpose() << "\n"
               << "b1   : " << b1_ << "\n"
               << "b2   : " << b2_ << "\n";
            return os;
        }
    };

    template <std::floating_point T, typename U = Eigen::Array<T, -1, 1>>
    class Yogi : public LearningRateUpdateRule<T, U> {
        T r_; // learning rate
        T e_; // epsilon
        U b1_; // exponential decay rate first moment
        U b2_; // exponential decay rate second moment
        mutable U m_; // first moment
        mutable U v_; // second moment

        bool debias_ { false };
        mutable uint64_t t_ { 0 }; // time step

    public:
        explicit Yogi(Eigen::Index dim, T r = 0.01, T e = 1e-8, T b1 = 0.9, T b2 = 0.999, bool debias = false)
            : LearningRateUpdateRule<T, U>("yogi")
            , r_ { r }
            , e_ { e }
            , b1_ { U::Constant(dim, b1) }
            , b2_ { U::Constant(dim, b2) }
            , m_ { U::Zero(dim) }
            , v_ { U::Zero(dim) }
            , debias_ { debias }
        {
        }

        auto Update(Eigen::Ref<U const> const& gradient) const -> U final
        {
            U result(gradient.size());
            Update(gradient, result);
            return result;
        }

        auto Update(Eigen::Ref<U const> const& gradient, Eigen::Ref<U> result) const -> void final
        {
            EXPECT(result.size() == gradient.size());
            ++t_;

            m_ -= (T { 1 } - b1_) * (m_ - gradient);
            v_ -= (T { 1 } - b2_) * (v_ - gradient.square()).sign() * gradient.square();

            if (debias_) {
                m_ /= (T { 1 } - b1_.pow(t_));
                v_ /= (T { 1 } - b2_.pow(t_));
            }

            result = r_ * m_ / (v_.sqrt() + e_);
        }

        virtual auto Print(std::ostream& os) const -> std::ostream& final
        {
            os << "lrate: " << r_ << "\n"
               << "eps  : " << e_ << "\n"
               << "m1   : " << m_.transpose() << "\n"
               << "m2   : " << v_.transpose() << "\n"
               << "b1   : " << b1_.transpose() << "\n"
               << "b2   : " << b2_.transpose() << "\n";
            return os;
        }
    };
} // namespace UpdateRule

template <typename Functor, typename Update>
struct SGDSolver {
    using Scalar = typename Functor::Scalar;
    using Vector = Eigen::Array<Scalar, -1, 1>;

    explicit SGDSolver(Functor const& functor, Update const& update)
        : functor_(functor)
        , update_(update)
    {
    }

    auto Optimize(Eigen::Ref<Vector const> const x0, int epochs = 1000) const noexcept -> Vector
    {
        auto const& fun = functor_.get();

        auto const np = fun.NumParameters();
        auto const no = fun.NumObservations();

        EXPECT(x0.size() == np);

        Vector grad(np);
        Vector x = x0;
        Vector beta(np);

        static constexpr auto tol { 1e-8 };

        for (epochs_ = 0; epochs_ < epochs; ++epochs_) {
            auto f = functor_(x, grad);
            // apply learning rate to grad and write result in beta
            update_.get().Update(grad, beta);
            if ((beta.abs() < tol).all()) {
                converged_ = true;
                break;
            }
            x -= beta;
            // std::cout << "===== epoch " << epochs_ << " =====\n";
            // std::cout << "x   : " << x.transpose() << "\n";
            // std::cout << "grad: " << grad.transpose() << "\n";
            // std::cout << "beta: " << beta.transpose() << "\n";
            // std::cout << update_.get() << "\n";
        }

        return x;
    }

    auto Epochs() const { return epochs_; }

private:
    std::reference_wrapper<Functor const> functor_;
    std::reference_wrapper<Update const> update_;

    mutable int epochs_ { 0 };
    mutable bool converged_ { false };
};
} // namespace Operon

#endif
