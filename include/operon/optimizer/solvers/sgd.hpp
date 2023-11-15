// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research
#ifndef OPERON_SOLVER_SGD_HPP
#define OPERON_SOLVER_SGD_HPP

#include "operon/core/types.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <fmt/core.h>
#include <iostream>
#include <memory>
#include <ostream>
#include <utility>

namespace Operon {

namespace UpdateRule {
    struct LearningRateUpdateRule {
        using T = Operon::Scalar;
        using U = Eigen::Array<T, -1, 1>;

        // apply the learning rate to the gradient
        [[nodiscard]] virtual auto Update(Eigen::Ref<U const> const& gradient) const -> U = 0;
        virtual auto Update(Eigen::Ref<U const> const& gradient, Eigen::Ref<U> result) const -> void = 0;

        virtual auto Print(std::ostream&) const -> std::ostream& = 0;
        virtual auto SetDimension(int dim) const -> void = 0;
        [[nodiscard]] virtual auto Clone(int dim) const -> std::unique_ptr<LearningRateUpdateRule> = 0;

        friend auto operator<<(std::ostream& os, LearningRateUpdateRule const& rule) -> std::ostream&
        {
            os << rule.Name() << "\n";
            return rule.Print(os);
        }

        LearningRateUpdateRule(const LearningRateUpdateRule&) = delete;
        LearningRateUpdateRule(LearningRateUpdateRule&&) = delete;
        auto operator=(const LearningRateUpdateRule&) -> LearningRateUpdateRule& = delete;
        auto operator=(LearningRateUpdateRule&&) -> LearningRateUpdateRule& = delete;

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
    class Constant : public LearningRateUpdateRule {
    public:
        using Scalar = T;
        using Vector = U;

    private:
        using Base = LearningRateUpdateRule;

        T r_{0.1};

    public:
        explicit Constant(Eigen::Index /*dim*/= 0, T r = 0.1)
            : Base("constant")
            , r_(r)
        {
        }

        explicit Constant(T r = 0.1) : Constant(0, r)
        {
        }

        [[nodiscard]] auto Update(Eigen::Ref<U const> const& gradient) const -> U final
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

        auto Print(std::ostream& os) const -> std::ostream& final
        {
            os << "constant learning rate update rule\n";
            os << "learning rate: " << r_ << "\n";
            return os;
        }

        auto SetDimension(int /*unused*/) const -> void final { }

        [[nodiscard]] auto Clone(int dim) const -> std::unique_ptr<LearningRateUpdateRule> final {
            return std::make_unique<Constant<T, U>>(dim, r_);
        }
    };

    template <std::floating_point T, typename U = Eigen::Array<T, -1, 1>>
    class Momentum : public LearningRateUpdateRule {
        using Base = LearningRateUpdateRule;

        T r_{0.01}; // learning rate
        T b_{0.9}; // beta
        mutable U m_; // first moment

    public:
        using Scalar = T;
        using Vector = U;

        explicit Momentum(Eigen::Index dim, T r = 0.01, T b = 0.9)
            : Base("momentum")
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

        auto Print(std::ostream& os) const -> std::ostream& final
        {
            os << "learning rate: " << r_ << "\n"
               << "beta         : " << b_ << "\n"
               << "first moment : " << m_.transpose() << "\n";
            return os;
        }

        auto SetDimension(int dim) const -> void final
        {
            if (m_.size() != dim) {
                m_ = U::Zero(dim);
            }
        }

        auto Clone(int dim) const -> std::unique_ptr<LearningRateUpdateRule> final
        {
            if (dim == 0) { dim = m_.size(); }
            return std::make_unique<Momentum<T, U>>(dim, r_, b_);
        }
    };

    template <std::floating_point T, typename U = Eigen::Array<T, -1, 1>>
    class RmsProp : public LearningRateUpdateRule {
        using Base = LearningRateUpdateRule;

        T r_{0.01}; // learning rate
        T b_{0.9};  // beta
        T e_{1e-6}; // epsilon
        mutable U m_; // second moment

    public:
        using Scalar = T;
        using Vector = U;

        explicit RmsProp(Eigen::Index dim, T r = 0.01, T b = 0.9, T e = 1e-6)
            : Base("rmsprop")
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

        auto Print(std::ostream& os) const -> std::ostream& final
        {
            os << "learning rate: " << r_ << "\n"
               << "beta         : " << b_ << "\n"
               << "epsilon      : " << e_ << "\n"
               << "moment       : " << m_.transpose() << "\n";
            return os;
        }

        auto SetDimension(int dim) const -> void final
        {
            if (m_.size() != dim) {
                m_ = U::Zero(dim);
            }
        }

        auto Clone(int dim) const -> std::unique_ptr<LearningRateUpdateRule> final
        {
            if (dim == 0) { dim = m_.size(); }
            return std::make_unique<RmsProp<T, U>>(dim, r_, b_);
        }
    };

    template <std::floating_point T, typename U = Eigen::Array<T, -1, 1>>
    class AdaDelta : public LearningRateUpdateRule {
        using Base = LearningRateUpdateRule;

        T b_{0.9}; // beta
        T e_{1e-6}; // epsilon
        mutable U m_; // moment gradient
        mutable U s_; // moment delta
        mutable U d_; // previous delta

    public:
        using Scalar = T;
        using Vector = U;

        explicit AdaDelta(Eigen::Index dim, T b = 0.9, T e = 1e-6)
            : Base("adadelta")
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

        auto Print(std::ostream& os) const -> std::ostream& final
        {
            os << "beta         : " << b_ << "\n"
               << "epsilon      : " << e_ << "\n"
               << "moment       : " << m_.transpose() << "\n"
               << "moment delta : " << s_.transpose() << "\n"
               << "prev delta   : " << d_.transpose() << "\n";
            return os;
        }

        auto SetDimension(int dim) const -> void final
        {
            if (m_.size() != dim) {
                m_ = U::Zero(dim);
                s_ = U::Zero(dim);
                d_ = U::Zero(dim);
            }
        }

        auto Clone(int dim) const -> std::unique_ptr<LearningRateUpdateRule> final
        {
            if (dim == 0) { dim = m_.size(); }
            return std::make_unique<AdaDelta<T, U>>(dim, b_, e_);
        }
    };

    template <std::floating_point T, typename U = Eigen::Array<T, -1, 1>>
    class AdaMax : public LearningRateUpdateRule {
        using Base = LearningRateUpdateRule;

        T r_{0.01}; // base learning rate
        T b1_{0.9}; // exponential decay rate first moment
        T b2_{0.999}; // exponential decay rate second moment
        mutable U m_; // first moment
        mutable U v_; // second moment

    public:
        using Scalar = T;
        using Vector = U;

        explicit AdaMax(Eigen::Index dim, T r = 0.01, T b1 = 0.9, T b2 = 0.999)
            : Base("adamax")
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

        auto Print(std::ostream& os) const -> std::ostream& final
        {
            os << "learning rate: " << r_ << "\n"
               << "b1           : " << b1_ << "\n"
               << "b2           : " << b2_ << "\n"
               << "m1           : " << m_.transpose() << "\n"
               << "m2           : " << v_.transpose() << "\n";
            return os;
        }

        auto SetDimension(int dim) const -> void final
        {
            if (m_.size() != dim) {
                m_ = U::Zero(dim);
                v_ = U::Zero(dim);
            }
        }

        auto Clone(int dim) const -> std::unique_ptr<LearningRateUpdateRule> final
        {
            if (dim == 0) { dim = m_.size(); }
            return std::make_unique<AdaMax<T, U>>(dim, r_, b1_, b2_);
        }
    };

    template <std::floating_point T, typename U = Eigen::Array<T, -1, 1>>
    class Adam : public LearningRateUpdateRule {
        using Base = LearningRateUpdateRule;

        T r_{0.01}; // base learning rate
        T e_{1e-8}; // epsilon
        T b1_{0.9}; // exponential decay rate first moment
        T b2_{0.999}; // exponential decay rate second moment
        mutable U m_; // first moment
        mutable U v_; // second moment

        bool debias_ { false };
        mutable uint64_t t_ { 0 }; // time step

    public:
        using Scalar = T;
        using Vector = U;

        explicit Adam(Eigen::Index dim, T r = 0.01, T e = 1e-8, T b1 = 0.9, T b2 = 0.999, bool debias = false)
            : Base("adam")
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

        auto Print(std::ostream& os) const -> std::ostream& final
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

        auto SetDimension(int dim) const -> void final
        {
            if (m_.size() != dim) {
                m_ = U::Zero(dim);
                v_ = U::Zero(dim);
            }
        }

        auto Clone(int dim) const -> std::unique_ptr<LearningRateUpdateRule> final
        {
           if (dim == 0) { dim = m_.size(); }
           return std::make_unique<Adam<T, U>>(dim, r_, e_, b1_, b2_, debias_);
        }
    };

    template <std::floating_point T, typename U = Eigen::Array<T, -1, 1>>
    class YamAdam : public LearningRateUpdateRule {
        using Base = LearningRateUpdateRule;

        T e_ { 1e-6 }; // epsilon
        mutable U m_ { 0.0 }; // first moment
        mutable U v_ { 0.0 }; // second moment
        mutable U s_ { 0.0 }; // second moment of delta
        mutable U d_ { 0.0 }; // current delta
        mutable U b_ { 0.0 }; // exponential moving average coefficient
        mutable U dp_ { 0.0 }; // previous delta

    public:
        explicit YamAdam(Eigen::Index dim, T e = 1e-6)
            : Base("yamadam")
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

        auto Print(std::ostream& os) const -> std::ostream& final
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

        auto SetDimension(int dim) const -> void final
        {
            if (m_.size() != dim) {
                m_ = U::Zero(dim);
                v_ = U::Zero(dim);
                s_ = U::Zero(dim);
                d_ = U::Zero(dim);
                b_ = U::Zero(dim);
                dp_ = U::Zero(dim);
            }
        }

        auto Clone(int dim) const -> std::unique_ptr<LearningRateUpdateRule> final
        {
            if (dim == 0) { dim = m_.size(); }
            return std::make_unique<YamAdam<T, U>>(dim, e_);
        }
    };

    template <std::floating_point T, typename U = Eigen::Array<T, -1, 1>>
    class AmsGrad : public LearningRateUpdateRule {
        using Base = LearningRateUpdateRule;

        T r_; // learning rate
        T e_; // epsilon
        T b1_; // exponential decay rate first moment
        T b2_; // exponential decay rate second moment
        mutable U m_; // first moment
        mutable U v_; // second moment

    public:
        explicit AmsGrad(Eigen::Index dim, T r = 0.01, T e = 1e-6, T b1 = 0.9, T b2 = 0.999)
            : Base("amsgrad")
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

        auto Print(std::ostream& os) const -> std::ostream& final
        {
            os << "lrate: " << r_ << "\n"
               << "eps  : " << e_ << "\n"
               << "m1   : " << m_.transpose() << "\n"
               << "m2   : " << v_.transpose() << "\n"
               << "b1   : " << b1_ << "\n"
               << "b2   : " << b2_ << "\n";
            return os;
        }

        auto SetDimension(int dim) const -> void final
        {
            if (m_.size() != dim) {
                m_ = U::Zero(dim);
                v_ = U::Zero(dim);
            }
        }

        auto Clone(int dim) const -> std::unique_ptr<LearningRateUpdateRule> final
        {
            if (dim == 0) { dim = m_.size(); }
            return std::make_unique<AmsGrad<T,U>>(dim, r_, e_, b1_, b2_);
        }
    };

    template <std::floating_point T, typename U = Eigen::Array<T, -1, 1>>
    class Yogi : public LearningRateUpdateRule {
        using Base = LearningRateUpdateRule;

        T r_{0.01}; // learning rate
        T e_{1e-8}; // epsilon
        T b1_{0.9}; // exponential decay rate first moment
        T b2_{0.999}; // exponential decay rate second moment
        mutable U m_; // first moment
        mutable U v_; // second moment

        bool debias_ { false };
        mutable uint64_t t_ { 0 }; // time step

    public:
        explicit Yogi(Eigen::Index dim, T r = 0.01, T e = 1e-8, T b1 = 0.9, T b2 = 0.999, bool debias = false)
            : Base("yogi")
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

            m_ -= (T { 1 } - b1_) * (m_ - gradient);
            v_ -= (T { 1 } - b2_) * (v_ - gradient.square()).sign() * gradient.square();

            if (debias_) {
                m_ /= (T { 1 } - std::pow(b1_, t_));
                v_ /= (T { 1 } - std::pow(b2_, t_));
            }

            result = r_ * m_ / (v_.sqrt() + e_);
        }

        auto Print(std::ostream& os) const -> std::ostream& final
        {
            os << "lrate: " << r_ << "\n"
               << "eps  : " << e_ << "\n"
               << "m1   : " << m_.transpose() << "\n"
               << "m2   : " << v_.transpose() << "\n"
               << "b1   : " << b1_ << "\n"
               << "b2   : " << b2_ << "\n";
            return os;
        }

        auto SetDimension(int dim) const -> void final
        {
            if (m_.size() != dim) {
                m_ = U::Zero(dim);
                v_ = U::Zero(dim);
            }
        }

        auto Clone(int dim) const -> std::unique_ptr<LearningRateUpdateRule> final
        {
            if (dim == 0) { dim = m_.size(); }
            return std::make_unique<Yogi<T, U>>(dim, r_, e_, b1_, b2_, debias_);
        }
    };
} // namespace UpdateRule

template <typename Functor>
struct SGDSolver {
    using Scalar = typename Functor::Scalar;
    using Vector = Eigen::Array<Scalar, -1, 1>;

    explicit SGDSolver(Functor const& functor, UpdateRule::LearningRateUpdateRule const& update)
        : functor_(functor)
        , update_(update)
    {
    }

    auto Optimize(Eigen::Ref<Vector const> const x0, int epochs = 1000) const noexcept -> Vector
    {
        auto const& fun = functor_.get();

        EXPECT(x0.size() == static_cast<Eigen::Index>(fun.NumParameters()));

        Vector grad(x0.size());
        Vector x = x0;
        Vector beta(x0.size());

        static constexpr auto tol { 1e-8 };

        for (epochs_ = 0; epochs_ < epochs; ++epochs_) {
            fun(x, grad);
            // apply learning rate to grad and write result in beta
            update_.get().Update(grad, beta);
            if ((beta.abs() < tol).all()) {
                converged_ = true;
                break;
            }
            x -= beta;
        }

        return x;
    }

    auto Epochs() const { return epochs_; }

private:
    std::reference_wrapper<Functor const> functor_;
    std::reference_wrapper<UpdateRule::LearningRateUpdateRule const> update_;

    mutable int epochs_ {1000};
    mutable bool converged_ { false };
};
} // namespace Operon

#endif
