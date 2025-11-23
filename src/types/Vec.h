/*
 * @Author: pengen.gao gaope.hb@gmail.com
 * @Date: 2025-11-07 01:40:59
 * Copyright (c) 2025 by gaope.hb@gmail.com, All Rights Reserved.
 */
#ifndef __VIO_VEC__
#define __VIO_VEC__

#include "Type.h"

class Vec : public Type
{
   public:
    Vec() : Type(3)
    {
        state_name = "Vec";
        _value = Eigen::VectorXd::Zero(3);
        _fej = Eigen::VectorXd::Zero(3);
    }

    Vec(Eigen::VectorXd value) : Type(static_cast<uint32_t>(value.rows()))
    {
        state_name = "Vec";
        _value = value;
        _fej = value;
    }

    virtual void reset() override
    {
        _value.setZero();
        _fej.setZero();
    }

    virtual void update(const Eigen::VectorXd& dx) override
    {
        assert(dx.rows() == _size);
        set_value(_value + dx);
    }

    virtual std::shared_ptr<Type> clone() const override
    {
        std::shared_ptr<Vec> clone_variable = std::make_shared<Vec>();
        clone_variable->set_value(this->value());
        clone_variable->set_fej(this->fej());
        clone_variable->set_ts(this->ts());
        return clone_variable;
    }

    Eigen::VectorXd vec() const { return this->value(); }
};

#endif