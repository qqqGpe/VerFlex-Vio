#ifndef __SCALAR_TYPE__
#define __SCALAR_TYPE__

#include "Type.h"

class Scalar : public Type
{
   public:
    Scalar() : Type(1)
    {
        state_name = "Scalar";
        _value = Eigen::VectorXd::Zero(1);
        _fej = Eigen::VectorXd::Zero(1);
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
        std::shared_ptr<Scalar> clone_variable = std::make_shared<Scalar>();
        clone_variable->set_value(this->value());
        // clone_variable->set_fej(this->fej());
        return clone_variable;
    }

    const double data() const { return this->value()(0, 0); }
};

#endif // __SCALAR_TYPE__