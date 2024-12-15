#ifndef __VIO_VEC__
#define __VIO_VEC__

#include "Type.h"

class Vec : public Type {
public:
    Vec() : Type(3) {
        state_name = "Vec";
        _value = Eigen::VectorXd::Zero(3);
        _fej = Eigen::VectorXd::Zero(3);
    }

    virtual void update(const Eigen::VectorXd& dx) override{
        assert(dx.rows() == _size);
        set_value(_value + dx);
    }

    virtual std::shared_ptr<Type> clone() override
    {
        std::shared_ptr<Vec> clone_variable = std::make_shared<Vec>();
        clone_variable->set_value(this->value());
        clone_variable->set_fej(this->fej());
        return clone_variable;
    }

    Eigen::VectorXd vec() const { return this->value(); }
};

#endif