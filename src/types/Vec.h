#ifndef __VIO_VEC__
#define __VIO_VEC__

#include "Type.h"

class Vec : public Type {
public:
    Vec() : Type(3) {
        _value = Eigen::VectorXd::Zero(3);
        _fej = Eigen::VectorXd::Zero(3);
    }

    virtual void update(const Eigen::VectorXd& dx) override{
        assert(dx.rows() == _size);
        set_value(_value + dx);
    }

    Eigen::VectorXd vec() const { return this->value(); }
};

#endif