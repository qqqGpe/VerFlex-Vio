#ifndef __VIO_QUAT__
#define __VIO_QUAT__

#include "Type.h"

class Quat : public Type {
public:
    Quat() : Type(4)
    {
        _value = Eigen::VectorXd::Zero(4);
        _fej = Eigen::VectorXd::Zero(4);
    }

    virtual void update(const Eigen::VectorXd& d_theta)
    {
        assert(d_theta.rows() == 3);
        Eigen::Vector4d dq_tmp;
        dq_tmp << 1, 0.5 * d_theta; // w, x, y, z
        dq_tmp = dq_tmp.eval().normalized();
        Eigen::Quaterniond dq(dq_tmp(0), dq_tmp(1), dq_tmp(2), dq_tmp(3));
        Eigen::Quaterniond q(_value(0), _value(1), _value(2), _value(3));
        q = q * dq;
        q = q.normalized();
        set_value(Eigen::Vector4d(q.w(), q.x(), q.y(), q.z()));
    }
};

#endif