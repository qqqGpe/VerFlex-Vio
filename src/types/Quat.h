#ifndef __VIO_QUAT__
#define __VIO_QUAT__

#include "Type.h"

class Quat : public Type {
public:
    Quat() : Type(3)
    {
        _quat.setIdentity();
        _quat_fej.setIdentity();
    }

    virtual void set_value(const Eigen::MatrixXd &new_value) override {
        assert(new_value.rows() == 4);
        assert(new_value.cols() == 1);
        _quat = Eigen::Quaterniond(new_value);
        _Rot = _quat.toRotationMatrix();
    }

    virtual void set_fej(const Eigen::MatrixXd &new_fej) override{
        assert(new_fej.rows() == 4);
        assert(new_fej.cols() == 1);
        _quat_fej = Eigen::Quaterniond(new_fej);
        _Rot_fej = _quat_fej.toRotationMatrix();
    }

    virtual void update(const Eigen::VectorXd& d_theta) override
    {
        assert(d_theta.rows() == 3);
        Eigen::Vector4d dq_tmp;
        dq_tmp << 1, 0.5 * d_theta; // w, x, y, z
        // dq_tmp = dq_tmp.eval().normalized();
        Eigen::Quaterniond dq(dq_tmp(0), dq_tmp(1), dq_tmp(2), dq_tmp(3));
        // _quat = Eigen::Quaterniond(_value(0), _value(1), _value(2), _value(3));
        _quat = _quat * dq;
        set_value(Eigen::Vector4d(_quat.w(), _quat.x(), _quat.y(), _quat.z()));
    }

    Eigen::Quaterniond q() const { return _quat; }

    Eigen::Quaterniond q_fej() const { return _quat_fej; }

    Eigen::Matrix3d Rot() const { return _Rot; }

    Eigen::Matrix3d Rot_fej() const { return _Rot_fej; }
    
protected:
    Eigen::Quaterniond _quat;
    Eigen::Quaterniond _quat_fej;
    Eigen::Matrix3d _Rot;
    Eigen::Matrix3d _Rot_fej;
};

#endif