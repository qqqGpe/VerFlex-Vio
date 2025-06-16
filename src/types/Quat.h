#ifndef __VIO_QUAT__
#define __VIO_QUAT__

#include <sophus/so3.hpp>
#include "Type.h"
#include "mathematical_tools.h"

class Quat : public Type
{
   public:
    Quat() : Type(3)
    {
        state_name = "Quat";
        _quat.setIdentity();
        _quat_fej.setIdentity();
    }

    virtual void reset() override
    {
        _quat.setIdentity();
        _quat_fej.setIdentity();
        set_value(_quat.coeffs());
        // _fej.setZero();
    }

    virtual void set_value(const Eigen::MatrixXd& new_value) override
    {
        assert(new_value.rows() == 4);
        assert(new_value.cols() == 1);
        _quat = Eigen::Quaterniond(new_value.block<4, 1>(0, 0));
        _Rot = _quat.toRotationMatrix();
        _value = _quat.coeffs();
    }

    virtual void set_fej(const Eigen::MatrixXd& new_fej) override
    {
        assert(new_fej.rows() == 4);
        assert(new_fej.cols() == 1);
        _quat_fej = Eigen::Quaterniond(new_fej.block<4, 1>(0, 0));
        _Rot_fej = _quat_fej.toRotationMatrix();
        _fej << _quat_fej.w(), _quat_fej.x(), _quat_fej.y(), _quat_fej.z();
    }

    virtual void update(const Eigen::VectorXd& d_theta) override
    {
        assert(d_theta.rows() == 3);
        Eigen::Matrix3d dR = Sophus::SO3d::exp(d_theta).matrix();
        Eigen::Quaterniond dq(dR);
        _quat = _quat * dq;
        set_value(_quat.coeffs());
    }

    virtual std::shared_ptr<Type> clone() const override
    {
        std::shared_ptr<Quat> clone_variable = std::make_shared<Quat>();
        clone_variable->set_value(_quat.coeffs());
        // clone_variable->set_fej(this->fej());    // disable fej for debug
        return clone_variable;
    }

    Eigen::Quaterniond q() const { return _quat; }

    Eigen::Quaterniond q_fej() const { return _quat_fej; }

    Eigen::Matrix3d Rot() const { return _Rot; }

    Eigen::Matrix3d Rot_fej() const { return _Rot_fej; }

    Eigen::Vector3d rpy()
    {
        return MathUtils::R2rpy(_quat.toRotationMatrix()) * 180.0 / M_PI;
    }

   protected:
    Eigen::Quaterniond _quat;
    Eigen::Quaterniond _quat_fej;
    Eigen::Matrix3d _Rot;
    Eigen::Matrix3d _Rot_fej;
};

#endif