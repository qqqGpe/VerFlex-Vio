#ifndef __VIO_IMU__
#define __VIO_IMU__
#include <memory>
#include "Pose.h"
#include "Type.h"
#include "Vec.h"

class ImuState : public Type
{
public:
    enum class VariableId
    {
        kQ = 0,
        kP = 3,
        kV = 6,
        kBg = 9,
        kBa = 12
    };

    ImuState() : Type(15)
    {
        state_name = "Imu_state";
        _pose = std::make_shared<Pose>();
        _v = std::make_shared<Vec>();
        _bg = std::make_shared<Vec>();
        _ba = std::make_shared<Vec>();
    }

    virtual void reset() override
    {
        _pose->reset();
        _v->reset();
        _bg->reset();
        _ba->reset();
        covariance_ = Eigen::MatrixXd::Zero(15, 15);
        sqrt_Pt_ = Eigen::MatrixXd::Zero(15, 15);
    }

    virtual void set_ts(const double ts) override
    {
        _ts = ts;
        _pose->set_ts(ts);
        _v->set_ts(ts);
        _ba->set_ts(ts);
        _bg->set_ts(ts);
    }

    virtual void update(const Eigen::VectorXd& dx) override
    {
        _pose->update(dx.segment<6>(0));
        _v->update(dx.segment<3>(6));
        _bg->update(dx.segment<3>(9));
        _ba->update(dx.segment<3>(12));
    }

    virtual void set_local_id(int new_id) override
    {
        _id = new_id;
        _pose->set_local_id(new_id);
        _v->set_local_id(_pose->id() + _pose->size());
        _bg->set_local_id(_v->id() + _v->size());
        _ba->set_local_id(_bg->id() + _bg->size());
    }

    virtual void set_value(const Eigen::MatrixXd& new_value) override
    {
        assert(new_value.rows() == 16);
        assert(new_value.cols() == 1);
        _pose->set_value(new_value.block<7, 1>(0, 0));
        _v->set_value(new_value.block<3, 1>(7, 0));
        _bg->set_value(new_value.block<3, 1>(10, 0));
        _ba->set_value(new_value.block<3, 1>(13, 0));
    }

    virtual void set_fej(const Eigen::MatrixXd& new_fej) override
    {
        assert(new_fej.rows() == 16);
        assert(new_fej.cols() == 1);
        _pose->set_fej(new_fej.block<7, 1>(0, 0));
        _v->set_fej(new_fej.block<3, 1>(7, 0));
        _bg->set_fej(new_fej.block<3, 1>(10, 0));
        _ba->set_fej(new_fej.block<3, 1>(13, 0));
    }

    void set_covariance(const Eigen::MatrixXd& covariance_new)
    {
        assert(covariance_new.rows() == 15);
        assert(covariance_new.cols() == 15);
        covariance_.noalias() = covariance_new;
    }

    void set_sqrt_Pt(const Eigen::MatrixXd& sqrt_Pt_new)
    {
        assert(sqrt_Pt_new.rows() == 15);
        assert(sqrt_Pt_new.cols() == 15);
        sqrt_Pt_.noalias() = sqrt_Pt_new;
    }

    void set_pose(const Eigen::Matrix3d& R, const Eigen::Vector3d& p)
    {
        _pose->set_pose(R, p);
    }

    void set_velocity(const Eigen::Vector3d& v)
    {
        _v->set_value(v);
    }

    void set_bg(const Eigen::Vector3d& bg)
    {
        _bg->set_value(bg);
    }

    void set_ba(const Eigen::Vector3d& ba)
    {
        _ba->set_value(ba);
    }

    virtual std::shared_ptr<Type> clone() const override
    {
        std::shared_ptr<ImuState> clone_variable = std::make_shared<ImuState>();
        clone_variable->_pose = std::dynamic_pointer_cast<Pose>(_pose->clone());
        clone_variable->_v = std::dynamic_pointer_cast<Vec>(_v->clone());
        clone_variable->_bg = std::dynamic_pointer_cast<Vec>(_bg->clone());
        clone_variable->_ba = std::dynamic_pointer_cast<Vec>(_ba->clone());

        clone_variable->_pose->set_ts(ts());
        clone_variable->_v->set_ts(ts());
        clone_variable->_bg->set_ts(ts());
        clone_variable->_ba->set_ts(ts());

        assert(clone_variable != nullptr && clone_variable != nullptr);
        return clone_variable;
    }

    Eigen::MatrixXd covariance() const { return covariance_; }
    Eigen::MatrixXd Sqrt_Pt() const { return sqrt_Pt_; }

    std::shared_ptr<Pose> pose() const { return _pose; }
    std::shared_ptr<Quat> q() const { return _pose->_q; }
    std::shared_ptr<Vec> p() const { return _pose->_p; }
    std::shared_ptr<Vec> v() const { return _v; }
    std::shared_ptr<Vec> bg() const { return _bg; }
    std::shared_ptr<Vec> ba() const { return _ba; }

    Eigen::Vector3d last_am = Eigen::Vector3d::Zero();
    Eigen::Vector3d last_wm = Eigen::Vector3d::Zero();
    Eigen::Vector3d gravity_inG = Eigen::Vector3d(0, 0, -9.81);

protected:
    std::shared_ptr<Pose> _pose;
    std::shared_ptr<Vec> _v;
    std::shared_ptr<Vec> _ba;
    std::shared_ptr<Vec> _bg;
    Eigen::MatrixXd covariance_ = Eigen::MatrixXd::Zero(15, 15);
    Eigen::MatrixXd sqrt_Pt_ = Eigen::MatrixXd::Zero(15, 15);
};

#endif