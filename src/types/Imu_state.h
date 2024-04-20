#ifndef __VIO_IMU__
#define __VIO_IMU__
#include <memory>
#include "Type.h"
#include "Vec.h"
#include "Pose.h"

class IMU_state : public Type {
public:
    IMU_state() : Type(15) {
        _pose = std::make_shared<Pose>();
        _v = std::make_shared<Vec>();
        _bg = std::make_shared<Vec>();
        _ba = std::make_shared<Vec>();
    }

    virtual void update(const Eigen::VectorXd &dx) override {
        _pose->update(dx.segment<6>(0));
        _v->update(dx.segment<3>(6));
        _bg->update(dx.segment<3>(9));
        _ba->update(dx.segment<3>(12));
    }

    virtual void set_local_id(int new_id) override {
        _id = new_id;
        _pose->set_local_id(new_id);
        _v->set_local_id(_pose->id() + _pose->size());
        _bg->set_local_id(_v->id() + _v->size());
        _ba->set_local_id(_bg->id() + _bg->size());
    }

    virtual void set_value(const Eigen::MatrixXd &new_value) override {
        assert(new_value.rows() == 16);
        assert(new_value.cols() == 1);
        _pose->set_value(new_value.block<7, 1>(0, 0));
        _v->set_value(new_value.block<3, 1>(7, 0));
        _bg->set_value(new_value.block<3, 1>(10, 0));
        _ba->set_value(new_value.block<3, 1>(13, 0));
    }

    virtual void set_fej(const Eigen::MatrixXd &new_fej) override {
        assert(new_fej.rows() == 16);
        assert(new_fej.cols() == 1);
        _pose->set_fej(new_fej.block<7, 1>(0, 0));
        _v->set_fej(new_fej.block<3, 1>(7, 0));
        _bg->set_fej(new_fej.block<3, 1>(10, 0));
        _ba->set_fej(new_fej.block<3, 1>(13, 0));
    }

    void set_covariance(const Eigen::MatrixXd &covariance_new) {
        assert(covariance_new.rows() == 15);
        assert(covariance_new.cols() == 15);
        covariance = covariance_new;
    }

    std::shared_ptr<Pose> pose() const { return _pose; }
    std::shared_ptr<Vec> v() const { return _v; }
    std::shared_ptr<Vec> bg() const { return _bg; }
    std::shared_ptr<Vec> ba() const { return _ba; }

 protected:
    std::shared_ptr<Pose> _pose;
    std::shared_ptr<Vec> _v;
    std::shared_ptr<Vec> _ba;
    std::shared_ptr<Vec> _bg;
    Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(15, 15);
};

#endif