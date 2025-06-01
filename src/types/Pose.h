#ifndef __VIO_POSE__
#define __VIO_POSE__

#include <memory>
#include "Quat.h"
#include "Type.h"
#include "Vec.h"

class Pose : public Type
{
   public:
    Pose() : Type(6)
    {
        state_name = "Pose";
        _q = std::make_shared<Quat>();
        _p = std::make_shared<Vec>();
    }

    Pose(const Eigen::Matrix3d R, const Eigen::Vector3d t) : Type(6)
    {
        state_name = "Pose";
        _q = std::make_shared<Quat>();
        _p = std::make_shared<Vec>();
        _q->set_value(Eigen::Quaterniond(R).coeffs());
        _p->set_value(t);
    }

    virtual void reset() override
    {
        _q->reset();
        _p->reset();
    }

    virtual void set_ts(const double ts) override
    {
        if (ts <= 0)
        {
            LOG(FATAL) << utils::Format("timestamp should > 0, ts: {0}s", ts);
            return;
        }
        _ts = ts;
        _q->set_ts(ts);
        _p->set_ts(ts);
    }

    virtual void update(const Eigen::VectorXd& dx) override
    {
        assert(dx.rows() == _size);
        _q->update(dx.segment<3>(0));
        _p->update(dx.segment<3>(_q->size()));
    }

    virtual void set_local_id(int new_id) override
    {
        _id = new_id;
        _q->set_local_id(new_id);
        _p->set_local_id(new_id + _q->size());
    }

    // eigen::vector(x, y, z, w)
    virtual void set_value(const Eigen::MatrixXd& new_value) override
    {
        assert(new_value.rows() == 7);
        assert(new_value.cols() == 1);
        _q->set_value(new_value.block<4, 1>(0, 0));
        _p->set_value(new_value.block<3, 1>(4, 0));
    }

    void set_pose(const Eigen::Matrix3d R, const Eigen::Vector3d t)
    {
        _q->set_value(Eigen::Quaterniond(R).coeffs());
        _p->set_value(t);
    }

    virtual std::shared_ptr<Type> clone() override
    {
        std::shared_ptr<Pose> clone_variable = std::make_shared<Pose>();
        clone_variable->_q = std::dynamic_pointer_cast<Quat>(_q->clone());
        clone_variable->_p = std::dynamic_pointer_cast<Vec>(_p->clone());
        clone_variable->set_ts(ts());
        assert(clone_variable != nullptr && clone_variable != nullptr);
        return clone_variable;
    }

    Eigen::Quaterniond quat() const { return _q->q(); }
    Eigen::Matrix3d R() const { return quat().toRotationMatrix(); }
    Eigen::Vector3d p() const { return _p->vec(); }

    Eigen::Quaterniond quat_fej() const { return _q->q_fej(); }
    Eigen::Vector3d p_fej() const { return _p->fej(); }

    friend class ImuState;

   protected:
    std::shared_ptr<Quat> _q;
    std::shared_ptr<Vec> _p;
};

#endif