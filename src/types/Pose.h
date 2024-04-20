#ifndef __VIO_POSE__
#define __VIO_POSE__

#include <memory>
#include "Type.h"
#include "Quat.h"
#include "Vec.h"

class Pose : public Type {
public:
    Pose() : Type(6) {
        _q = std::make_shared<Quat>();
        _p = std::make_shared<Vec>();
    }

    virtual void update(const Eigen::VectorXd &dx) override{
        _q->update(dx.segment<3>(0));
        _p->update(dx.segment<3>(_q->size()));
    }

    virtual void set_local_id(int new_id) override {
        _id = new_id;
        _q->set_local_id(new_id);
        _p->set_local_id(new_id + _q->size());
    }

    virtual void set_value(const Eigen::MatrixXd &new_value) override {
        assert(new_value.rows() == 7);
        assert(new_value.cols() == 1);
        _q->set_value(new_value.segment<4>(0));
        _p->set_value(new_value.segment<3>(4));
    }

    Eigen::Quaterniond quat() const {return _q->q(); }

    Eigen::Vector3d p() const {return _p->vec(); }

protected:
    std::shared_ptr<Quat> _q;
    std::shared_ptr<Vec> _p;
};

#endif