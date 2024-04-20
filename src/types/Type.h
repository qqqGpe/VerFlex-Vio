#ifndef __VIO_TYPES__
#define __VIO_TYPES__

#include <Eigen/Eigen>
#include <Eigen/Geometry>

class Type {
public:
    Type(int size)
        : _size(size)
    {
    }

    virtual void set_local_id(int new_id) { _id = new_id; }

    virtual int id() { return _id; }

    int size() { return _size; }

    virtual void update(const Eigen::VectorXd& dx) = 0;

    virtual const Eigen::MatrixXd& value() const { return _value; }

    virtual const Eigen::MatrixXd& fej() { return _fej; }

    virtual void set_value(const Eigen::MatrixXd &new_value) {
        assert(new_value.rows() == _value.rows());
        assert(new_value.cols() == _value.cols());
        _value = new_value;
    }

    virtual void set_fej(const Eigen::MatrixXd &new_fej) {
        assert(new_fej.rows() == _fej.rows());
        assert(new_fej.cols() == _fej.cols());
        _fej = new_fej;
    }

protected:
    int _size = -1;
    int _id = -1;
    Eigen::MatrixXd _value;
    Eigen::MatrixXd _fej;
};

#endif