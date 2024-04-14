#ifndef __VIO_BACKEND_STATE__
#define __VIO_BACKEND_STATE__
#include <Eigen/Eigen>
#include <memory>
#include "types/Quat.h"
#include "types/Vec.h"

class State {
public:

    State(){}
    ~State(){}

    std::shared_ptr<Vec> _p, _v, _ba, _bg;
    std::shared_ptr<Quat> _q;
};

#endif