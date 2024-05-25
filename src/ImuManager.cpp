#include "ImuManager.h"
#include <glog/logging.h>

namespace {
    constexpr int kMaxImuBufferSize = 1000;
}

bool ImuManager::feed_imu_measurement(const ImuData &imu_measurement) {

    if(_data->empty()) {
        _data->push_back(imu_measurement);
        return true;
    }

    if(imu_measurement.ts_sec <= _data->back().ts_sec) {
        LOG(WARNING) << "latest imu data ts: " << _data->back().ts_sec << ", input imu ts: " << imu_measurement.ts_sec;
        return false;
    } else {
        if(_data->size() < kMaxImuBufferSize) {
            _data->push_back(imu_measurement);
        } else {
            _data->pop_front();
            _data->push_back(imu_measurement);
        }
    }

    return true;
}

ImuData ImuManager::interpolate_data(const ImuData &imu_1, const ImuData &imu_2, double timestamp) {
    // time-distance lambda
    double lambda = (timestamp - imu_1.ts_sec) / (imu_2.ts_sec - imu_1.ts_sec);
    // PRINT_DEBUG("lambda - %d\n", lambda);
    // interpolate between the two times
    ImuData data;
    data.ts_sec = timestamp;
    data.am = (1 - lambda) * imu_1.am + lambda * imu_2.am;
    data.wm = (1 - lambda) * imu_1.wm + lambda * imu_2.wm;
    return data;
}


std::vector<ImuData> ImuManager::access_interval_imu_measurment(const double ts_start, const double ts_end) {

    std::vector<ImuData> output;
    if(ts_end <= ts_start) {
        LOG(ERROR) << "ts_end: " << ts_end << ", should later than ts_start: " << ts_start;
        return output;
    }

    for(int i = 0; i < _data->size(); i++) {
        if(_data->at(i).ts_sec >= ts_start && _data->at(i).ts_sec <= ts_end) {
            output.push_back(_data->at(i));
        }
        else if (i < _data->size() - 1 && (_data->at(i).ts_sec < ts_start && _data->at(i + 1).ts_sec > ts_start)) {
            output.push_back(interpolate_data(_data->at(i), _data->at(i + 1), ts_start));
        }
        else if(i > 0 && (_data->at(i).ts_sec > ts_end && _data->at(i - 1).ts_sec < ts_end)) {
            output.push_back(interpolate_data(_data->at(i - 1), _data->at(i), ts_end));
            break;
        }
    }

    return output;
}

void ImuManager::delete_old_measurements(const double ts) {

    while(!_data->empty()) {
        if(_data->front().ts_sec < ts) {
            _data->pop_front();
        }
    }
}
