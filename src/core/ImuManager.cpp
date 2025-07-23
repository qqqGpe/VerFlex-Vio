#include "ImuManager.h"
#include "eskf_solver.h"
#include "mathematical_tools.h"
#include "utils.h"

#include <glog/logging.h>
#include <opencv2/opencv.hpp>


namespace
{
constexpr int kMaxImuBufferSize = 2000;
}

bool ImuManager::FeedImuMeasurement(const ImuData& imu_measurement)
{
    if (_data->empty())
    {
        _data->push_back(imu_measurement);
    }
    else if (imu_measurement.ts_sec <= _data->back().ts_sec)
    {
        LOG(WARNING) << "latest imu data ts: " << _data->back().ts_sec << ", input imu ts: " << imu_measurement.ts_sec;
        return false;
    }
    else
    {
        _data->push_back(imu_measurement);
        if (_data->size() > kMaxImuBufferSize)
        {
            _data->pop_front();
        }
    }
    _imu_latest_timestamp = _data->back().ts_sec;
    return true;
}

ImuData ImuManager::InterpolateImuData(const ImuData& imu_1, const ImuData& imu_2, double timestamp)
{
    // time-distance lambda
    double lambda = (timestamp - imu_1.ts_sec) / (imu_2.ts_sec - imu_1.ts_sec);
    // interpolate between the two times
    ImuData data;
    data.ts_sec = timestamp;
    data.am = (1 - lambda) * imu_1.am + lambda * imu_2.am;
    data.wm = (1 - lambda) * imu_1.wm + lambda * imu_2.wm;
    return data;
}

ImuData ImuManager::GetImuData(double timestamp)
{
    auto it =
        std::lower_bound(_data->begin(), _data->end(), timestamp, [](ImuData imu_data, double timestamp) { return imu_data.ts_sec < timestamp; });
    assert(it->ts_sec >= timestamp);
    if (_data->size() > 1 && it != _data->end() - 1)
    {
        ImuData ret = InterpolateImuData(*it, *(it + 1), timestamp);
        return ret;
    }
    else
    {
        return *it;
    }
}

std::vector<ImuData> ImuManager::AccessIntervalImuMeasurements(const double ts_start, const double ts_end)
{
    std::vector<ImuData> output;
    if (ts_end <= ts_start)
    {
        LOG(ERROR) << "ts_end: " << ts_end << ", should later than ts_start: " << ts_start;
        return output;
    }

    if (_data->front().ts_sec > ts_start || _data->back().ts_sec < ts_end)
    {
        LOG(WARNING) << "ts_start: " << ts_start << ", ts_end: " << ts_end
                     << ", but imu data range is [" << _data->front().ts_sec << ", " << _data->back().ts_sec << "]";
        return output;
    }

    for (int i = 0; i < _data->size(); i++)
    {
        if (_data->at(i).ts_sec >= ts_start && _data->at(i).ts_sec <= ts_end)
        {
            output.push_back(_data->at(i));
        }
        else if (i < _data->size() - 1 && (_data->at(i).ts_sec < ts_start && _data->at(i + 1).ts_sec > ts_start))
        {
            output.push_back(InterpolateImuData(_data->at(i), _data->at(i + 1), ts_start));
        }
        else if (i > 0 && (_data->at(i).ts_sec > ts_end && _data->at(i - 1).ts_sec < ts_end))
        {
            output.push_back(InterpolateImuData(_data->at(i - 1), _data->at(i), ts_end));
            break;
        }
    }

    return output;
}

void ImuManager::ZuptUpdate(std::shared_ptr<State> state)
{
    Eigen::MatrixXd Hx;
    std::unordered_map<std::shared_ptr<Type>, size_t> _map_hx;
    std::vector<std::shared_ptr<Type>> _Hx_order;
    Eigen::VectorXd res;
    ImuData imu_data = GetImuData(state->ts_sec());
    ConstructZuptConstraint(state, imu_data, Hx, _Hx_order, _map_hx, res);
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(res.rows(), res.rows());
    solver_->update(state, Hx, res, _Hx_order, _map_hx, R);
}

void ImuManager::ConstructZuptConstraint(std::shared_ptr<State> state,
                                         ImuData imu_data,
                                         Eigen::MatrixXd& Hx,
                                         std::vector<std::shared_ptr<Type>>& _Hx_order,
                                         std::unordered_map<std::shared_ptr<Type>, size_t>& _map_hx,
                                         Eigen::VectorXd& res)
{
    bool force_pos_equal_zero = false;
    _Hx_order.push_back(state->_imu_state->q());
    _Hx_order.push_back(state->_imu_state->bg());
    _Hx_order.push_back(state->_imu_state->v());
    if (force_pos_equal_zero)
    {
        _Hx_order.push_back(state->_imu_state->p());
    }

    int total_hx = 0;
    _map_hx.clear();
    // insert R_ItoG
    _map_hx.insert({state->_imu_state->q(), total_hx});
    total_hx += state->_imu_state->q()->size();

    // insert bg
    _map_hx.insert({state->_imu_state->bg(), total_hx});
    total_hx += state->_imu_state->bg()->size();

    // insert v
    _map_hx.insert({state->_imu_state->v(), total_hx});
    total_hx += state->_imu_state->v()->size();

    if (force_pos_equal_zero)
    {
        _map_hx.insert({state->_imu_state->p(), total_hx});
        total_hx += state->_imu_state->p()->size();
    }

    Hx = Eigen::MatrixXd::Zero(9, total_hx);
    res = Eigen::VectorXd::Zero(9);

    Eigen::Vector3d bg = state->_imu_state->bg()->vec();
    Eigen::Vector3d p = state->_imu_state->p()->vec();
    Eigen::Vector3d v = state->_imu_state->v()->vec();
    Eigen::Matrix3d R_ItoG = state->_imu_state->q()->Rot();

    res.segment(0, 3) = -(imu_data.am - R_ItoG.transpose() * _gravity_magn);
    res.segment(3, 3) = -(imu_data.wm - bg);
    res.segment(6, 3) = -v;
    if (force_pos_equal_zero)
    {
        res.segment(9, 3) = -p;
    }

    // jacobian for R_ItoG
    Hx.block(0, _map_hx[state->_imu_state->q()], 3, 3) = -MathUtils::skew(R_ItoG.transpose() * _gravity_magn);

    // jacobian for bg
    Hx.block(3, _map_hx[state->_imu_state->bg()], 3, 3) = -Eigen::Matrix3d::Identity();

    // jacobian for v
    Hx.block(6, _map_hx[state->_imu_state->v()], 3, 3) = Eigen::Matrix3d::Identity();

    // jacobian for p
    if (force_pos_equal_zero)
    {
        Hx.block(9, _map_hx[state->_imu_state->p()], 3, 3) = Eigen::Matrix3d::Identity();
    }

    Eigen::MatrixXd weight = Eigen::MatrixXd::Identity(Hx.rows(), Hx.rows());
    Hx = weight * Hx;
    res = weight * res;
}

void ImuManager::ClearExpiredMeasurements(const double ts)
{
    while (!_data->empty())
    {
        if (_data->front().ts_sec < ts)
        {
            _data->pop_front();
        }
    }
}

void ImuManager::SetImuNoise(const double sigma_na, const double sigma_nw, const double sigma_ba, const double sigma_bg)
{
    _sigma_na = sigma_na;
    _sigma_nw = sigma_nw;
    _sigma_ba = sigma_ba;
    _sigma_bg = sigma_bg;
    LOG(INFO) << "Imu noise set: na: " << _sigma_na << ", nw: " << _sigma_nw << ", ba: " << _sigma_ba << ", bg: " << _sigma_bg;
}

bool ImuManager::IsStaticStatus()
{
    constexpr uint32_t kImuFreq = 200; // 200Hz
    constexpr int kImuLenToCalc = 1.0 * kImuFreq; // 1s
    constexpr int kIsStaticCntThres = 50;

    if (_data->size() < kImuLenToCalc)
    {
        return false;
    }

    std::deque<ImuData> data_buffer(_data->end() - kImuLenToCalc, _data->end());

    Eigen::Vector3d acc_mean = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro_mean = Eigen::Vector3d::Zero();
    for (auto it = data_buffer.begin(); it != data_buffer.end(); it++)
    {
        acc_mean += it->am;
        gyro_mean += it->wm;
    }

    acc_mean = acc_mean / data_buffer.size();
    gyro_mean = gyro_mean / data_buffer.size();

    double acc_var = 0.0;
    for (auto it = data_buffer.begin(); it != data_buffer.end(); it++)
    {
        acc_var += (it->am - acc_mean).dot(it->am - acc_mean);
    }
    acc_var = acc_var / data_buffer.size();  // 加计的方差，若方差小于阈值则认为系统处于静止状态

    static int is_static_cnt = 0;
    if (acc_var < _imu_acc_var_static_thres && gyro_mean.norm() < _imu_gyro_static_thres)
    {
        if (is_static_cnt < kIsStaticCntThres)
        {
            is_static_cnt++;
        }
    }
    else
    {
        if (is_static_cnt > 0)
        {
            is_static_cnt--;
        }
    }

    if (is_static_cnt == kIsStaticCntThres)
    {
        return true;
    }
    else
    {
        return false;
    }
}
