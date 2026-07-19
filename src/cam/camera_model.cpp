#include "camera_model.h"

#include "fisheye.h"
#include "pinhole.h"

std::unique_ptr<CamModel>& CamModel::storage_()
{
    static std::unique_ptr<CamModel> instance;
    return instance;
}

CamModel& CamModel::getInstance()
{
    if (!storage_())
        throw std::runtime_error("CamModel::Init() must be called before getInstance().");
    return *storage_();
}

void CamModel::Init(const Parameter& params)
{
    // Factory: pick the concrete subclass from params.camera_model.
    if (params.camera_model == "fisheye")
        storage_() = std::make_unique<Fisheye>();
    else
        storage_() = std::make_unique<Pinhole>();
    storage_()->initImpl(params);
}

void CamModel::initCommon(const Parameter& params)
{
    camera_num_ = params.camera_num;
    image_size_ = cv::Size(params.img_width, params.img_height);
    vRic_.clear();
    vTic_.clear();
    vK_.clear();
    vD_.clear();
    for (int i = 0; i < camera_num_; i++)
    {
        vRic_.push_back(params.Ric[i]);
        vTic_.push_back(params.tic[i]);
    }
}

void CamModel::RectifyImage(const int32_t cam_id, const cv::Mat& img_raw_ptr, cv::Mat* img_rectified_ptr)
{
    // Both projection models now operate on the raw image (fisheye projects via
    // KB directly, pinhole keeps raw K/D), so rectification is a plain clone.
    (void)cam_id;
    *img_rectified_ptr = img_raw_ptr.clone();
}
