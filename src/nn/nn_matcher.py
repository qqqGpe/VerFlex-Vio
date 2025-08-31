from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
import cv2
import numpy as np
import time

# 优化SuperPoint参数以提高速度
extractor = (
    SuperPoint(
        max_num_keypoints=256,  # 从256减少到128，减少计算量
        keypoint_threshold=0.01,  # 提高阈值，减少特征点数量
        nms_radius=3,  # 减少NMS半径
        remove_borders=4,
    ).eval().cuda()
)

matcher = (
    LightGlue(features="superpoint", n_layers=3, depth_confidence=0.9, width_confidence=0.95).eval().cuda()
)

# load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
image0 = load_image("/home/gao/dataset/euroc_mav/MH_04_difficult/mav0/cam0/data/1403638189095097088.png").cuda()
image1 = load_image("/home/gao/dataset/euroc_mav/MH_04_difficult/mav0/cam0/data/1403638190845096960.png").cuda()

# 记录匹配开始时间
# 分别统计特征提取和匹配的耗时
start_time_extract = time.time()
feats0 = extractor.extract(image0)
feats1 = extractor.extract(image1)
end_time_extract = time.time()
extract_time = end_time_extract - start_time_extract

start_time_match = time.time()
matches01 = matcher({"image0": feats0, "image1": feats1})
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
matches = matches01["matches"]
end_time_match = time.time()
match_time = end_time_match - start_time_match

# 将torch tensor转为numpy
kpts0 = feats0["keypoints"].cpu().numpy()
kpts1 = feats1["keypoints"].cpu().numpy()
matches_np = matches.cpu().numpy()

# 读取原始图片（BGR格式，便于opencv显示）
img0 = cv2.imread("/home/gao/dataset/euroc_mav/MH_04_difficult/mav0/cam0/data/1403638189095097088.png")
img1 = cv2.imread("/home/gao/dataset/euroc_mav/MH_04_difficult/mav0/cam0/data/1403638190845096960.png")

# 构造DMatch对象列表
cv_matches = []
for i, (idx0, idx1) in enumerate(matches_np):
    cv_matches.append(cv2.DMatch(_queryIdx=int(idx0), _trainIdx=int(idx1), _imgIdx=0, _distance=0))

# 转为KeyPoint对象
cv_kpts0 = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in kpts0]
cv_kpts1 = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in kpts1]

# 绘制匹配结果
img_matches = cv2.drawMatches(img0, cv_kpts0, img1, cv_kpts1, cv_matches, None, flags=2)

# 显示图片
cv2.imshow("Feature Matches", img_matches)
print(f"特征提取用时: {extract_time:.4f} 秒, 匹配用时: {match_time:.4f} 秒, 匹配点数: {len(matches_np)}")
cv2.waitKey(0)
cv2.destroyAllWindows()
