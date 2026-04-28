# C++ OpenCV 实时圆弧检测方案

## Summary
新增一套面向树莓派的 C++/OpenCV 检测库，替代当前 Python RANSAC 实现。目标输入为 `800x600@60`，主输出为圆弧几何信息和可选二值 ROI；默认不显示、不保存调试图，单帧未可靠检测时立即返回 `found=false`。

## Key Changes
- 新增 C++ 类库接口：
  - `ArcDetector::detect(const cv::Mat& frame) -> ArcDetection`
  - `ArcDetection` 包含 `found`、`center`、`radius`、`angle_start/end`、`confidence`、`roi_rect`、可选 `binary_roi`
  - 输入支持 BGR 或灰度 `cv::Mat`，输出坐标统一映射回原始 800x600 图像坐标
- 新增 CMake 构建，依赖系统 OpenCV：
  - 使用 `find_package(OpenCV REQUIRED)`
  - 默认构建静态/动态库
  - 额外提供离线 benchmark 小工具处理 `pic/*.png`，只用于验证速度和识别效果
- 保留当前 Python 脚本作为离线参考，不作为实时路径。

## Detection Pipeline
- 每帧先缩放到 `400x300` 处理，结果再按比例映射回原图，降低计算量。
- 使用 HSV/灰度组合生成候选区域：
  - 先估计绿色场地区域并膨胀，排除黑墙、机器人结构等非场地干扰。
  - 在场地区域内提取暗色候选，得到黑色圆弧二值 mask。
  - 用小核 morphology 去掉散点和断裂噪声。
- 用 OpenCV `HoughCircles` 在低分辨率 mask/灰度图上生成少量圆候选，半径范围按 800x600 下目标大小约束。
- 对候选圆做快速评分，选择最可信目标：
  - 圆环带内黑色支持点数量足够。
  - 角度覆盖达到最小圆弧长度。
  - 圆内部黑色密度较低，避免选中机器人结构或实心干扰。
  - 排除过长直线、边界墙线等非圆弧候选。
- 如果上一帧检测成功，仅用上一帧圆心/半径附近的 ROI 优先搜索；失败时仍返回 `found=false`，不沿用上一帧结果。

## Performance Targets
- 树莓派目标：`800x600` 输入下平均处理时间 `<= 16.7ms/frame`，即约 60 FPS。
- 默认关闭显示和保存；调试图只通过配置开关生成 `binary_roi`，避免影响帧率。
- 如果 full-frame Hough 在目标板上超时，默认降级策略为：
  - 继续保持输入 800x600；
  - 处理尺度从 `0.5` 改为 `0.4`；
  - 限制 Hough 最大候选数和半径范围；
  - 不改变公开接口。

## Test Plan
- 离线准确性：用 `pic/1.png` 到 `pic/10.png` 跑 benchmark，要求每张图都返回 `found=true`，圆弧位置与当前 `contact_sheet` 的目标圆弧一致。
- 离线性能：benchmark 输出每张图耗时、平均耗时、最大耗时。
- 树莓派实测：接入摄像头帧后测 300 帧平均耗时和最差耗时，验收标准为平均 `<=16.7ms`。
- 失败场景：无圆弧、圆弧被遮挡、画面中黑色干扰增多时，应返回低置信度或 `found=false`，不能输出明显错误圆弧。

## Assumptions
- 部署平台有可用的 C++ OpenCV，建议 OpenCV 4.x。
- 摄像头能稳定提供 `800x600@60`；算法只保证处理端尽量达到 60 FPS。
- 圆弧目标位于绿色场地区域内，黑色墙体、机器人结构和线缆属于应抑制的干扰。
- 实时控制主要使用圆弧几何信息，二值 ROI 只用于调试或可选下游处理。
