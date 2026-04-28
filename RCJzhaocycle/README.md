# RCJzhaocycle

用于从图片或摄像头画面中提取黑色、不完整、有一定宽度的圆弧。实时运行主路径是 C++/OpenCV 检测库；Python 脚本保留为离线参考。

## C++ 实时检测库

树莓派依赖：

```bash
sudo apt install cmake g++ libopencv-dev
```

构建：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

离线 benchmark：

```bash
./build/arc_benchmark --input pic --repeat 3
```

输出字段：

```text
image,width,height,found,center_x,center_y,radius,angle_start,angle_end,confidence,avg_ms
```

`found=1` 表示本帧检测到圆弧。坐标和半径都映射回原始输入图像坐标。

### C++ 接口

```cpp
#include "arc_detector.hpp"

rcj::ArcDetectorConfig config;
config.return_binary_roi = false;

rcj::ArcDetector detector(config);
rcj::ArcDetection result = detector.detect(frame);

if (result.found) {
    // result.center, result.radius, result.angle_start/end, result.confidence
}
```

默认处理流程会把 `800x600` 输入缩放到 `400x300` 做检测，并在检测成功后输出圆弧几何信息。调试时可以设置 `return_binary_roi=true`，返回目标附近的二值 ROI；实时运行建议关闭。

## Web 实时预览和参数调节

从本机通过 SSH 启动树莓派服务：

```bash
./scripts/start_remote_web_tuner.sh rcj@10.32.0.172
```

默认会同步当前工程到树莓派 `/home/rcj/RCJzhaocycle`，编译 `arc_web_tuner`，并在后台启动：

```text
http://10.32.0.172:8080/
```

网页中可以查看摄像头实时画面、目标圆弧叠加、二值 ROI，并用滑条调节检测参数。每次滑条变化都会自动保存参数：

```text
params/latest_params.json
params/arc_params_YYYYMMDD_HHMM.json
params/arc_params_history.csv
```

文件名中的时间戳精确到分钟。调参结束后，把树莓派上的参数文档拉回本机：

```bash
./scripts/fetch_remote_params.sh rcj@10.32.0.172
```

默认保存到本机：

```text
remote_params/
```

如果需要换端口或摄像头编号：

```bash
PORT=8090 CAMERA=0 ./scripts/start_remote_web_tuner.sh rcj@10.32.0.172
```

停止远端服务：

```bash
ssh rcj@10.32.0.172 'pkill -x arc_web_tuner'
```

## Python 离线参考

### 安装依赖

```bash
pip install -r requirements.txt
```

当前离线图片处理只需要 `numpy`、`Pillow`、`scipy`。如果要使用摄像头模式，需要安装 `opencv-python`。

### 处理 pic 文件夹中的测试图

```bash
python3 arc_binary.py -i pic -o out
```

输出文件会写入 `out` 目录，例如：

```text
out/1_binary.png
out/2_binary.png
out/3_binary.png
```

默认输出为黑底白色圆弧。如果需要白底黑色圆弧：

```bash
python3 arc_binary.py -i pic -o out --invert
```

### 摄像头模式

```bash
python3 arc_binary.py --camera 0 -o out
```

摄像头窗口中：

- 按 `s` 保存当前帧的二值化结果
- 按 `q` 退出

### 常用调参

- `--min-contrast`：提高该值会减少较浅的黑色干扰，降低该值会保留更多暗区域。
- `--bg-sigma`：背景估计的模糊尺度。光照不均或背景渐变明显时可调大。
- `--min-radius-ratio` / `--max-radius-ratio`：限制待识别圆弧半径范围，已知目标大小时可收窄以减少误检。
- `--iterations`：圆模型搜索次数。干扰较多时可增大，但会变慢。
