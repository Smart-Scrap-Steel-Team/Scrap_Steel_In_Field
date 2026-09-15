
# 测试套件 - 独立模块测试

所有模块都已拆分成独立的可运行测试脚本，每个都有可视化效果！

## 快速启动

### 方式1: 使用主菜单（推荐）
```bash
cd test_suite
python main.py
```

### 方式2: 直接运行单个脚本
```bash
cd test_suite
python test_1_camera.py        # 摄像头测试
python test_2_detection.py     # 矩形检测测试
python test_3_coordinate.py    # 坐标转换测试
python test_4_arm.py           # 机械臂测试
python test_5_full.py          # 完整工作流
```

## 测试模块说明

### 1. 摄像头测试 (test_1_camera.py)
- 实时显示摄像头画面
- 按空格拍照保存
- 按q退出

### 2. 矩形检测测试 (test_2_detection.py)
- 实时显示：原图 + HSV掩码 + 检测结果
- 检测黄色矩形并标记角点
- 按空格保存检测结果

### 3. 坐标转换测试 (test_3_coordinate.py)
- 实时显示：检测结果 + 坐标信息面板
- 像素坐标 → 世界坐标转换
- 计算3个抓取点（25%/50%/75%）
- 按空格保存结果

### 4. 机械臂测试 (test_4_arm.py)
两种模式：
- 服务器模式：等待真实机械臂连接
- 模拟器模式：模拟机械臂连接和通讯

### 5. 完整工作流 (test_5_full.py)
- 完整流程：等待yes_grab → 拍照 → 检测 → 坐标转换 → 发送抓取点
- 保存所有中间结果

## 文件结构

```
test_suite/
├── main.py                    # 主菜单
├── config.py                  # 配置文件
├── camera.py                  # 摄像头模块
├── vision.py                  # 视觉处理模块
├── arm_control.py             # 机械臂控制模块
├── transform_matrix.json      # 坐标转换矩阵
│
├── test_1_camera.py           # ← 独立测试1
├── test_2_detection.py        # ← 独立测试2
├── test_3_coordinate.py       # ← 独立测试3
├── test_4_arm.py              # ← 独立测试4
├── test_5_full.py             # ← 独立测试5
│
├── README.md                  # 本文档
└── 启动测试.bat               # Windows快捷启动
```

## 配置说明

修改 `config.py` 来调整参数：
```python
CAMERA_CONFIG = {
    "camera_ip": "192.168.3.10",  # 摄像头IP
    "channel": 1,                  # 通道号
}

RECTANGLE_DETECT_PARAMS = {
    "hsv_lower": [15, 70, 150],    # 黄色下限
    "hsv_upper": [45, 255, 255],  # 黄色上限
}
```

## 依赖库

需要安装以下库：
```bash
pip install opencv-python numpy requests
```

