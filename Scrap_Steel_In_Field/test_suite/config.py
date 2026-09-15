
# 测试套件配置
# 摄像头配置
CAMERA_CONFIG = {
    "camera_ip": "192.168.3.10",
    "username": "admin",
    "password": "",
    "channel": 1,
    "rtsp_path": "/live/ch00_0"
}

# 黄色矩形检测参数
RECTANGLE_DETECT_PARAMS = {
    "hsv_lower": [15, 70, 150],
    "hsv_upper": [45, 255, 255],
    "min_area": 2000
}

# 机械臂配置
ARM_CONFIG = {
    "server_ip": "0.0.0.0",
    "server_port": 8770
}

# 保存路径配置
SAVE_CONFIG = {
    "base_dir": "photos",
    "sub_dirs": {
        "original": "original",
        "result": "result"
    }
}

