import os
import sys
# 添加父目录到系统路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 相机配置
CAMERA_CONFIG = {
    "camera_ip": "192.168.3.10",
    "username": "admin",
    "password": "",  # 根据RTSP地址，密码为空
    "channel": 1, # 废钢摄像头
    "channels": [1,2,3,4,5,6,7,8,9], # 所有摄像头
    # RTSP URL格式: rtsp://{username}:{password}@{ip}:554/live/ch00_0
    "rtsp_path": "/live/ch00_0"  # 自定义RTSP路径
}

# # 各通道对应的功能描述
# 1: "四号相机（出场地磅）",
# 3: "三号相机（出场闸机）", 
# 4: "七号摄像头（场外堆料区）", 
# 5: "二号摄像头（安全生产）", 
# 6: "一号摄像头（火焰烟雾监测）", 
# 7: "五号摄像头（进场地磅）", 
# 8: "六号摄像头（进场车牌识别）", 
# 9: "九号摄像头（场内堆料区）"