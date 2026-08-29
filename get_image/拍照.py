from get_img import Camera
import cv2
import time
import sys
import os
from datetime import datetime

# 添加父目录到系统路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Scrap_Steel_In_Field.get_image.get_clear_picture import get_clear_image
from Scrap_Steel_In_Field.config.camera_config import CAMERA_CONFIG

def take_photo():
    # 创建Camera对象，使用配置文件中的参数
    camera = Camera(
        nvr_ip=CAMERA_CONFIG["camera_ip"],
        nvr_username=CAMERA_CONFIG["username"],
        nvr_password=CAMERA_CONFIG["password"]
    )
    
    # 获取清晰图像
    frame = get_clear_image(
        camera=camera,
        channel=CAMERA_CONFIG["channel"],
        max_attempts=30,
        wait_time=0.5,
        timeout=30
    )
    
    if frame is not None:
        # 生成文件名（使用时间戳）
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"photo_{timestamp}.jpg"
        
        # 保存图片
        cv2.imwrite(filename, frame)
        print(f"照片已保存为: {filename}")
    else:
        print("无法获取图像")

if __name__ == "__main__":
    take_photo()
