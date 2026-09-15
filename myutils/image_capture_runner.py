import time
import os
from datetime import datetime
import cv2
from get_image.get_clear_picture import get_clear_image
from get_image.get_img import Camera
from config.camera_config import CAMERA_CONFIG
from config.detection_config import SAVE_CONFIG

# 可选：引入日志模块
try:
    from myutils.logger import setup_logger
    logger = setup_logger('image_capture', SAVE_CONFIG['base_dir'], SAVE_CONFIG['sub_dirs'].get('log', 'log'))
except Exception:
    logger = None

def start_image_capture(photo_type="original", max_retries=3, retry_delay=1):
    """图像采集模块主循环，可独立运行，支持保存不同类型图片和重试机制"""
    camera = Camera(CAMERA_CONFIG["camera_ip"], CAMERA_CONFIG["username"], CAMERA_CONFIG["password"])
    while True:
        try:
            if logger:
                logger.info("开始获取清晰图像...")
            # 获取清晰图像
            frame = get_clear_image(
                camera,
                CAMERA_CONFIG["channel"],
                max_attempts=max_retries,
                wait_time=retry_delay,
                timeout=max_retries * retry_delay * 2,
                save_config=SAVE_CONFIG
            )
            if frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                sub_dir = SAVE_CONFIG["sub_dirs"].get(photo_type, "original")
                save_dir = os.path.join(SAVE_CONFIG["base_dir"], sub_dir)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{photo_type}_image_{timestamp}.jpg")
                cv2.imwrite(save_path, frame)
                if logger:
                    logger.info(f"已保存{photo_type}图像: {save_path}")
                else:
                    print(f"[INFO] Saved {photo_type} image: {save_path}")
            else:
                if logger:
                    logger.error("未能获取清晰图像")
                else:
                    print("[ERROR] Failed to get clear image.")
        except Exception as e:
            if logger:
                logger.error(f"采集图像过程中发生错误: {str(e)}")
            else:
                print(f"[ERROR] Exception during image capture: {str(e)}")
        time.sleep(5) 