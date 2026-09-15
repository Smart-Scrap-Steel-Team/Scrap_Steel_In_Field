import cv2
import numpy as np
import os
import sys
from datetime import datetime
import time
# 添加父目录到系统路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Scrap_Steel_In_Field.config.detection_config import LOG_CONFIG
from Scrap_Steel_In_Field.config.camera_config import CAMERA_CONFIG
from Scrap_Steel_In_Field.utils.logger import setup_logger
from Scrap_Steel_In_Field.get_image.get_img import Camera

# 创建日志记录器
logger = setup_logger('image', LOG_CONFIG['base_dir'], LOG_CONFIG['sub_dirs']['image'])

def check_image_quality(image, methods=None):
    """检查图像质量，判断是否模糊"""
    if methods is None:
        methods = ['laplacian', 'sobel', 'canny_count', 'fft']
    
    if image is None:
        return {}, True, "图像为空"
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    scores = {}
    is_blurry = False
    feedback = []
    
    if 'laplacian' in methods:
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_variance = laplacian.var()
        scores['laplacian'] = lap_variance
        lap_threshold = 50
        if lap_variance < lap_threshold:
            is_blurry = True
            feedback.append(f"拉普拉斯变异度低 ({lap_variance:.2f} < {lap_threshold})")
        else:
            feedback.append(f"拉普拉斯变异度正常 ({lap_variance:.2f})")
    
    if 'sobel' in methods:
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        sobel_mean = np.mean(sobel_magnitude)
        scores['sobel'] = sobel_mean
        sobel_threshold = 20
        if sobel_mean < sobel_threshold:
            is_blurry = True
            feedback.append(f"Sobel梯度均值低 ({sobel_mean:.2f} < {sobel_threshold})")
        else:
            feedback.append(f"Sobel梯度均值正常 ({sobel_mean:.2f})")
    
    if 'canny_count' in methods:
        edges = cv2.Canny(gray, 100, 200)
        edge_count = np.count_nonzero(edges)
        edge_percentage = edge_count / (gray.shape[0] * gray.shape[1]) * 100
        scores['canny_count'] = edge_percentage
        canny_threshold = 0.5
        if edge_percentage < canny_threshold:
            is_blurry = True
            feedback.append(f"边缘检测点少 ({edge_percentage:.2f}% < {canny_threshold}%)")
        else:
            feedback.append(f"边缘检测点正常 ({edge_percentage:.2f}%)")
    
    if 'fft' in methods:
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.uint8)
        center_size = min(20, min(rows, cols) // 4)
        mask[crow-center_size:crow+center_size, ccol-center_size:ccol+center_size] = 0
        
        high_freq_energy = np.sum(magnitude_spectrum * mask) / np.sum(mask)
        scores['fft'] = high_freq_energy
        
        fft_threshold = 20
        if high_freq_energy < fft_threshold:
            is_blurry = True
            feedback.append(f"高频能量低 ({high_freq_energy:.2f} < {fft_threshold})")
        else:
            feedback.append(f"高频能量正常 ({high_freq_energy:.2f})")
    
    quality_info = "图像质量: " + ("模糊" if is_blurry else "清晰") + "\n"
    quality_info += " | ".join(feedback)
    
    return scores, is_blurry, quality_info

def get_clear_image(camera, channel, max_attempts=30, wait_time=0.5, timeout=30, save_config=None):
    """获取清晰的图像，多次尝试直到获取到清晰的图像
    
    Args:
        camera: 相机对象
        channel: 相机通道号
        max_attempts: 最大尝试次数
        wait_time: 每次尝试之间的等待时间
        timeout: 超时时间
        save_config: 保存配置，包含base_dir和sub_dirs
        
    Returns:
        numpy.ndarray: 清晰的图像，如果获取失败则返回None
    """
    start_time = time.time()
    attempt_count = 0
    frame = None
    clear_frame = None
    last_frame = None  # 保存最后一帧图像
    
    while attempt_count < max_attempts:
        if (time.time() - start_time) > timeout:
            logger.error(f"获取清晰图像超时 ({timeout}秒)")
            if last_frame is not None:
                logger.warning("返回超时前的最后一帧图像")
                return last_frame
            return None
            
        attempt_count += 1
        logger.info(f"正在获取图像，尝试 {attempt_count}/{max_attempts} (已耗时: {time.time() - start_time:.1f}秒)")
        
        try:
            frame = camera.get_camera_image_by_rtspurl(channel)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "output_images"
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"frame_{timestamp}.jpg")
            cv2.imwrite(save_path, frame)

            if frame is not None:
                last_frame = frame  # 更新最后一帧图像
        except Exception as e:
            logger.error(f"获取图像失败: {e}")
            time.sleep(wait_time)
            continue
        
        if frame is None:
            logger.error("无法获取相机图像")
            time.sleep(wait_time)
            continue
        
        scores, is_blurry, quality_info = check_image_quality(frame)
        logger.info(quality_info.replace("\n", " "))
        
        if is_blurry:
            logger.warning(f"图像模糊，尝试重新获取 (第{attempt_count}次尝试)")
            time.sleep(wait_time)
            continue
        else:
            logger.info("图像清晰度良好，可以进行检测")
            clear_frame = frame
            break
    
    if clear_frame is None:
        logger.warning(f"在{attempt_count}次尝试后未能获取清晰图像，使用最后一次获取的图像")
        clear_frame = last_frame
    
    return clear_frame

if __name__ == "__main__":
    # 创建相机对象
    camera = Camera(CAMERA_CONFIG["camera_ip"], CAMERA_CONFIG["username"], CAMERA_CONFIG["password"])
    # 调用get_clear_image函数
    frame = get_clear_image(camera, CAMERA_CONFIG["channel"], max_attempts=30, wait_time=0.5, timeout=30, save_config=None)
    if frame is not None:
        print("成功获取清晰图像")
        # 显示图像
        cv2.imshow("清晰图像", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("无法获取清晰图像")