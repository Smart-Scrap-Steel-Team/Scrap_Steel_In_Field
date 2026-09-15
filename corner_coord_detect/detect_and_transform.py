import cv2
import numpy as np
import os
import sys
import time
import json
from datetime import datetime

# 添加父目录到系统路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from get_image.get_clear_picture import get_clear_image
from get_image.get_img import Camera
from config.camera_config import CAMERA_CONFIG

# 摄像头参数配置
CAMERA_PARAMS = {
    "ip": CAMERA_CONFIG["camera_ip"],
    "username": CAMERA_CONFIG["username"],
    "password": CAMERA_CONFIG["password"],
    "channel": CAMERA_CONFIG["channel"],
    "max_attempts": 1000  # 获取清晰图像的最大尝试次数
}

# 黄色矩形检测参数
RECTANGLE_DETECT_PARAMS = {
    'hsv_lower': np.array([15, 70, 150]),  # HSV颜色空间下限
    'hsv_upper': np.array([45, 255, 255]),  # HSV颜色空间上限
    'min_area': 2000  # 最小矩形面积
}

def transform_corners_to_world(corners):
    """
    将检测到的矩形角点转换为世界坐标
    
    Args:
        corners: 检测到的四个角点坐标，格式为 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        
    Returns:
        list: 转换后的世界坐标点列表，格式为 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        如果转换失败则返回 None
    """
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    matrix_path = os.path.join(current_dir, "transform_matrix.json")
    
    # 检查转换矩阵文件是否存在
    if not os.path.exists(matrix_path):
        print("错误：未找到转换矩阵文件，请先运行坐标转换程序")
        return None
    
    try:
        # 加载转换矩阵
        with open(matrix_path, 'r') as f:
            matrix_list = json.load(f)
        transform_matrix = np.array(matrix_list)
        
        # 打印转换矩阵用于调试
        print("\n转换矩阵:")
        print(transform_matrix)
        
        # 转换所有角点
        world_corners = []
        print("\n像素坐标 -> 世界坐标 (转换过程):")
        for i, corner in enumerate(corners):
            # 将像素坐标转换为齐次坐标
            pixel_homogeneous = np.array([corner[0], corner[1], 1])
            
            # 应用转换矩阵
            transformed_point = np.dot(transform_matrix, pixel_homogeneous)
            
            # 转换回非齐次坐标
            world_point = transformed_point[:2] / transformed_point[2]
            world_corners.append(tuple(world_point))
            
            # 打印每个点的转换信息
            print(f"角点 {i+1}: 像素({corner[0]:.1f}, {corner[1]:.1f}) -> 齐次({pixel_homogeneous}) -> 变换({transformed_point}) -> 世界({world_point[0]:.1f}, {world_point[1]:.1f})")
            
        return world_corners
        
    except Exception as e:
        print(f"转换坐标时出错: {e}")
        return None

def detect_yellow_rectangle(image, params=None):
    """
    检测图像中的黄色矩形并返回其角点
    
    Args:
        image: 输入图像
        params: 检测参数，包含HSV阈值等
    
    Returns:
        corners: 检测到的角点坐标列表
        mask: 黄色区域的掩码
    """
    if params is None:
        params = RECTANGLE_DETECT_PARAMS
    
    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 创建掩码
    mask = cv2.inRange(hsv, params['hsv_lower'], params['hsv_upper'])
    
    # 形态学操作，去除噪点
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, mask
    
    # 找到最大的轮廓（假设是黄色矩形）
    max_contour = max(contours, key=cv2.contourArea)
    
    # 检查面积是否满足最小要求
    if cv2.contourArea(max_contour) < params['min_area']:
        return None, mask
    
    # 使用approxPolyDP简化轮廓
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    
    # 如果找到4个点，则认为是矩形
    if len(approx) == 4:
        corners = approx.reshape(-1, 2)
        return corners, mask
    
    return None, mask

def visualize_detection(image, corners, world_corners=None):
    """
    在图像上可视化检测结果
    
    Args:
        image: 输入图像
        corners: 检测到的角点坐标
        world_corners: 转换后的世界坐标（可选）
    
    Returns:
        image_with_marks: 标记后的图像
    """
    image_with_marks = image.copy()
    
    # 绘制角点和标签
    corner_names = ["左上", "右上", "右下", "左下"]
    for i, (x, y) in enumerate(corners):
        # 绘制角点
        cv2.circle(image_with_marks, (int(x), int(y)), 5, (0, 0, 255), -1)
        
        # 添加序号和坐标标签
        cv2.putText(image_with_marks, f"{i+1}", (int(x)-10, int(y)-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 如果有世界坐标，也显示出来
        if world_corners is not None:
            world_x, world_y = world_corners[i]
            coord_text = f"({world_x:.1f}, {world_y:.1f})"
            cv2.putText(image_with_marks, coord_text, (int(x)+10, int(y)+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return image_with_marks

def detect_and_transform_rectangle(show_result=False, max_attempts=5, image_path=None):
    """
    检测黄色矩形并转换坐标的主函数
    
    Args:
        show_result: 是否显示检测结果
        max_attempts: 最大尝试次数
    
    Returns:
        tuple: (corners, world_corners)
            - corners: 检测到的角点坐标
            - world_corners: 转换后的世界坐标
    """
    # 创建相机对象
    camera = Camera(CAMERA_PARAMS["ip"], 
                   CAMERA_PARAMS["username"], 
                   CAMERA_PARAMS["password"])
    
    # 使用while循环尝试直到成功
    attempt = 0
    success = False
    
    while not success and attempt < max_attempts:
        attempt += 1
        print(f"尝试 {attempt}/{max_attempts}")
        
        # 获取清晰图像
        frame = get_clear_image(camera, CAMERA_PARAMS["channel"], max_attempts=CAMERA_PARAMS["max_attempts"])
        if frame is None:
            # print("无法获取清晰图像")
            # continue
            frame = cv2.imread(image_path)
        
        # 检测黄色矩形
        corners, mask = detect_yellow_rectangle(frame)
        if corners is None:
            print("未检测到黄色矩形")
            continue
        
        # 转换到世界坐标
        world_corners = transform_corners_to_world(corners)
        if world_corners is None:
            print("坐标转换失败")
            continue
        
        # 如果执行到这里，说明成功获取了图像并检测到了矩形
        success = True
        print(f"成功获取图像并检测到矩形（尝试 {attempt}/{max_attempts}）")
    
    if not success:
        print(f"经过 {max_attempts} 次尝试后仍未成功获取有效图像和矩形")
        return None, None
        
    # 显示结果
    if show_result:
        # 可视化检测结果
        result_image = visualize_detection(frame, corners, world_corners)
        
        # 缩放图像以便显示
        scale_percent = 30
        width = int(result_image.shape[1] * scale_percent / 100)
        height = int(result_image.shape[0] * scale_percent / 100)
        resized_image = cv2.resize(result_image, (width, height))
        
        # 显示图像
        cv2.imshow("检测结果", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # # 打印坐标信息
    # print("\n检测结果：")
    # print("像素坐标 -> 世界坐标")
    # for i, (corner, world_corner) in enumerate(zip(corners, world_corners)):
    #     print(f"角点 {i+1}: ({corner[0]:.1f}, {corner[1]:.1f}) -> ({world_corner[0]:.1f}, {world_corner[1]:.1f})")
    
    return corners, world_corners

def calculate_grasp_points(world_corners):
    """
    根据矩形的四个角点计算三个抓取点的坐标
    
    Args:
        world_corners: 矩形的四个角点坐标，格式为 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        
    Returns:
        list: 三个抓取点的坐标，格式为 [(x1,y1), (x2,y2), (x3,y3)]
    """
    if len(world_corners) != 4:
        return None
    
    try:
        # 将角点转换为numpy数组
        corners = np.array(world_corners)
        
        # 计算矩形的中心点
        center = np.mean(corners, axis=0)
        
        # 计算主方向（使用PCA方法）
        centered_points = corners - center
        cov_matrix = np.dot(centered_points.T, centered_points)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 获取主方向向量（对应最大特征值的特征向量）
        main_direction = eigenvectors[:, np.argmax(eigenvalues)]
        # 获取次方向向量（对应次大特征值的特征向量）
        secondary_direction = eigenvectors[:, np.argmin(eigenvalues)]
        
        # 计算所有点到主方向和次方向的距离
        main_distances = np.abs(np.dot(centered_points, main_direction))
        secondary_distances = np.abs(np.dot(centered_points, secondary_direction))
        
        # 计算矩形的长度和宽度
        length = 2 * np.max(main_distances)
        width = 2 * np.max(secondary_distances)
        
        # 计算三个抓取点
        grasp_points = []
        for ratio in [0.25, 0.5, 0.75]:
            # 计算在长边方向上的偏移
            length_offset = (ratio - 0.5) * length * main_direction
            # 计算在宽度方向上的偏移（保持在中心）
            width_offset = 0 * width * secondary_direction  # 宽度方向保持在中心
            # 计算抓取点坐标
            grasp_point = center + length_offset + width_offset
            grasp_points.append(tuple(grasp_point))
        
        return grasp_points
        
    except Exception as e:
        print(f"计算抓取点时出错: {e}")
        return None

if __name__ == "__main__":
    # 运行检测和转换
    corners, world_corners = detect_and_transform_rectangle()
    
    if world_corners is not None:
        # 计算抓取点
        grasp_points = calculate_grasp_points(world_corners)
        # print(grasp_points)
        if grasp_points is not None:
            # print("\n抓取点坐标：")
            for i, point in enumerate(grasp_points):
                print(f"{point[0]:.1f},{point[1]:.1f}") 