import cv2
import numpy as np
import os
import sys
import json
from datetime import datetime

# 添加父目录到系统路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Scrap_Steel_In_Field.get_image.get_clear_picture import get_clear_image
from Scrap_Steel_In_Field.get_image.get_img import Camera
from Scrap_Steel_In_Field.config.camera_config import CAMERA_CONFIG

# 摄像头参数配置
CAMERA_PARAMS = {
    "ip": CAMERA_CONFIG["camera_ip"],
    "username": CAMERA_CONFIG["username"],
    "password": CAMERA_CONFIG["password"],
    "channel": CAMERA_CONFIG["channel"],
    "max_attempts": 1000  # 获取清晰图像的最大尝试次数
}

def detect_yellow_rectangle(image):
    """
    检测图像中的黄色矩形并返回其角点
    
    Args:
        image: 输入图像
    
    Returns:
        corners: 检测到的角点坐标列表
        mask: 黄色区域的掩码
    """
    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 定义黄色的HSV范围
    lower_yellow = np.array([15, 70, 150])
    upper_yellow = np.array([45, 255, 255])
    
    # 创建掩码
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 形态学操作，去除噪点
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, mask
    
    # 找到最大的轮廓（黄色矩形）
    max_contour = max(contours, key=cv2.contourArea)
    
    # 使用approxPolyDP简化轮廓
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    
    # 如果找到4个点，则认为是矩形
    if len(approx) == 4:
        corners = approx.reshape(-1, 2)
        return corners, mask
    
    return None, mask

def create_pixel_coordinate_system(image, grid_size=100):
    """
    在图像上创建像素坐标系
    
    Args:
        image: 输入图像
        grid_size: 网格大小（像素）
    
    Returns:
        image_with_grid: 带有坐标系的图像
    """
    # 创建图像副本
    image_with_grid = image.copy()
    height, width = image.shape[:2]
    
    # 绘制坐标轴
    # X轴（红色）
    cv2.line(image_with_grid, (0, 0), (width, 0), (0, 0, 255), 3)  # 加粗线条
    # Y轴（绿色）
    cv2.line(image_with_grid, (0, 0), (0, height), (0, 255, 0), 3)  # 加粗线条
    
    # 绘制网格线
    for x in range(0, width, grid_size):
        cv2.line(image_with_grid, (x, 0), (x, height), (128, 128, 128), 1)
        # 添加X轴刻度
        cv2.putText(image_with_grid, str(x), (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # 增大字体
    
    for y in range(0, height, grid_size):
        cv2.line(image_with_grid, (0, y), (width, y), (128, 128, 128), 1)
        # 添加Y轴刻度
        cv2.putText(image_with_grid, str(y), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # 增大字体
    
    # 添加原点标记
    cv2.putText(image_with_grid, "O(0,0)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)  # 增大字体
    
    return image_with_grid

def calculate_transform_matrix(pixel_points, world_points):
    """
    计算像素坐标系到世界坐标系的转换矩阵
    
    Args:
        pixel_points: 像素坐标系中的点坐标列表
        world_points: 世界坐标系中的点坐标列表
    
    Returns:
        transform_matrix: 转换矩阵
    """
    # 确保输入点的数量正确
    if len(pixel_points) != 4 or len(world_points) != 4:
        raise ValueError("需要4个点来计算转换矩阵")
    
    # 转换为numpy数组
    pixel_points = np.array(pixel_points, dtype=np.float32)
    world_points = np.array(world_points, dtype=np.float32)
    
    # 计算单应性矩阵
    transform_matrix, _ = cv2.findHomography(pixel_points, world_points)
    
    return transform_matrix

def save_transform_matrix(matrix, save_path):
    """
    保存转换矩阵到文件
    
    Args:
        matrix: 转换矩阵
        save_path: 保存路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 将矩阵转换为列表并保存为JSON
    matrix_list = matrix.tolist()
    with open(save_path, 'w') as f:
        json.dump(matrix_list, f)
    
    print(f"转换矩阵已保存至: {save_path}")

def resize_image(image, scale=1/3):
    """
    缩放图像
    
    Args:
        image: 输入图像
        scale: 缩放比例
    
    Returns:
        resized_image: 缩放后的图像
    """
    height, width = image.shape[:2]
    new_height = int(height * scale)
    new_width = int(width * scale)
    return cv2.resize(image, (new_width, new_height))

def transform_corners_to_world(corners):
    """
    将检测到的矩形角点转换为世界坐标
    
    Args:
        corners: 检测到的四个角点坐标，格式为 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        
    Returns:
        list: 转换后的世界坐标点列表，格式为 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        如果转换失败则返回 None
    """
    # 尝试加载最新的转换矩阵
    if not os.path.exists("output"):
        print("错误：未找到转换矩阵文件，请先运行坐标转换程序")
        return None
        
    # 获取最新的转换矩阵文件
    matrix_files = [f for f in os.listdir("output") if f.startswith("transform_matrix_")]
    if not matrix_files:
        print("错误：未找到转换矩阵文件，请先运行坐标转换程序")
        return None
        
    latest_matrix_file = sorted(matrix_files)[-1]
    matrix_path = os.path.join("output", latest_matrix_file)
    
    try:
        # 加载转换矩阵
        with open(matrix_path, 'r') as f:
            matrix_list = json.load(f)
        transform_matrix = np.array(matrix_list)
        
        # 转换所有角点
        world_corners = []
        for corner in corners:
            # 将像素坐标转换为齐次坐标
            pixel_homogeneous = np.array([corner[0], corner[1], 1])
            
            # 应用转换矩阵
            transformed_point = np.dot(transform_matrix, pixel_homogeneous)
            
            # 转换回非齐次坐标
            world_point = transformed_point[:2] / transformed_point[2]
            world_corners.append(tuple(world_point))
            
        return world_corners
        
    except Exception as e:
        print(f"转换坐标时出错: {e}")
        return None

def main():
    # 创建相机对象
    camera = Camera(CAMERA_PARAMS["ip"], CAMERA_PARAMS["username"], CAMERA_PARAMS["password"])
    
    while True:
        # 获取清晰图像
        frame = get_clear_image(camera, CAMERA_PARAMS["channel"], max_attempts=CAMERA_PARAMS["max_attempts"])
        
        
        if frame is not None:
            # 检测黄色矩形角点
            corners, mask = detect_yellow_rectangle(frame)
            
            if corners is not None:
                # 创建像素坐标系
                image_with_grid = create_pixel_coordinate_system(frame)
                
                # 绘制黄色矩形轮廓
                _, mask = detect_yellow_rectangle(frame)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    max_contour = max(contours, key=cv2.contourArea)
                    cv2.drawContours(image_with_grid, [max_contour], 0, (0, 255, 255), 3)  # 黄色轮廓，加粗显示
                
                # 在图像上标记角点
                print("\n检测到的四个角点顺序如下：")
                print("1, 2, 3, 4")
                print("\n各角点的像素坐标：")
                
                for i, (x, y) in enumerate(corners):
                    # 绘制角点
                    cv2.circle(image_with_grid, (int(x), int(y)), 8, (0, 0, 255), -1)  # 增大点的半径
                    # 添加序号和坐标标签
                    cv2.putText(image_with_grid, f"{i+1}", (int(x)-15, int(y)-15),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)  # 增大字体
                    cv2.putText(image_with_grid, f"({int(x)},{int(y)})", (int(x)+15, int(y)+15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # 增大字体
                    print(f"角点 {i+1}: ({x}, {y})")
                
                # 缩放图像
                resized_image = resize_image(image_with_grid)
                
                # 先显示图像
                cv2.imshow("检测结果", resized_image)
                print("\n请查看检测到的角点，按任意键继续...")
                cv2.waitKey(0)
                
                # 获取用户输入的世界坐标
                print("\n请输入四个角点在世界坐标系中的坐标（格式：x,y）：")
                print("请按照上述顺序（1->2->3->4）依次输入世界坐标")
                world_points = []
                for i in range(4):
                    while True:
                        try:
                            coords = input(f"请输入第{i+1}个角点的世界坐标 (x,y): ").split(',')
                            if len(coords) == 2:
                                x, y = map(float, coords)
                                world_points.append([x, y])
                                break
                            else:
                                print("请输入两个数字，用英文逗号分隔")
                        except ValueError:
                            print("输入无效，请输入两个数字，用英文逗号分隔")
                
                # 计算转换矩阵
                transform_matrix = calculate_transform_matrix(corners, world_points)
                
                # 保存转换矩阵
                current_dir = os.path.dirname(os.path.abspath(__file__))
                matrix_path = os.path.join(current_dir, "transform_matrix.json")
                save_transform_matrix(transform_matrix, matrix_path)
                
                # 验证转换矩阵
                print("\n转换矩阵验证：")
                for i, (pixel_point, world_point) in enumerate(zip(corners, world_points)):
                    # 将像素坐标转换为齐次坐标
                    pixel_homogeneous = np.array([pixel_point[0], pixel_point[1], 1])
                    # 应用转换矩阵
                    transformed_point = np.dot(transform_matrix, pixel_homogeneous)
                    # 转换回非齐次坐标
                    transformed_point = transformed_point[:2] / transformed_point[2]
                    print(f"角点 {i+1}:")
                    print(f"  原始世界坐标: ({world_point[0]:.2f}, {world_point[1]:.2f})")
                    print(f"  转换后坐标: ({transformed_point[0]:.2f}, {transformed_point[1]:.2f})")
                
                # 缩放图像
                resized_image = resize_image(image_with_grid)
                
                # 显示图像
                cv2.imshow("检测结果", resized_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                # 保存完转换矩阵后直接退出
                return
            else:
                print("未检测到黄色矩形，按ESC键退出，其他键重试...")
                key = cv2.waitKey(0)
                if key == 27:  # ESC键退出
                    break
        else:
            print("无法获取清晰图像，按ESC键退出，其他键重试...")
            key = cv2.waitKey(0)
            if key == 27:  # ESC键退出
                break

if __name__ == "__main__":
    main()
