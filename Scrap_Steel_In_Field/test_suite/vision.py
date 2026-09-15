import cv2
import numpy as np
import json
import os

# 黄色矩形检测参数
RECTANGLE_DETECT_PARAMS = {
    'hsv_lower': np.array([15, 70, 150]),
    'hsv_upper': np.array([45, 255, 255]),
    'min_area': 2000
}


def detect_yellow_rectangle(image, params=None):
    if params is None:
        params = RECTANGLE_DETECT_PARAMS

    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 创建掩码
    mask = cv2.inRange(hsv, params['hsv_lower'], params['hsv_upper'])

    # 形态学操作，去除噪点
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations=1)
    # mask = cv2.dilate(mask, kernel, iterations=5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,iterations=1)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, mask

    # 找到最大的轮廓（假设是黄色矩形）
    max_contour = max(contours, key=cv2.contourArea)

    # 检查面积是否满足最小要求
    if cv2.contourArea(max_contour) < params['min_area']:
        return None, mask

    # 方法1：使用approxPolyDP简化轮廓
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)

    # 如果找到4个点，则认为是矩形
    if len(approx) == 4:
        corners = approx.reshape(-1, 2)
        # 排序角点
        corners = _sort_corners(corners)
        return corners, mask
    else:
        return None,mask


def _sort_corners(box):
    """按左上、右上、右下、左下排序"""
    # 按y坐标排序，分为上下两组
    sorted_by_y = box[box[:, 1].argsort()]
    top_two = sorted_by_y[:2]
    bottom_two = sorted_by_y[2:]

    # 每组按x坐标排序
    top_left = top_two[top_two[:, 0].argmin()]
    top_right = top_two[top_two[:, 0].argmax()]
    bottom_left = bottom_two[bottom_two[:, 0].argmin()]
    bottom_right = bottom_two[bottom_two[:, 0].argmax()]

    return np.array([top_left, top_right, bottom_right, bottom_left])


def visualize_detection(image, corners, world_corners=None):
    image_with_marks = image.copy()

    for i, (x, y) in enumerate(corners):
        cv2.circle(image_with_marks, (int(x), int(y)), 5, (0, 0, 255), -1)

        cv2.putText(image_with_marks, str(i + 1), (int(x) - 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if world_corners is not None:
            world_x, world_y = world_corners[i]
            coord_text = "(" + str(round(world_x, 1)) + "," + str(round(world_y, 1)) + ")"
            cv2.putText(image_with_marks, coord_text, (int(x) + 10, int(y) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image_with_marks


def transform_corners_to_world(corners, matrix_file="transform_matrix.json"):
    if not os.path.exists(matrix_file):
        print("Error: transform matrix file not found")
        return None

    try:
        with open(matrix_file, 'r') as f:
            matrix_list = json.load(f)
        transform_matrix = np.array(matrix_list)

        world_corners = []
        for i, corner in enumerate(corners):
            pixel_homogeneous = np.array([corner[0], corner[1], 1])
            transformed_point = np.dot(transform_matrix, pixel_homogeneous)
            world_point = transformed_point[:2] / transformed_point[2]
            world_corners.append(tuple(world_point))

        return world_corners

    except Exception as e:
        print("Transform error:", e)
        return None


def calculate_grasp_points(world_corners):
    if len(world_corners) != 4:
        return None

    try:
        corners = np.array(world_corners)
        center = np.mean(corners, axis=0)

        centered = corners - center
        cov = np.dot(centered.T, centered)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        main_dir = eigenvectors[:, np.argmax(eigenvalues)]
        secondary_dir = eigenvectors[:, np.argmin(eigenvalues)]

        main_dist = np.abs(np.dot(centered, main_dir))
        sec_dist = np.abs(np.dot(centered, secondary_dir))
        length = 2 * np.max(main_dist)
        width = 2 * np.max(sec_dist)

        grasp_points = []
        for ratio in [0.25, 0.5, 0.75]:
            offset = (ratio - 0.5) * length * main_dir
            point = center + offset
            grasp_points.append(tuple(point))

        return grasp_points
        # --- 核心修改：在最后一步，直接把所有抓取点的 x 坐标取一半 ---
        final_grasp_points = []
        for point in grasp_points:
            # point[0] 是 x 坐标，point[1:] 是剩下的 y, z 等坐标
            new_point = (point[0] * 2 /3,) + point[1:]
            final_grasp_points.append(new_point)

        return final_grasp_points
        # -------------------------------------------------------

    except Exception as e:
        print("Grasp points error:", e)
        return None
