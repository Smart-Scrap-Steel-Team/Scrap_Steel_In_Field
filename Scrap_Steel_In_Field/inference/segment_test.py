import cv2
import numpy as np
import os
import sys
import logging
import json
from datetime import datetime
from config.segment_config import MODEL_CONFIG, SAVE_CONFIG, LOG_CONFIG
from utils.logger import setup_logger

# --- 新增：导入 Ultralytics YOLO ---
from ultralytics import YOLO

# --- 保持原有的 COLOR_MAP 和 logger 定义 ---
logger = setup_logger('segmentation', LOG_CONFIG['base_dir'], LOG_CONFIG['sub_dirs']['segmentation'])

COLOR_MAP = {
    0: (0, 0, 0),  # 方管1
    1: (128, 0, 0),  # 方管2
    2: (0, 128, 0),  # 方管3
    3: (128, 128, 0),  # 圆管1
    4: (0, 0, 128),  # 圆管2
    5: (128, 0, 128),  # 圆管3
    6: (0, 128, 128),  # 平整钢板
    7: (192, 192, 192),  # 长条钢板1
    8: (128, 64, 128),  # 长钢板2
    9: (255, 0, 0),  # 长条钢板3
    10: (128, 128, 128),  # 角钢
    11: (64, 64, 64),  # 钢筋1
    12: (192, 0, 0),  # 钢筋2
    13: (128, 192, 0),  # 管道连接件
    14: (192, 128, 0),  # 金属连接件
    15: (0, 64, 0)  # 圆铁片
}


# --- 1. 定义一个兼容的推理函数 ---
def run_yolo_inference(image_path, model_path, conf_threshold=0.25):
    """
    使用 YOLO 模型执行推理 (强制使用 CPU 并指定任务类型)
    """
    try:
        # 1. 加载模型 (显式指定任务类型，防止 WARNING)
        # 请根据你的实际模型类型修改 task='segment' 或 task='detect'
        model = YOLO(model_path, task='segment')

        # 2. 读取图像
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

        # 3. 执行预测 (强制使用 CPU)
        results = model.predict(
            source=original_image,
            conf=conf_threshold,
            imgsz=MODEL_CONFIG["model_input_size"],
            device='cpu'  # <--- 关键修复：改为 'cpu'
        )

        # 4. 提取数据...
        # (后续代码保持不变，因为数据结构已经标准化)
        result = results[0]

        # ... (后续处理 detections, mask 的代码保持不变)
        # 如果之前的 "detections" 赋值报错，请继续使用我们之前讨论的 np.array() 强制转换逻辑
        # detections = np.array(result.boxes.xyxy.cpu().numpy()) ... etc

        # --- 4.1 提取检测框 (detections) ---
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

            detections = np.concatenate([
                boxes,
                scores.reshape(-1, 1),
                class_ids.reshape(-1, 1)
            ], axis=1)
        else:
            detections = np.array([])

        # --- 4.2 提取分割掩码 (mask) ---
        if result.masks is not None:
            masks_tensor = result.masks.data.cpu().numpy()
            h, w = original_image.shape[:2]
            mask = np.zeros((640, 640), dtype=np.int32)

            for i in range(len(masks_tensor)):
                cls_id = int(class_ids[i]) if i < len(class_ids) else 0
                mask[masks_tensor[i] > 0.5] = cls_id + 1

            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            h, w = original_image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

        # --- 4.3 统计信息 ---
        class_counts = {}
        area_stats = {}
        unique, counts = np.unique(mask, return_counts=True)
        for u, c in zip(unique, counts):
            if u > 0:
                cls_id = u - 1
                if cls_id < len(MODEL_CONFIG["categories"]):
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                    area_stats[cls_id] = area_stats.get(cls_id, 0) + c

        return detections, mask, class_counts, area_stats, original_image

    except Exception as e:
        logger.error(f"YOLO 推理失败: {e}")
        return None, None, {}, {}, None


# --- 2. 保持原有的 save_segmentation_results 函数不变 ---
# (这里直接粘贴你上传代码中的 save_segmentation_results 函数)
# ... (代码太长，此处省略，保持原样即可) ...

# --- 3. 修改主运行逻辑 ---
def main():
    image_path = "test_image.jpg"  # 替换为你的测试图片路径
    if not os.path.exists(image_path):
        print(f"错误：图像文件不存在: {image_path}")
        return

    print("正在使用 YOLO 框架进行推理...")
    detections, mask, class_counts, area_stats, original_image = run_yolo_inference(
        image_path=image_path,
        model_path=MODEL_CONFIG["model_path"],  # 你的 .pt 模型路径
        conf_threshold=MODEL_CONFIG["conf_threshold"]
    )

    if original_image is not None:
        # 构建保存路径
        save_dir = os.path.join(SAVE_CONFIG["base_dir"], SAVE_CONFIG["sub_dirs"]["segmentation"])
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"segment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")

        # 调用原有的保存逻辑 (这里应该不会再报错了，因为数据是标准的)
        success = save_segmentation_results(original_image, mask, detections, save_path, class_counts, area_stats)

        if success:
            print(f"分割完成，结果保存至: {save_path}")
            print(f"检测到 {len(detections)} 个物体")
            print(f"类别统计: {class_counts}")
            print(f"面积统计: {area_stats}")
        else:
            print("保存结果失败")
    else:
        print("推理失败")


if __name__ == "__main__":
    main()