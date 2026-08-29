import time
import os
from datetime import datetime
import json
from inference.scrap_segmentation import simple_segment
from config.detection_config import SAVE_CONFIG

try:
    from myutils.logger import setup_logger
    logger = setup_logger('scrap_segmentation', SAVE_CONFIG['base_dir'], SAVE_CONFIG['sub_dirs'].get('log', 'log'))
except Exception:
    logger = None

def start_scrap_segmentation():
    """废钢分割模块主循环，可独立运行，异常不影响主循环和前端界面"""
    image_dir = os.path.join(SAVE_CONFIG["base_dir"], "original")
    segmentation_json_path = os.path.join(SAVE_CONFIG["base_dir"], "segmentation_result.json")
    processed = set()
    while True:
        try:
            image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png"))]
            for fname in image_files:
                fpath = os.path.join(image_dir, fname)
                if fpath not in processed:
                    try:
                        detections, mask, save_path, class_counts, area_stats = simple_segment(image_path=fpath)
                        processed.add(fpath)
                        if logger:
                            if mask is not None:
                                logger.info(f"[分割完成] {fname} -> {save_path}")
                                if len(detections) > 0:
                                    logger.info(f"检测到 {len(detections)} 个物体")
                                    logger.info(f"类别统计: {class_counts}")
                                    logger.info(f"面积统计: {area_stats}")
                                else:
                                    logger.info("未检测到物体")
                                logger.info(f"废钢分割结果保存路径: {save_path}")
                            else:
                                logger.warning("[警告] 废钢分割失败，将按默认流程继续")
                        else:
                            print(f"[INFO] Segmentation done: {fname} -> {save_path}")
                        # 分割成功时写入JSON
                        if mask is not None:
                            segmentation_result = {
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "image_path": fpath,
                                "result_path": save_path,
                                "detections": detections,
                                "class_counts": class_counts,
                                "area_stats": area_stats
                            }
                            try:
                                with open(segmentation_json_path, 'a', encoding='utf-8') as f:
                                    json.dump(segmentation_result, f, ensure_ascii=False)
                                    f.write('\n')
                                if logger:
                                    logger.info(f"废钢分割结果已写入JSON文件: {segmentation_json_path}")
                                else:
                                    print(f"[INFO] Segmentation result written to {segmentation_json_path}")
                            except Exception as e:
                                if logger:
                                    logger.error(f"写入分割结果到JSON文件时出错: {str(e)}")
                                else:
                                    print(f"[ERROR] Failed to write segmentation result: {str(e)}")
                    except Exception as e:
                        if logger:
                            logger.error(f"[分割异常] {fname}: {str(e)}")
                        else:
                            print(f"[ERROR] Segmentation failed for {fname}: {str(e)}")
        except Exception as e:
            if logger:
                logger.error(f"[主循环异常] {str(e)}")
            else:
                print(f"[ERROR] Main loop exception: {str(e)}")
        time.sleep(5) 