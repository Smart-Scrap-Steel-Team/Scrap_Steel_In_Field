import time
import os
from datetime import datetime
import json
from inference.detection_inference_simplified import simple_detect
from config.detection_config import SAVE_CONFIG

try:
    from myutils.logger import setup_logger
    logger = setup_logger('dangerous_detection', SAVE_CONFIG['base_dir'], SAVE_CONFIG['sub_dirs'].get('log', 'log'))
except Exception:
    logger = None

def start_dangerous_detection():
    """危险物检测模块主循环，可独立运行，异常不影响主循环和前端界面"""
    image_dir = os.path.join(SAVE_CONFIG["base_dir"], "original")
    detection_json_path = os.path.join(SAVE_CONFIG["base_dir"], "detection_results.json")
    processed = set()
    danger_types = ['GasCyl1', 'GasCyl2', 'FireExt1', 'FireExt2']
    while True:
        try:
            image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png"))]
            for fname in image_files:
                fpath = os.path.join(image_dir, fname)
                if fpath not in processed:
                    try:
                        detections, detect_success = simple_detect(image_path=fpath, config=None)
                        processed.add(fpath)
                        if logger:
                            if detect_success:
                                logger.info(f"[检测完成] {fname} -> 检测成功")
                            else:
                                logger.warning(f"[检测完成] {fname} -> 检测失败")
                        else:
                            print(f"[INFO] Detection done: {fname}, success={detect_success}")
                        # 检测成功时写入JSON
                        if detect_success:
                            counts = {dtype: 0 for dtype in danger_types}
                            for det in detections:
                                if det.get('label') in danger_types:
                                    counts[det['label']] += 1
                            detection_result = {
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "image_path": fpath,
                                "detections": detections,
                                "counts": counts,
                                "total_dangerous": sum(counts.values())
                            }
                            try:
                                with open(detection_json_path, 'a', encoding='utf-8') as f:
                                    json.dump(detection_result, f, ensure_ascii=False)
                                    f.write('\n')
                                if logger:
                                    logger.info(f"危险品检测结果已写入JSON文件: {detection_json_path}")
                                else:
                                    print(f"[INFO] Detection result written to {detection_json_path}")
                            except Exception as e:
                                if logger:
                                    logger.error(f"写入检测结果到JSON文件时出错: {str(e)}")
                                else:
                                    print(f"[ERROR] Failed to write detection result: {str(e)}")
                    except Exception as e:
                        if logger:
                            logger.error(f"[检测异常] {fname}: {str(e)}")
                        else:
                            print(f"[ERROR] Detection failed for {fname}: {str(e)}")
        except Exception as e:
            if logger:
                logger.error(f"[主循环异常] {str(e)}")
            else:
                print(f"[ERROR] Main loop exception: {str(e)}")
        time.sleep(5) 