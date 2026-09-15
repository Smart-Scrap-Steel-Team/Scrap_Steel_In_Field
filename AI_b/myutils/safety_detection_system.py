import threading
import datetime
import time
import json
import os
from collections import defaultdict
from get_img import Camera
from ultralytics import YOLO  # 导入YOLO模型
from flask import jsonify
from logger_config import logger
from myutils.config_util import config

class SafetyDetectionSystem:
    def __init__(self):
        # 检查配置是否加载成功
        if config is None:
            logger.error("Configuration not loaded, using default settings")
            self.enabled = False
            return
        
        # 检查安全检测是否启用
        helmet_config = config.get('algorithms', {}).get('helmet_detection', {})
        fire_config = config.get('algorithms', {}).get('fire_detection', {})
        
        self.helmet_enabled = helmet_config.get('enabled', False)
        self.fire_enabled = fire_config.get('enabled', False)
        
        if not self.helmet_enabled and not self.fire_enabled:
            logger.info("Both helmet and fire detection are disabled in configuration")
            self.enabled = False
            return
        else:
            logger.info(f"Safety detection enabled - Helmet: {self.helmet_enabled}, Fire: {self.fire_enabled}")
        
        self.enabled = True
        
        # 从配置文件读取摄像头设置
        camera_config = config.get('camera', {})
        nvr_config = camera_config.get('nvr', {})
        nvr_ip = nvr_config.get('ip', "192.168.3.168")
        nvr_username = nvr_config.get('username', "admin")
        nvr_password = nvr_config.get('password', "Xd2025328")
        
        # 初始化摄像头
        self.camera = Camera(nvr_ip, nvr_username, nvr_password)
        
        # 从配置文件读取通道设置 - 统一使用火焰检测通道
        channels_config = camera_config.get('channels', {})
        self.helmet_channel = channels_config.get('helmet', 8)
        self.fire_channel = channels_config.get('fire', 7)
        
        # 初始化模型
        if self.helmet_enabled:
            helmet_model_config = helmet_config.get('model', {})
            helmet_model_path = helmet_model_config.get('path', 'models/helmet_head_person_s.pt')
            
            # 使用 torch.hub 加载 YOLOv5 模型
            try:
                import torch
                self.helmet_model = torch.hub.load('ultralytics/yolov5', 'custom', path=helmet_model_path)
                self.helmet_model.eval()  # 设置为评估模式
                logger.info(f"Helmet model loaded successfully with torch.hub from {helmet_model_path}")
            except Exception as e:
                logger.error(f"Failed to load helmet model with torch.hub: {str(e)}")
                logger.warning("Helmet detection will be disabled due to model loading failure")
                self.helmet_enabled = False
        
        if self.fire_enabled:
            fire_model_config = fire_config.get('model', {})
            fire_model_path = fire_model_config.get('path', '../models/smoke_fire_yolov8.pt')
            # YOLOv8 模型
            self.fire_model = YOLO(fire_model_path)
        
        # 初始化线程锁
        self.frame_lock = threading.Lock()
        self.frame_lock_fire = threading.Lock()
        
        # 存储最新的帧和检测结果
        self.current_frame = None
        self.current_frame_fire = None
        self.latest_results = [[] for _ in range(2)]
        self.latest_frame_times = [0] * 2
        
        # 存储检测历史
        self.detection_history = [[] for _ in range(2)]
        self.history_size = 100  # 存储最近100条记录
        
        # 从配置文件读取存储设置
        storage_config = config.get('storage', {})
        results_directory = storage_config.get('results_directory', 'detection_results')
        results_files = storage_config.get('results_files', {})
        
        # 检测结果文件路径
        self.helmet_results_file = results_files.get('helmet_detections', f"{results_directory}/helmet_detections.json")
        self.fire_results_file = results_files.get('fire_detections', f"{results_directory}/fire_detections.json")
        
        # 确保结果目录存在
        os.makedirs(results_directory, exist_ok=True)
        
        # 启动所有线程
        self.start_all_threads()
    
    def start_all_threads(self):
        """启动所有线程"""
        threads = []
        
        # 启动摄像头线程（始终启动，用于视频流显示）
        # 安全帽摄像头线程
        if self.helmet_enabled:
            helmet_camera_thread = threading.Thread(target=self.camera_stream, args=(self.helmet_channel, 'helmet'), daemon=True)
            helmet_process_thread = threading.Thread(target=self.process_frame_helmet, daemon=True)
            threads.extend([helmet_camera_thread, helmet_process_thread])
            logger.info("Helmet detection threads created")
        
        # 如果启用火焰检测，启动相关线程
        if self.fire_enabled:
            fire_camera_thread = threading.Thread(target=self.camera_stream, args=(self.fire_channel, 'fire'), daemon=True)
            fire_process_thread = threading.Thread(target=self.process_frame_fire, daemon=True)
            threads.extend([fire_camera_thread, fire_process_thread])
            logger.info("Fire detection threads created")
        
        # 启动所有线程
        for thread in threads:
            thread.start()
            logger.info(f"Started safety detection thread: {thread.name}")
    
    def camera_stream(self, channel, stream_type):
        """
        持续读取摄像头画面的线程函数
        Args:
            channel (int): 摄像头通道号
            stream_type (str): 视频流类型 ('helmet' 或 'fire')
        """
        # 检查对应检测是否启用
        if stream_type == 'helmet' and not self.helmet_enabled:
            logger.info(f"Helmet detection is disabled, skipping {stream_type} camera stream")
            return
        elif stream_type == 'fire' and not self.fire_enabled:
            logger.info(f"Fire detection is disabled, skipping {stream_type} camera stream")
            return
            
        logger.info(f"开始{stream_type}视频流，通道号为{channel}")
        try:
            while True:
                try:
                    # 获取单帧图像
                    frame = self.camera.get_camera_image_by_rtspurl(channel=channel)
                    
                    if frame is not None:
                        # 根据视频流类型选择不同的锁和帧变量
                        if stream_type == 'helmet':
                            with self.frame_lock:
                                self.current_frame = frame.copy()
                        else:
                            with self.frame_lock_fire:
                                self.current_frame_fire = frame.copy()
                    else:
                        logger.warning(f"无法获取摄像头{channel}的画面，请检查摄像头是否正常")
                    
                    time.sleep(0.1)  # 每0.1秒获取一次图像
                    
                except Exception as e:
                    logger.error(f"获取{stream_type}画面时发生错误: {str(e)}")
                    time.sleep(1)  # 发生错误时等待1秒后重试
                    
        except Exception as e:
            logger.error(f"获取{stream_type}画面时发生致命错误: {str(e)}", exc_info=True)
    
    def save_detection_results(self, results, file_path):
        """将检测结果保存到JSON文件
        Args:
            results (list): 检测结果列表
            file_path (str): 保存路径
        """
        try:
            # 读取现有数据
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        existing_data = []
            else:
                existing_data = []
            
            # 添加新结果
            for result in results:
                result['save_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                existing_data.append(result)
            
            # 保存更新后的数据
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"检测结果已保存到 {file_path}")
        except Exception as e:
            logger.error(f"保存检测结果时发生错误: {str(e)}")
    
    def process_frame_helmet(self):
        """处理安全帽检测图像并进行目标检测"""
        # 检查安全帽检测是否启用
        if not self.helmet_enabled:
            logger.info("Helmet detection is disabled, skipping helmet processing")
            return
            
        while True:
            if self.current_frame is not None:
                with self.frame_lock:
                    frame = self.current_frame.copy()
                # 使用 YOLO 模型进行检测
                results = self.helmet_model(frame)
                # 处理检测结果
                processed_results = self.process_helmet_results(results)
                self.latest_results[0] = processed_results
                logger.info(f"安全帽检测结果: {processed_results}")
                # 更新历史记录
                self.update_history(0, processed_results)
                # 保存检测结果到文件
                if processed_results:
                    self.save_detection_results(processed_results, self.helmet_results_file)
                # 更新报警信息
                if 'alarm_manager' in globals():
                    alarm_manager.update_helmet_alarm(processed_results)
            # 每10秒更新一次
            time.sleep(10)
    
    def process_frame_fire(self):
        """处理火焰烟雾检测图像并进行目标检测"""
        # 检查火焰检测是否启用
        if not self.fire_enabled:
            logger.info("Fire detection is disabled, skipping fire processing")
            return
            
        while True:
            if self.current_frame_fire is not None:
                with self.frame_lock_fire:
                    frame = self.current_frame_fire.copy()
                
                # 使用火焰烟雾检测模型
                results = self.fire_model(frame)
                
                # 处理检测结果
                processed_results = self.process_yolov8_results(results)
                self.latest_results[1] = processed_results
                logger.info(f"火焰烟雾检测结果: {processed_results}")
                
                # 更新历史记录
                self.update_history(1, processed_results)
                
                # 保存检测结果到文件
                if processed_results:  # 只在有检测结果时保存
                    self.save_detection_results(processed_results, self.fire_results_file)
                # 更新左上角报警信息
                if 'alarm_manager' in globals():
                    alarm_manager.update_fire_alarm(processed_results)
            # 每10秒更新一次
            time.sleep(10) 
    
    def process_helmet_results(self, results):
        """处理安全帽检测结果"""
        processed_results = []
        
        # 设置置信度阈值
        CONFIDENCE_THRESHOLD = 0.3  # 置信度阈值
        
        # 检查结果类型，处理不同的模型输出格式
        if hasattr(results, 'pandas'):  # torch.hub 格式
            pred = results.pandas().xyxy[0]  # 获取第一张图片的检测结果
            for _, row in pred.iterrows():
                if float(row['confidence']) < CONFIDENCE_THRESHOLD:
                    continue  # 跳过低置信度的检测
                    
                # 根据检测类别设置标签
                label = row['name']
                if label == 'helmet':
                    label = '佩戴安全帽'
                elif label == 'head':
                    label = '未正确佩戴安全帽'
                elif label == 'person':
                    label = '人员'
                
                processed_results.append({
                    'type': 'detection',
                    'label': label,
                    'confidence': float(row['confidence']),
                    'bbox': [float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])],
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        else:  # YOLO 格式
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = map(float, box.xyxy[0])
                        # 获取置信度
                        conf = float(box.conf[0])
                        # 获取类别
                        cls = int(box.cls[0])
                        class_name = result.names[cls]
                        
                        # 跳过低置信度的检测
                        if conf < CONFIDENCE_THRESHOLD:
                            continue
                        
                        # 根据检测类别设置中文标签
                        if class_name == 'helmet':
                            label = '佩戴安全帽'
                        elif class_name == 'head':
                            label = '未正确佩戴安全帽'
                        elif class_name == 'person':
                            label = '人员'
                        else:
                            label = class_name
                        
                        processed_results.append({
                            'type': 'detection',
                            'label': label,
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2],
                            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })

        logger.info(f"安全帽检测结果: {processed_results}")
        return processed_results
    
    def process_yolov8_results(self, results):
        """处理 YOLOv8 检测结果
        Args:
            results (list): 检测结果列表
        """
        processed_results = []
        # 类别映射
        label_map = {
            'Fire': '火焰',
            'default': '无火焰'
        }
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                label_en = result.names[int(cls)]
                label_cn = label_map.get(label_en, label_en)  # 默认用英文名
                processed_results.append({
                    'type': 'detection',
                    'label': label_cn,
                    'confidence': float(conf),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        logger.info(f"火焰检测结果: {processed_results}")
        return processed_results
    
    def update_history(self, cam_index, results):
        """更新检测历史"""
        self.detection_history[cam_index].extend(results)
        if len(self.detection_history[cam_index]) > self.history_size:
            self.detection_history[cam_index] = self.detection_history[cam_index][-self.history_size:]
        logger.info(f"检测历史更新完成，当前历史记录长度为{len(self.detection_history[cam_index])}")
    
    def get_latest_frame(self, cam_index):
        """获取指定摄像头的最新帧"""
        if cam_index == 0:
            with self.frame_lock:
                return self.current_frame
        else:
            with self.frame_lock_fire:
                return self.current_frame_fire
    
    def get_latest_results(self, cam_index):
        """获取指定摄像头的最新检测结果"""
        return self.latest_results[cam_index]
    
    def get_history(self, cam_index, start_time=None, end_time=None):
        """获取指定时间范围内的历史记录
        Args:
            cam_index (int): 摄像头索引
            start_time (str): 开始时间 (格式: "%Y-%m-%d")
            end_time (str): 结束时间 (格式: "%Y-%m-%d")
        """
        history = self.detection_history[cam_index]
        if start_time and end_time:
            start_time = datetime.datetime.strptime(start_time, "%Y-%m-%d")
            end_time = datetime.datetime.strptime(end_time, "%Y-%m-%d")
            history = [record for record in history 
                     if start_time <= datetime.datetime.strptime(record['timestamp'], "%Y-%m-%d %H:%M:%S") <= end_time]
        return history
    
    def is_belong_to(self, inner_box, outer_box, iou_threshold=0.3):
        # 判断inner_box与outer_box的IOU是否大于阈值
        return self.bbox_iou(inner_box, outer_box) > iou_threshold

    def bbox_iou(self, boxA, boxB):
        """
        计算两个bbox的IOU
        boxA, boxB: [x1, y1, x2, y2]
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou