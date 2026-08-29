import threading
import datetime
import time
import json
import os
from collections import deque
from flask import jsonify
import cv2
import requests
from myutils.vehicle_information import VehicleInformation
from get_img import Camera
from logger_config import logger
from myutils.config_util import config
from ultralytics import YOLO  # 导入YOLO模型

class VehicleDetection:
    def __init__(self):
        # 检查配置是否加载成功
        if config is None:
            logger.error("Configuration not loaded, using default settings")
            self.enabled = False
        else:
            # 检查车辆检测是否启用
            vehicle_config = config.get('algorithms', {}).get('vehicle_detection', {})
            self.enabled = vehicle_config.get('enabled', False)
        
        # 初始化基本属性（无论是否启用都需要）
        self.frame_lock = threading.Lock()
        self.frame_lock_exit = threading.Lock()
        self.current_frame = None
        self.current_frame_exit = None
        self.detection_history = deque(maxlen=5)
        self.detection_history_exit = deque(maxlen=5)
        self.vehicle_information = None
        self.model = None
        self.last_recorded_plates = {}
        self.entrance_camera = None
        self.exit_camera = None
        self.entrance_gate_timer = None
        self.exit_gate_timer = None
        self.entrance_gate_status = 'close'
        self.exit_gate_status = 'close'
        
        # 如果车辆检测被禁用，只初始化基本属性
        if not self.enabled:
            logger.info("Vehicle detection is disabled in configuration")
            return
        
        # 从配置文件读取摄像头设置
        camera_config = config.get('camera', {})
        nvr_config = camera_config.get('nvr', {})
        self.nvr_ip = nvr_config.get('ip', "192.168.3.168")
        self.nvr_username = nvr_config.get('username', "admin")
        self.nvr_password = nvr_config.get('password', "Xd2025328")
        
        # 从配置文件读取场景设置
        self.entrance_enabled = vehicle_config.get('entrance_scene', {}).get('enabled', True)
        self.exit_enabled = vehicle_config.get('exit_scene', {}).get('enabled', True)
        self.entrance_channel = vehicle_config.get('entrance_scene', {}).get('camera_channel', 2)
        self.exit_channel = vehicle_config.get('exit_scene', {}).get('camera_channel', 6)
        self.entrance_auto_gate = vehicle_config.get('entrance_scene', {}).get('auto_gate_control', True)
        self.exit_auto_gate = vehicle_config.get('exit_scene', {}).get('auto_gate_control', True)
        self.entrance_record_interval = vehicle_config.get('entrance_scene', {}).get('record_interval', 60)
        self.exit_record_interval = vehicle_config.get('exit_scene', {}).get('record_interval', 60)
        
        # 全局变量存储当前帧
        self.frame_lock = threading.Lock()
        self.frame_lock_exit = threading.Lock()
        self.current_frame = None
        self.current_frame_exit = None
        # 设置最大长度为5的入场队列
        self.detection_history = deque(maxlen=5)
        # 设置最大长度为5的出场队列
        self.detection_history_exit = deque(maxlen=5)
        # 加载 YOLO 模型
        self.vehicle_information = VehicleInformation(config)
        model_config = vehicle_config.get('model', {})
        model_path = model_config.get('path', 'models/best.pt')
        logger.info(f"Loading YOLO model from {model_path}")
        self.model = YOLO(model_path, task='detect')
        self.init_detection_history()
        # 添加最近记录的车牌和时间的字典
        self.last_recorded_plates = {}
        # 设置最小记录间隔（秒）
        self.min_record_interval = vehicle_config.get('processing', {}).get('min_record_interval', 60)
        
        # 初始化摄像头实例
        self.entrance_camera = None
        self.exit_camera = None
        # 添加闸机状态跟踪变量，并修改闸机控制逻辑
        self.entrance_gate_timer = None
        self.exit_gate_timer = None
        self.entrance_gate_status = 'close'  # 入场闸机状态
        self.exit_gate_status = 'close'      # 出场闸机状态
        
        # 闸机控制配置
        gate_config = vehicle_config.get('gate_control', {})
        self.gate_control_enabled = gate_config.get('enabled', True)
        self.auto_close_delay = gate_config.get('auto_close_delay', 30)
        self.entry_gate_url = gate_config.get('entry_gate_url', "http://192.168.3.204:5000/remote/entry_gate")
        self.exit_gate_url = gate_config.get('exit_gate_url', "http://192.168.3.204:5000/remote/exit_gate")
        
        logger.info(f"VehicleDetection system initialized successfully")
        logger.info(f"Entrance scene enabled: {self.entrance_enabled}, Exit scene enabled: {self.exit_enabled}")

    def detect_license_plate(self, frame, stream_type):
        """
        检测并识别车牌
        :param frame: 输入图像
        :return: (车牌号, 处理后的图像)
        """
        if not self.enabled or self.model is None or self.vehicle_information is None:
            logger.warning("Vehicle detection is disabled or not properly initialized")
            return None, None, None, frame
            
        try:
            # 使用 YOLO 检测车辆
            results = self.model(frame)
            # 记录检测结果
            if results and len(results) > 0:
                for result in results:
                    boxes = result.boxes
                    if len(boxes) > 0:
                        # 获取检测到的类别和置信度
                        detections = []
                        for box in boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = self.vehicle_information.names[cls]
                            detections.append(f"{class_name} ({conf:.2f})")
                        logger.info(f"线程{stream_type} YOLO检测到: {', '.join(detections)}")
                    else:
                        logger.info(f"线程{stream_type} YOLO未检测到任何目标")
            else:
                logger.info(f"线程{stream_type} YOLO未检测到任何目标")
            
            # 创建帧的副本用于绘制
            display_frame = frame.copy()
            # 初始化返回值
            plate_text = None
            conf_score = None
            plate_color = None
            
            # 在图像上绘制检测结果
            if results and len(results) > 0:
                # 绘制检测框和标签
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # 获取置信度
                        conf = float(box.conf[0])
                        # 获取类别
                        cls = int(box.cls[0])
                        # 绘制边界框
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                        # 获取车牌信息
                        plate_text, conf_score, plate_color = self.vehicle_information.get_vehicle_information([result], frame)
                        # 如果车牌号无效，返回None以触发重新识别
                        if plate_text and not self.vehicle_information.validate_chinese_char(plate_text):
                            logger.warning(f"线程{stream_type} 中无效的中文字符在车牌中: {plate_text}")
                            return None, None, None, display_frame
                        
                        # 准备标签文本
                        if plate_text and conf_score:
                            label = f"{plate_text} ({plate_color}) {conf_score:.2f}"
                            logger.info(f"线程{stream_type} 检测到车牌: {label}")
                        else:
                            label = f"{self.vehicle_information.names[cls]} {conf:.2f}"
                            logger.debug(f"线程{stream_type} 检测到对象: {label}")
                        
                        # 绘制标签背景
                        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(display_frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)
                        
                        # 绘制标签文本
                        cv2.putText(display_frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                return plate_text, conf_score, plate_color, display_frame
            else:
                logger.debug(f"线程{stream_type} 未在帧中检测到任何对象")
                return None, None, None, display_frame
        except Exception as e:
            logger.error(f"线程{stream_type} 的detect_license_plate中发生错误: {str(e)}", exc_info=True)
            return None, None, None, frame

    def process_frame_entrance(self):
        """处理入场图像并进行目标检测"""
        # 检查车辆检测是否启用
        if not self.enabled or self.vehicle_information is None:
            logger.info("Vehicle detection is disabled or not properly initialized, skipping entrance processing")
            return
            
        # 检查入场场景是否启用
        if not self.entrance_enabled:
            logger.info("Entrance scene is disabled, skipping entrance processing")
            return
            
        while True:
            if self.current_frame is not None:
                with self.frame_lock:
                    frame = self.current_frame.copy()
                # 检测并识别车牌
                plate_number, conf_score, plate_color, processed_frame = self.detect_license_plate(frame, 'entrance')
                current_time = datetime.datetime.now()
                time_str = current_time.strftime('%H:%M:%S')
                
                if plate_number and conf_score:
                    # 检查是否应该记录这个车牌
                    should_record = self._check_plate_record(plate_number, current_time, 'entry')
                    
                    if should_record:
                        # 记录车牌信息与到达时间
                        self.vehicle_information.record_plate_entry(plate_number, time_str, plate_color)
                        # 更新最后记录时间
                        self.last_recorded_plates[plate_number] = current_time

                        # 更新前端显示的记录信息
                        detection = {
                            'time': time_str,
                            'plate_number': plate_number,
                            'confidence': conf_score,
                            'plate_color': plate_color,
                            'status': 'entry'
                        }
                        # 获取检测历史记录
                        self.add_detection_history(detection)
                        with self.frame_lock:
                            self.current_frame = processed_frame.copy()
                        
                        # 如果启用自动闸机控制，则开启入场闸机
                        if self.entrance_auto_gate and self.gate_control_enabled:
                            self.control_door('open', 'entrance')
                            logger.info(f"入场闸机已开启，车牌号: {plate_number}")
                        
                        # 车牌成功记录后，当前线程休眠配置的间隔时间
                        logger.info(f"入场线程: 车牌 {plate_number} 记录成功，线程休眠{self.entrance_record_interval}秒")
                        time.sleep(self.entrance_record_interval)
                        continue  # 跳过本次循环的剩余部分
            time.sleep(0.5)

    def process_frame_exit(self):
        """处理出场图像并进行目标检测"""
        # 检查车辆检测是否启用
        if not self.enabled or self.vehicle_information is None:
            logger.info("Vehicle detection is disabled or not properly initialized, skipping exit processing")
            return
            
        # 检查出场场景是否启用
        if not self.exit_enabled:
            logger.info("Exit scene is disabled, skipping exit processing")
            return
            
        while True:
            if self.current_frame_exit is not None:
                with self.frame_lock_exit:
                    frame = self.current_frame_exit.copy()
                # 检测并识别车牌
                plate_number, conf_score, plate_color, processed_frame = self.detect_license_plate(frame, 'exit')
                current_time = datetime.datetime.now()
                time_str = current_time.strftime('%H:%M:%S')
                logger.info(f"出厂车牌识别信息: {plate_number}, {conf_score}, {plate_color},{time_str}")
                if plate_number and conf_score:
                    # 检查是否应该记录这个车牌
                    should_record = self._check_plate_record(plate_number, current_time, 'exit')
                    logger.info(f"出厂车牌是否记录: {should_record}")
                    if should_record:
                        # 检查车辆是否在场
                        if plate_number in self.vehicle_information.current_vehicles:
                            # 记录车牌信息与出场时间
                            if self.vehicle_information.record_plate_exit(plate_number, time_str):
                                # 更新最后记录时间
                                self.last_recorded_plates[plate_number] = current_time

                                # 更新前端显示的记录信息
                                detection = {
                                    'time': time_str,
                                    'plate_number': plate_number,
                                    'confidence': conf_score,
                                    'plate_color': plate_color,
                                    'status': 'exit'
                                }
                                # 添加出场检测历史记录
                                self.add_detection_history_exit(detection)
                                # 更新当前帧
                                with self.frame_lock_exit:
                                    self.current_frame_exit = processed_frame.copy()
                                
                                # 如果启用自动闸机控制，则开启出场闸机
                                if self.exit_auto_gate and self.gate_control_enabled:
                                    self.control_door('open', 'exit')
                                    logger.info(f"出场闸机已开启，车牌号: {plate_number}")
                                
                                logger.info(f"出厂车牌记录成功: {plate_number}")
                                # 出厂车牌记录成功后，当前线程休眠配置的间隔时间
                                logger.info(f"出场线程: 车牌 {plate_number} 记录成功，线程休眠{self.exit_record_interval}秒")
                                time.sleep(self.exit_record_interval)
                                continue  # 跳过本次循环的剩余部分
                            else:
                                logger.warning(f"出厂车牌记录失败: {plate_number}")

                        else:
                            # 记录异常车辆信息
                            self.record_abnormal_vehicle(plate_number, time_str, conf_score, plate_color)
                            # 更新前端显示的记录信息
                            detection = {
                                'time': time_str,
                                'plate_number': plate_number,
                                'confidence': conf_score,
                                'plate_color': plate_color,
                                'status': 'abnormal'
                            }
                            # 添加异常检测历史记录
                            self.add_detection_history_exit(detection)
                            # 更新当前帧
                            with self.frame_lock_exit:
                                self.current_frame_exit = processed_frame.copy()
                            logger.warning(f"异常车辆记录成功: {plate_number}")
                            # 异常车辆记录成功后，当前线程休眠配置的间隔时间
                            logger.info(f"出场线程: 异常车辆 {plate_number} 记录成功，线程休眠{self.exit_record_interval}秒")
                            time.sleep(self.exit_record_interval)
                            continue  # 跳过本次循环的剩余部分
            time.sleep(0.5)

    def _check_plate_record(self, plate_number, current_time, status='entry'):
        """
        从json文件中检查是否应该记录这个车牌
        :param plate_number: 车牌号
        :param current_time: 当前时间
        :param status: 车辆状态 ('entry', 'exit', 'abnormal')
        :return: True if should record, False otherwise
        """
        if not self.enabled or self.vehicle_information is None:
            return False
            
        try:
            # 根据状态选择要读取的文件
            if status == 'entry':
                # 只读取入场记录
                with open(self.vehicle_information.entry_record_file, 'r', encoding='utf-8') as f:
                    records = json.load(f)
            else:
                # 出场和异常状态读取出场和异常记录
                records = []
                # 读取出场记录
                with open(self.vehicle_information.exit_record_file, 'r', encoding='utf-8') as f:
                    exit_records = json.load(f)
                    if exit_records:
                        records.extend(exit_records)
                # 读取异常记录
                with open(self.vehicle_information.abnormal_record_file, 'r', encoding='utf-8') as f:
                    abnormal_records = json.load(f)
                    if abnormal_records:
                        records.extend(abnormal_records)
            
            # 如果没有记录，直接返回True
            if not records:
                return True
            
            # 按时间戳排序，获取最近的记录
            records.sort(key=lambda x: datetime.datetime.strptime(x['timestamp'], '%Y-%m-%d %H:%M:%S'), 
                        reverse=True)
            
            # 查找该车牌在最近记录中的记录
            for record in records:
                if record['plate_number'] == plate_number:
                    # 计算时间差
                    last_time = datetime.datetime.strptime(record['timestamp'], '%Y-%m-%d %H:%M:%S')
                    time_diff = (current_time - last_time).total_seconds()
                    logger.info(f"Time difference: {time_diff} status: {record.get('status')}")
                    
                    # 如果时间差小于最小记录间隔
                    if time_diff < self.min_record_interval:
                        logger.info(f"Vehicle {plate_number} was recorded as {record.get('status')} within {self.min_record_interval} seconds, skipping...")
                        return False
            
            # 如果没有找到记录，允许记录
            return True
            
        except (FileNotFoundError, json.JSONDecodeError):
            # 如果文件不存在或损坏，允许记录
            return True
        except Exception as e:
            logger.error(f"Error checking plate record: {str(e)}", exc_info=True)
            # 发生其他错误时，为安全起见允许记录
            return True
    
    def init_detection_history(self):
        if not self.enabled or self.vehicle_information is None:
            logger.info("Vehicle detection is disabled, skipping history initialization")
            return
            
        try:
            # 初始化入场历史
            with open(self.vehicle_information.entry_record_file, 'r', encoding='utf-8') as f:
                entry_records = json.load(f)
            
            if entry_records:  # 如果文件不为空
                # 按时间戳排序，获取最新的3条记录
                sorted_records = sorted(entry_records, 
                                     key=lambda x: datetime.datetime.strptime(x['timestamp'], '%Y-%m-%d %H:%M:%S'), 
                                     reverse=True)[:3]
                
                # 添加到队列中
                for record in sorted_records:
                    self.detection_history.append({
                        'time': record['time'],
                        'plate_color': record['plate_color'],
                        'plate_number': record['plate_number'],
                        'confidence': record.get('confidence', 1.0)
                    })
            
            # 初始化出场历史
            with open(self.vehicle_information.exit_record_file, 'r', encoding='utf-8') as f:
                exit_records = json.load(f)
            
            if exit_records:  # 如果文件不为空
                # 按时间戳排序，获取最新的3条记录
                sorted_records = sorted(exit_records, 
                                     key=lambda x: datetime.datetime.strptime(x['timestamp'], '%Y-%m-%d %H:%M:%S'), 
                                     reverse=True)[:3]
                
                # 添加到队列中
                for record in sorted_records:
                    self.detection_history_exit.append({
                        'time': record['time'],
                        'plate_color': record['plate_color'],
                        'plate_number': record['plate_number'],
                        'confidence': record.get('confidence', 1.0)
                    })
                
        except (FileNotFoundError, json.JSONDecodeError):
            # 如果文件不存在或为空，使用空队列
            logger.warning("No existing records found, starting with empty history.")
            self.detection_history = deque(maxlen=5)
            self.detection_history_exit = deque(maxlen=5)
        except Exception as e:
            # 其他错误处理
            logger.error(f"Error initializing detection history: {str(e)}", exc_info=True)
            # 确保即使出错也有空队列
            self.detection_history = deque(maxlen=5)
            self.detection_history_exit = deque(maxlen=5)

    def add_detection_history(self, detection):
        """添加检测历史记录"""
        self.detection_history.appendleft(detection)
        if len(self.detection_history) > 3:  # 保留最近的3条记录
            self.detection_history.pop()

    def add_detection_history_exit(self, detection):
        """添加出场检测历史记录"""
        self.detection_history_exit.appendleft(detection)
        if len(self.detection_history_exit) > 3:  # 保留最近的3条记录
            self.detection_history_exit.pop()

    def get_detection_history_entrance(self):
        """获取入场检测历史记录"""
        return jsonify(list(self.detection_history))

    def get_detection_history_exit(self):
        """获取出场检测历史记录"""
        return jsonify(list(self.detection_history_exit))

    def get_current_vehicles_info(self):
        """获取当前在场车辆信息"""
        if not self.enabled or self.vehicle_information is None:
            return jsonify({
                'count': 0,
                'vehicles': {}
            })
        current_vehicles = self.vehicle_information.get_current_vehicles()
        return jsonify({
            'count': len(current_vehicles),
            'vehicles': current_vehicles
        })

    def camera_stream(self, camera_index, stream_type='entrance'):
        """
        持续读取摄像头画面的线程函数
        Args:
            camera_index (int): 摄像头通道号 (1-8)
            stream_type (str): 视频流类型 ('entrance' 或 'exit')
        """
        # 检查对应场景是否启用
        if stream_type == 'entrance' and not self.entrance_enabled:
            logger.info(f"Entrance scene is disabled, skipping {stream_type} camera stream")
            return
        elif stream_type == 'exit' and not self.exit_enabled:
            logger.info(f"Exit scene is disabled, skipping {stream_type} camera stream")
            return
            
        logger.info(f"Starting {stream_type} camera stream with index {camera_index}")
        try:
            # 初始化摄像头实例
            camera = Camera(self.nvr_ip, self.nvr_username, self.nvr_password)
            if stream_type == 'entrance':
                self.entrance_camera = camera
            else:
                self.exit_camera = camera
            
            while True:
                try:
                    # 获取单帧图像
                    frame = camera.get_camera_image_by_rtspurl(channel=camera_index)
                    
                    if frame is not None:
                        # 根据视频流类型选择不同的锁和帧变量
                        if stream_type == 'entrance':
                            with self.frame_lock:
                                self.current_frame = frame.copy()
                        else:
                            with self.frame_lock_exit:
                                self.current_frame_exit = frame.copy()
                    else:
                        logger.warning(f"Failed to get frame from camera {camera_index} for {stream_type}")
                    
                    time.sleep(0.1)  # 每0.1秒获取一次图像
                    
                except Exception as e:
                    logger.error(f"Error in getting frame for {stream_type}: {str(e)}")
                    time.sleep(1)  # 发生错误时等待1秒后重试
                    
        except Exception as e:
            logger.error(f"Fatal error in camera_stream for {stream_type}: {str(e)}", exc_info=True)
        finally:
            # 确保在结束时释放资源
            if stream_type == 'entrance' and self.entrance_camera:
                self.entrance_camera = None
            elif self.exit_camera:
                self.exit_camera = None

    def generate_frames_entrance(self):
        """生成入场图像流"""
        while True:
            with self.frame_lock:
                if self.current_frame is not None:
                    # 将 OpenCV 图像编码为 JPEG
                    ret, buffer = cv2.imencode('.jpg', self.current_frame)
                    if ret:
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
    
    def generate_frames_exit(self):
        """生成出场图像流"""
        while True:
            with self.frame_lock_exit:
                if self.current_frame_exit is not None:
                    # 将 OpenCV 图像编码为 JPEG
                    ret, buffer = cv2.imencode('.jpg', self.current_frame_exit)
                    if ret:
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)

    def record_abnormal_vehicle(self, plate_number, time_str, confidence, plate_color):
        """记录异常车辆信息到JSON文件"""
        if not self.enabled:
            logger.info("Vehicle detection is disabled, skipping abnormal vehicle recording")
            return
            
        try:
            # 创建异常记录文件路径
            abnormal_file = 'data/abnormal_vehicles.json'
            os.makedirs(os.path.dirname(abnormal_file), exist_ok=True)
            
            # 读取现有记录
            records = []
            if os.path.exists(abnormal_file):
                with open(abnormal_file, 'r', encoding='utf-8') as f:
                    try:
                        records = json.load(f)
                    except json.JSONDecodeError:
                        records = []
            
            # 添加新记录
            new_record = {
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'time': time_str,
                'plate_number': plate_number,
                'confidence': confidence,
                'plate_color': plate_color,
                'status': 'abnormal'
            }
            records.append(new_record)
            
            # 保存记录
            with open(abnormal_file, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Recorded abnormal vehicle {plate_number} to {abnormal_file}")
            
        except Exception as e:
            logger.error(f"Error recording abnormal vehicle: {str(e)}", exc_info=True)

    def control_door(self, action, stream_type='entrance'):
        """
        控制进出场闸机的开关
        args:
            action (str): 'open' 或 'close'
        """
        # 获取对应的闸机状态和定时器
        gate_status = self.entrance_gate_status if stream_type == 'entrance' else self.exit_gate_status
        gate_timer = self.entrance_gate_timer if stream_type == 'entrance' else self.exit_gate_timer
        
        # 进场闸机的控制接口
        base_url_entry = "http://192.168.3.204:5000/remote/entry_gate"
        # 出场闸机的控制接口
        base_url_exit = "http://192.168.3.204:5000/remote/exit_gate"
    
        # 选择请求URL
        base_url = base_url_entry if stream_type == 'entrance' else base_url_exit
    
        # 如果是开启闸机的请求
        if action == 'open':
            # 如果闸机已经开启，取消之前的定时关闭
            if gate_timer is not None:
                gate_timer.cancel()
                if stream_type == 'entrance':
                    self.entrance_gate_timer = None
                else:
                    self.exit_gate_timer = None
                logger.info(f"{stream_type}闸机已重置定时器")
            
            # 如果闸机当前是关闭状态，则开启
            if gate_status == 'close':
                # 构建开启请求参数
                params = {
                    "entry_gate_state" if stream_type == 'entrance' else "exit_gate_state": "true"
                }
                
                try:
                    response = requests.get(base_url, params=params, timeout=5)
                    response.raise_for_status()
                    logger.info(f"{stream_type}闸机开启成功. Response: {response.text}")
                    # 更新闸机状态
                    if stream_type == 'entrance':
                        self.entrance_gate_status = 'open'
                    else:
                        self.exit_gate_status = 'open'
                except requests.RequestException as e:
                    logger.error(f"开启{stream_type}闸机失败: {str(e)}", exc_info=True)
                    return False
            
            # 设置新的定时器（30秒后关闭）
            new_timer = threading.Timer(30, self.control_door, args=['close', stream_type])
            new_timer.start()
            if stream_type == 'entrance':
                self.entrance_gate_timer = new_timer
            else:
                self.exit_gate_timer = new_timer
            logger.info(f"已设置{stream_type}闸机30秒后自动关闭")
            
        # 如果是关闭闸机的请求
        elif action == 'close':
            # 构建关闭请求参数
            params = {
                "entry_gate_state" if stream_type == 'entrance' else "exit_gate_state": "false"
            }
            
            try:
                response = requests.get(base_url, params=params, timeout=5)
                response.raise_for_status()
                logger.info(f"{stream_type}闸机关闭成功. Response: {response.text}")
                # 更新闸机状态
                if stream_type == 'entrance':
                    self.entrance_gate_status = 'close'
                    self.entrance_gate_timer = None
                else:
                    self.exit_gate_status = 'close'
                    self.exit_gate_timer = None
            except requests.RequestException as e:
                logger.error(f"关闭{stream_type}闸机失败: {str(e)}", exc_info=True)
                return False
        
        return True
