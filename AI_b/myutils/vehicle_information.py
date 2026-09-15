import json
import os
import re
from paddleocr import PaddleOCR
import easyocr
from datetime import datetime
import cv2
from logger_config import logger
import numpy as np
from ultralytics import YOLO  # 添加YOLO导入
import time

class VehicleInformation:
    def __init__(self, config=None):
        self.names = ['GreenLicense','BlueLicense']
        
        # 从配置文件读取OCR设置
        if config is None:
            logger.warning("Configuration not provided, using default OCR settings")
            self.paddleocr_enabled = True
            self.easyocr_enabled = True
            self.paddleocr_config = {
                'use_angle_cls': True,
                'lang': 'ch',
                'confidence_threshold': 0.7
            }
            self.easyocr_config = {
                'languages': ['ch_sim', 'en'],
                'gpu': False,
                'confidence_threshold': 0.7
            }
        else:
            # 读取OCR配置
            ocr_config = config.get('algorithms', {}).get('vehicle_detection', {}).get('ocr', {})
            
            # PaddleOCR配置
            paddleocr_config = ocr_config.get('paddleocr', {})
            self.paddleocr_enabled = paddleocr_config.get('enabled', True)
            self.paddleocr_config = {
                'use_angle_cls': paddleocr_config.get('use_angle_cls', True),
                'lang': paddleocr_config.get('lang', 'ch'),
                'confidence_threshold': paddleocr_config.get('confidence_threshold', 0.7)
            }
            
            # EasyOCR配置
            easyocr_config = ocr_config.get('easyocr', {})
            self.easyocr_enabled = easyocr_config.get('enabled', True)
            self.easyocr_config = {
                'languages': easyocr_config.get('languages', ['ch_sim', 'en']),
                'gpu': easyocr_config.get('gpu', False),
                'confidence_threshold': easyocr_config.get('confidence_threshold', 0.7)
            }
        
        # 初始化 YOLO 模型
        model_path = 'models/best.pt'
        logger.info(f"Loading YOLO model from {model_path}")
        self.model = YOLO(model_path, task='detect')
        
        # 初始化 PaddleOCR（如果启用）
        self.ocr = None
        if self.paddleocr_enabled:
            try:
                self.ocr = PaddleOCR(
                    use_angle_cls=self.paddleocr_config['use_angle_cls'], 
                    lang=self.paddleocr_config['lang']
                )
                logger.info(f"PaddleOCR initialized with config: {self.paddleocr_config}")
            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
                self.paddleocr_enabled = False
        
        # 初始化 EasyOCR（如果启用）
        self.easy_ocr = None
        if self.easyocr_enabled:
            try:
                # 设置EasyOCR模型路径
                model_storage_directory = os.path.expanduser('models')
                os.makedirs(model_storage_directory, exist_ok=True)
                
                self.easy_ocr = easyocr.Reader(
                    self.easyocr_config['languages'], 
                    gpu=self.easyocr_config['gpu'], 
                    model_storage_directory=model_storage_directory, 
                    download_enabled=False
                )
                logger.info(f"EasyOCR initialized with config: {self.easyocr_config}")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {str(e)}")
                self.easyocr_enabled = False
        
        # 检查是否至少有一个OCR引擎可用
        if not self.paddleocr_enabled and not self.easyocr_enabled:
            logger.error("No OCR engine available! Please check configuration.")
        
        self.entry_record_file = 'data/entry_records.json'
        self.exit_record_file = 'data/exit_records.json'
        self.abnormal_record_file = 'data/abnormal_vehicles.json'
        self.current_vehicles = {}
        
        # 中国车牌汉字列表
        self.valid_chinese_chars = {
            '京', '津', '冀', '晋', '蒙', '辽', '吉', '黑', '沪', '苏',
            '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤', '桂',
            '琼', '渝', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
            '新', '港', '澳', '台'
        }
        
        # 确保数据目录存在
        os.makedirs('data', exist_ok=True)
        
        # 确保文件存在
        for record_file in [self.entry_record_file, self.exit_record_file, self.abnormal_record_file]:
            if not os.path.exists(record_file):
                with open(record_file, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False, indent=4)
                logger.info(f"Created new vehicle records file: {record_file}")

    def validate_chinese_char(self, plate_number):
        """
        验证车牌中的汉字是否为有效的中国车牌汉字
        :param plate_number: 车牌号
        :return: bool
        """
        if not plate_number:
            return False
            
        # 提取第一个字符（汉字）
        first_char = plate_number[0]
        
        # 检查是否为有效的中国车牌汉字
        if first_char in self.valid_chinese_chars:
            logger.info(f"Valid Chinese character in plate: {first_char}")
            return True
        else:
            logger.warning(f"Invalid Chinese character in plate: {first_char}")
            return False

    def record_plate_entry(self, plate_number, time_str, plate_color):
        """
        记录车辆进场信息
        :param plate_number: 车牌号
        :param time_str: 时间字符串 (HH:MM:SS)
        :param plate_color: 车牌颜色
        """
        # 获取当前日期
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 创建新记录
        new_record = {
            'plate_number': plate_number,
            'date': current_date,
            'time': time_str,
            'timestamp': current_timestamp,
            'plate_color': plate_color,
            'status': 'entry'
        }

        try:
            # 读取现有记录
            with open(self.entry_record_file, 'r', encoding='utf-8') as f:
                records = json.load(f)

            # 添加新记录
            records.append(new_record)

            # 写入更新后的记录
            with open(self.entry_record_file, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=4)

            # 更新在场车辆
            self.current_vehicles[plate_number] = {
                'entry_time': current_timestamp,
                'plate_color': plate_color
            }

        except Exception as e:
            print(f"Error recording vehicle entry: {e}")
            # 如果文件损坏或出错，重新创建文件
            with open(self.entry_record_file, 'w', encoding='utf-8') as f:
                json.dump([new_record], f, ensure_ascii=False, indent=4)

    def record_plate_exit(self, plate_number, time_str):
        """
        记录车辆出场信息
        :param plate_number: 车牌号
        :param time_str: 时间字符串 (HH:MM:SS)
        :return: bool 是否成功记录
        """
        try:
            # 获取当前日期和时间戳
            current_date = datetime.now().strftime('%Y-%m-%d')
            current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 创建新记录
            new_record = {
                'plate_number': plate_number,
                'date': current_date,
                'time': time_str,
                'timestamp': current_timestamp,
                'plate_color': self.current_vehicles[plate_number]['plate_color'],
                'status': 'exit',
                'entry_time': self.current_vehicles[plate_number]['entry_time']
            }
            
            # 读取现有记录
            with open(self.exit_record_file, 'r', encoding='utf-8') as f:
                records = json.load(f)
            
            # 添加新记录
            records.append(new_record)
            
            # 写入更新后的记录
            with open(self.exit_record_file, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=4)
            
            # 从在场车辆中移除
            if plate_number in self.current_vehicles:
                del self.current_vehicles[plate_number]
                logger.info(f"Successfully recorded exit for plate: {plate_number}")
                return True
            else:
                logger.warning(f"Plate {plate_number} not found in current vehicles")
                return False
                
        except Exception as e:
            logger.error(f"Error recording vehicle exit: {str(e)}", exc_info=True)
            return False

    def get_current_vehicle_count(self):
        """获取当前在场车辆数量"""
        return len(self.current_vehicles)

    def get_current_vehicles(self):
        """获取当前在场车辆信息"""
        return self.current_vehicles

    # 使用ocr进行车牌识别
    def get_license_result(self, image):
        """
        使用多种OCR引擎进行车牌识别
        :param image: 输入的车牌截取照片
        :return: (车牌号, 置信度)
        """
        def try_paddle_ocr(img):
            """尝试使用PaddleOCR识别"""
            if not self.paddleocr_enabled or self.ocr is None:
                logger.warning("PaddleOCR is disabled or not initialized")
                return None, None

            # 基本检查
            if img is None or len(img.shape) < 2:
                logger.error("PaddleOCR输入图像无效")
                return None, None
            # 图像格式转换
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                
            logger.info(f"PaddleOCR开始识别，图像尺寸: {img.shape}")
            # 最多尝试3次
            for attempt in range(3):
                try:
                    result = self.ocr.ocr(img, cls=True)
                    # 检查识别结果
                    if not result or not result[0]:
                        logger.warning(f"PaddleOCR第{attempt + 1}次尝试未检测到文字")
                        continue
                    # 提取车牌号和置信度
                    license_name, conf = result[0][0][1]
                    license_name = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', license_name)
                    if license_name:
                        logger.info(f"PaddleOCR识别结果: {license_name}, 置信度: {conf}")
                        return license_name, conf 
                    logger.warning(f"PaddleOCR第{attempt + 1}次尝试识别结果为空")
                except Exception as e:
                    logger.error(f"PaddleOCR第{attempt + 1}次尝试失败: {str(e)}")
            return None, None

        def try_easy_ocr(img):
            """尝试使用EasyOCR识别"""
            if not self.easyocr_enabled or self.easy_ocr is None:
                logger.warning("EasyOCR is disabled or not initialized")
                return None, None
                
            try:
                if img is None:
                    logger.error("EasyOCR输入图像为空")
                    return None, None
                    
                # 检查图像尺寸
                height, width = img.shape[:2]
                if width < 20 or height < 10:
                    logger.error(f"EasyOCR图像尺寸过小: {width}x{height}")
                    return None, None
                
                logger.info(f"EasyOCR开始识别，图像尺寸: {img.shape}")
                results = self.easy_ocr.readtext(img)
                
                if not results:
                    logger.warning("EasyOCR返回空结果")
                    return None, None
                    
                # 获取置信度最高的结果
                best_result = max(results, key=lambda x: x[2])
                license_name = best_result[1]
                conf = best_result[2]
                
                if not license_name:
                    logger.warning("EasyOCR识别结果为空")
                    return None, None
                    
                # 清理车牌号
                license_name = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', license_name)
                
                logger.info(f"EasyOCR识别结果: {license_name}, 置信度: {conf}")
                return license_name, conf
                
            except Exception as e:
                logger.error(f"EasyOCR识别失败: {str(e)}", exc_info=True)
                return None, None

        if image is None:
            logger.error("输入图像为空")
            return None, None

        # 尝试PaddleOCR
        license_name, conf = try_paddle_ocr(image)
        if license_name and conf:
            return license_name, conf
            
        # 如果PaddleOCR失败，尝试EasyOCR
        license_name, conf = try_easy_ocr(image)
        if license_name and conf:
            return license_name, conf
            
        logger.warning("所有识别方法均失败")
        return None, None

    def get_license_img(self, location_list, plate_color, now_img):
        """
        识别车牌
        :param location_list: 车牌位置信息
        :param plate_color: 车牌颜色
        :param now_img: 当前图像
        :return: 车牌号与置信度
        """
        if len(location_list) >= 1:
            location_list = [list(map(int, e)) for e in location_list]
            license_imgs = []
            for each in location_list:
                x1, y1, x2, y2 = each
                # 添加边界检查
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(now_img.shape[1], x2)
                y2 = min(now_img.shape[0], y2)
                cropImg = now_img[y1:y2, x1:x2]
                # 图像预处理
                try:
                    # 确保图像是3通道的
                    if len(cropImg.shape) == 2:
                        cropImg = cv2.cvtColor(cropImg, cv2.COLOR_GRAY2BGR)
                    license_imgs.append(cropImg)
                    logger.info(f"图像预处理完成，确保图像通道的正确性: {cropImg.shape}")
                    
                except Exception as e:
                    logger.error(f"图像预处理失败: {str(e)}")
                    # 如果预处理失败，使用原始图像
                    license_imgs.append(cropImg)
        
        lisence_res = []
        conf_list = []
        for each in license_imgs:
            license_num, conf = self.get_license_result(each)
            
            # 使用配置的置信度阈值
            paddleocr_threshold = self.paddleocr_config.get('confidence_threshold', 0.7)
            easyocr_threshold = self.easyocr_config.get('confidence_threshold', 0.7)
            # 使用较高的阈值作为最终阈值
            confidence_threshold = max(paddleocr_threshold, easyocr_threshold)
            
            # 添加置信度阈值和车牌格式验证
            if license_num and conf and conf > confidence_threshold:

                # 验证车牌格式
                if self._validate_plate_format(license_num, plate_color):
                    logger.info("识别到的车牌:{} {}".format(plate_color, license_num))
                    lisence_res.append(license_num)
                    conf_list.append(conf)
                else:
                    logger.warning("无法识别的车牌")
                    lisence_res.append('无法识别')
                    conf_list.append(0)
            else:
                lisence_res.append('无法识别')
                conf_list.append(0)
        return lisence_res, conf_list

    def _validate_plate_format(self, plate_number, plate_color):
        """
        验证车牌格式
        :param plate_number: 车牌号
        :param plate_color: 车牌颜色
        :return: bool
        """
        if plate_color == 'GreenLicense':
            # 新能源车牌格式：1位汉字 + 1位字母 + 6位字母或数字
            pattern = r'^[\u4e00-\u9fa5][A-Z][A-Z0-9]{6}$'
        elif plate_color == 'BlueLicense':
            # 普通车牌格式：1位汉字 + 1位字母 + 5位字母或数字
            pattern = r'^[\u4e00-\u9fa5][A-Z][A-Z0-9]{5}$'
        else:
            return False
        return bool(re.match(pattern, plate_number))
    
    def get_vehicle_information(self, results, frame):
        """
        从检测结果中获取车辆信息
        :param results: YOLO检测结果列表
        :param frame: 原始图像
        :return: (车牌号, 置信度, 车牌颜色)
        """
        if not results or len(results) == 0:
            return None, None, None
        
        # 获取第一个检测结果
        result = results[0]
        if not hasattr(result, 'boxes') or len(result.boxes) == 0:
            logger.info("没有检测到目标")
            return None, None, None
        
        # 处理检测结果
        location_list = result.boxes.xyxy.tolist()
        if not location_list:  # 如果没有检测到目标
            logger.info("没有检测到目标")
            return None, None, None
        
        # 车牌颜色
        try:
            plate_color = self.names[int(result.boxes.cls[0])]
        except (IndexError, ValueError):
            logger.info("车牌颜色识别错误")
            return None, None, None
        
        # 车牌识别
        lisence_res, conf_list = self.get_license_img(location_list, plate_color, frame)
        if not lisence_res or not conf_list:  # 如果没有识别到车牌
            return None, None, None
        
        # 车牌号
        plate_number = lisence_res[0]
        # 置信度
        conf_score = conf_list[0]

        return plate_number, conf_score, plate_color

    def test_license_recognition(self, image_path):
        """
        测试车牌识别功能
        :param image_path: 测试图片路径
        :return: None
        """
        try:
            # 读取测试图片
            if not os.path.exists(image_path):
                logger.error(f"测试图片不存在: {image_path}")
                return
                
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"无法读取图片: {image_path}")
                return
                
            logger.info(f"开始测试图片: {image_path}")
            logger.info(f"图片尺寸: {image.shape}")
            
            # 使用YOLO模型检测车牌位置
            results = self.model(image)
            if not results or len(results) == 0:
                logger.warning("未检测到车牌")
                return
                
            # 获取检测结果
            result = results[0]
            if not hasattr(result, 'boxes') or len(result.boxes) == 0:
                logger.warning("未检测到车牌")
                return
                
            # 获取车牌位置和颜色
            location_list = result.boxes.xyxy.tolist()
            plate_color = self.names[int(result.boxes.cls[0])]
            
            logger.info(f"检测到车牌颜色: {plate_color}")
            logger.info(f"车牌位置: {location_list}")
            
            # 裁剪车牌区域并识别
            for i, location in enumerate(location_list):
                x1, y1, x2, y2 = map(int, location)
                # 添加边界检查
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)
                
                # 裁剪车牌区域
                crop_img = image[y1:y2, x1:x2]
                if crop_img.size == 0:
                    logger.warning("裁剪的车牌区域为空")
                    continue
                    
                logger.info(f"开始识别第 {i+1} 个车牌")
                
                # 使用PaddleOCR识别
                license_name, conf = self.get_license_result(crop_img)
                if license_name and conf:
                    logger.info(f"PaddleOCR识别结果: {license_name}, 置信度: {conf}")
                else:
                    logger.warning("PaddleOCR识别失败")
                
                # 使用EasyOCR识别
                license_name, conf = self.get_license_result(crop_img)
                if license_name and conf:
                    logger.info(f"EasyOCR识别结果: {license_name}, 置信度: {conf}")
                else:
                    logger.warning("EasyOCR识别失败")
                
                # 验证车牌格式
                if license_name:
                    if self._validate_plate_format(license_name, plate_color):
                        logger.info(f"车牌格式验证通过: {license_name}")
                    else:
                        logger.warning(f"车牌格式验证失败: {license_name}")
            
        except Exception as e:
            logger.error(f"测试过程中发生错误: {str(e)}", exc_info=True)

# if __name__ == '__main__':
#     # 创建测试实例
#     vehicle_info = VehicleInformation()
    
#     # 测试图片路径
#     test_images = [
#         'test_images/picture.jpg',  # 普通车牌
#         'test_images/picture2.jpg',  # 新能源车牌
#         'test_images/picture3.jpg'
#     ]
    
#     # 运行测试
#     for image_path in test_images:
#         vehicle_info.test_license_recognition(image_path)
#         print("\n" + "="*50 + "\n")  # 分隔不同图片的测试结果
