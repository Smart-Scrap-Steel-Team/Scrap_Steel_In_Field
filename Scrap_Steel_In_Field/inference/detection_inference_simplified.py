import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import cv2
from datetime import datetime
import os
import sys
# 不再导入相机模块
from config.detection_config import MODEL_CONFIG, SAVE_CONFIG, LOG_CONFIG
from utils.logger import setup_logger

# 添加父目录到系统路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 创建日志记录器
logger = setup_logger('detection', LOG_CONFIG['base_dir'], LOG_CONFIG['sub_dirs']['detection'])

# CUDA相关全局变量
cuda_initialized = False
device = None
context = None


def initialize_cuda():
    """初始化CUDA环境，用于GPU加速"""
    global cuda_initialized, device, context

    if not cuda_initialized:
        try:
            cuda.init()
            device = cuda.Device(0)
            context = device.make_context()
            cuda_initialized = True
            logger.info("CUDA 初始化成功")
        except Exception as e:
            logger.error(f"CUDA 初始化失败: {e}")
            raise


def release_cuda():
    """释放全局CUDA资源"""
    global cuda_initialized, context, device

    if not cuda_initialized:
        logger.info("CUDA 未初始化，无需释放")
        return

    if context is None:
        logger.warning("CUDA 上下文已为空")
        return

    try:
        context.detach()
        context = None
        device = None
        cuda_initialized = False
        logger.info("全局 CUDA 资源已释放")
    except Exception as e:
        logger.error(f"释放全局 CUDA 资源失败: {e}")
        raise


class TRTInferenceEngine:
    """TensorRT推理引擎类，用于加载和运行模型"""

    def __init__(self, engine_path, categories, conf_threshold, nms_threshold):
        self.categories = categories
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.engine = None
        self.context = None
        self.input_name = None
        self.output_name = None
        self.engine, self.context, self.input_name, self.output_name = self.load_engine(engine_path)
        if self.engine is None or self.context is None:
            raise RuntimeError("Failed to load TensorRT engine.")

    def __del__(self):
        """析构函数，确保资源被正确释放"""
        try:
            self.release()
        except Exception as e:
            logger.error(f"析构时释放资源失败: {e}")

    def release(self):
        """释放引擎实例资源"""
        if hasattr(self, 'context') and self.context:
            try:
                self.context = None
                logger.info("引擎上下文已释放")
            except Exception as e:
                logger.error(f"释放引擎上下文失败: {e}")

        if hasattr(self, 'engine') and self.engine:
            try:
                self.engine = None
                logger.info("TensorRT 引擎已释放")
            except Exception as e:
                logger.error(f"释放 TensorRT 引擎失败: {e}")

    def load_engine(self, engine_path):
        """加载TensorRT引擎"""
        if not os.path.exists(engine_path):
            logger.error(f"引擎文件未找到: {engine_path}")
            return None, None, None, None

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        try:
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            with trt.Runtime(TRT_LOGGER) as runtime:
                engine = runtime.deserialize_cuda_engine(engine_data)
                if engine is None:
                    logger.error("引擎反序列化失败")
                    return None, None, None, None
                context = engine.create_execution_context()
                if context is None:
                    logger.error("创建执行上下文失败")
                    return None, None, None, None
                input_name = engine.get_tensor_name(0)
                output_name = engine.get_tensor_name(1)
                logger.info(f"引擎加载成功: {os.path.basename(engine_path)}")
                return engine, context, input_name, output_name
        except Exception as e:
            logger.error(f"加载引擎出错: {e}")
            return None, None, None, None

    def infer(self, input_tensor, ratio_pad, orig_shape):
        """执行推理"""
        if self.engine is None or self.context is None:
            logger.error("引擎或上下文未初始化")
            return []

        global context
        if context:
            context.push()

        try:
            stream = cuda.Stream()
            input_shape = self.context.get_tensor_shape(self.input_name)
            output_shape = self.context.get_tensor_shape(self.output_name)

            if tuple(input_tensor.shape) != tuple(input_shape):
                raise ValueError(f"输入形状 {input_tensor.shape} 与引擎期望 {input_shape} 不匹配")

            d_input = cuda.mem_alloc(input_tensor.nbytes)
            d_output = cuda.mem_alloc(trt.volume(output_shape) * np.float32().itemsize)
            host_output = np.empty(output_shape, dtype=np.float32)

            try:
                cuda.memcpy_htod_async(d_input, input_tensor, stream)
                self.context.set_tensor_address(self.input_name, d_input)
                self.context.set_tensor_address(self.output_name, d_output)

                self.context.execute_async_v3(stream.handle)

                cuda.memcpy_dtoh_async(host_output, d_output, stream)
                stream.synchronize()

                detections = self.process_output([host_output], ratio_pad, orig_shape)
                results = [{
                    "type": "detection",
                    "label": self.categories[int(cls_id)],
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(conf),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                } for (x1, y1, x2, y2, conf, cls_id) in detections]

                return results
            finally:
                stream.synchronize()
                d_input.free()
                d_output.free()
        finally:
            if context:
                context.pop()

    def process_output(self, outputs, ratio_pad, orig_shape):
        """处理模型输出，转换为检测框"""
        try:
            predictions = outputs[0].squeeze(0)
            xc, yc, w, h = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
            obj_conf = predictions[:, 4]
            cls_conf = predictions[:, 5:].max(axis=1)
            cls_ids = predictions[:, 5:].argmax(axis=1)
            boxes = np.column_stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
            scores = obj_conf * cls_conf
            valid_mask = (scores > self.conf_threshold) & (cls_ids < len(self.categories))
            if not valid_mask.any():
                return []
            boxes = boxes[valid_mask]
            scores = scores[valid_mask]
            cls_ids = cls_ids[valid_mask]
            scale, pad_x, pad_y = ratio_pad
            boxes -= [pad_x, pad_y, pad_x, pad_y]
            boxes /= scale
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_shape[1])
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_shape[0])
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_threshold, self.nms_threshold)
            if indices is not None and len(indices) > 0:
                return [(*map(int, boxes[i][:4]), float(scores[i]), int(cls_ids[i])) for i in indices.flatten()]
            return []
        except Exception as e:
            logger.error(f"处理输出出错: {e}")
            return []


def visualize_detections(image, detections, output_path=None):
    """可视化检测结果，绘制检测框和标签

    Args:
        image: 输入图像，RGB格式
        detections: 检测结果列表，每个元素包含label、bbox和confidence
        output_path: 输出图像保存路径，如果为None则显示图像

    Returns:
        numpy.ndarray: 绘制了检测框的图像
    """
    colors = {
        'GasCyl1': (0, 0, 255),  # 红色
        'GasCyl2': (0, 128, 255),  # 橙色
        'FireExt1': (0, 255, 0),  # 绿色
        'FireExt2': (255, 0, 0)  # 蓝色
    }

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for det in detections:
        label = det["label"]
        bbox = det["bbox"]
        conf = det["confidence"]

        color = colors.get(label, (0, 255, 255))

        cv2.rectangle(img_bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        text = f"{label}: {conf:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

        cv2.rectangle(img_bgr, (bbox[0], bbox[1] - text_height - 5),
                      (bbox[0] + text_width, bbox[1]), color, -1)

        cv2.putText(img_bgr, text, (bbox[0], bbox[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if output_path:
        cv2.imwrite(output_path, img_bgr)
    else:
        cv2.imshow("检测结果", img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img_bgr


def process_detection(engine, image, model_input_size):
    """处理图像检测，包括预处理、推理和后处理

    Args:
        engine: TensorRT推理引擎实例
        image: 输入图像
        model_input_size: 模型输入尺寸

    Returns:
        tuple: (检测结果列表, 原始图像)
    """
    input_tensor, ratio_pad, orig_shape, original_image = preprocess_image(image, model_input_size)
    logger.info(f"图像预处理完成，原始尺寸: {orig_shape}")

    logger.info("开始执行目标检测...")
    detections = engine.infer(input_tensor, ratio_pad, orig_shape)
    logger.info(f"检测完成，找到 {len(detections)} 个物体")

    for i, det in enumerate(detections):
        logger.info(f"物体 {i + 1}: {det['label']} (置信度: {det['confidence']:.2f})")

    return detections, original_image


def save_and_visualize_results(original_image, detections, detection_dir, categories, car_bucket_corners=None):
    """保存并可视化检测结果

    Args:
        original_image: 原始图像
        detections: 检测结果列表
        detection_dir: 检测结果保存目录
        categories: 类别列表
        car_bucket_corners: 车兜区域的 4 个角点 (可选)

    Returns:
        int: 检测到的物体数量
    """
    try:
        # 确保目录存在
        os.makedirs(detection_dir, exist_ok=True)

        # 生成时间戳和文件路径
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 使用绝对路径
        detection_dir = os.path.abspath(detection_dir)
        detection_path = os.path.join(detection_dir, f"detection_{timestamp}.jpg")
        logger.info(f"准备保存检测结果到: {detection_path}")

        # 检查图像是否有效
        if original_image is None:
            logger.error("原始图像为空")
            return 0

        # 打印图像信息
        logger.info(f"图像形状: {original_image.shape}")
        logger.info(f"图像数据类型: {original_image.dtype}")
        logger.info(f"图像最大值: {np.max(original_image)}")
        logger.info(f"图像最小值: {np.min(original_image)}")

        # 定义不同类别的颜色
        colors = {
            'GasCyl1': (0, 0, 255),  # 红色
            'GasCyl2': (0, 128, 255),  # 橙色
            'FireExt1': (0, 255, 0),  # 绿色
            'FireExt2': (255, 0, 0)  # 蓝色
        }

        # 转换图像格式
        img_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

        # 绘制检测框和标签
        for det in detections:
            label = det["label"]
            bbox = det["bbox"]
            conf = det["confidence"]

            color = colors.get(label, (0, 255, 255))

            # 绘制边界框
            cv2.rectangle(img_bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # 准备标签文本
            text = f"{label}: {conf:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # 绘制标签背景
            cv2.rectangle(img_bgr, (bbox[0], bbox[1] - text_height - 5),
                          (bbox[0] + text_width, bbox[1]), color, -1)

            # 绘制标签文本
            cv2.putText(img_bgr, text, (bbox[0], bbox[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 绘制车兜区域
        if car_bucket_corners is not None:
            corners = car_bucket_corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_bgr, [corners], True, (0, 255, 255), 3)
            for i, pt in enumerate(corners.reshape(-1, 2)):
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(img_bgr, (x, y), 15, (0, 0, 255), -1)
                cv2.putText(img_bgr, str(i + 1), (x - 15, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # 保存检测结果图像
        try:
            # 检查保存路径是否可写
            if not os.access(os.path.dirname(detection_path), os.W_OK):
                logger.error(f"目录没有写入权限: {os.path.dirname(detection_path)}")
                return 0

            logger.info(f"开始保存检测结果图像: 图像尺寸={img_bgr.shape}, 类型={img_bgr.dtype}, 路径={detection_path}")

            success = cv2.imwrite(detection_path, img_bgr)
            if success:
                logger.info(f"成功保存检测结果到: {detection_path}")
                # 检查文件是否实际存在
                if os.path.exists(detection_path):
                    file_size = os.path.getsize(detection_path)
                    logger.info(f"保存的检测结果文件大小: {file_size} 字节")
                else:
                    logger.warning(f"imwrite返回成功，但文件不存在: {detection_path}")
            else:
                logger.error(f"保存检测结果失败: {detection_path}, OpenCV imwrite返回False")
                return 0
        except Exception as e:
            logger.error(f"保存检测结果时出错: {e}")
            import traceback
            logger.error(f"保存检测结果异常详情: {traceback.format_exc()}")
            return 0

        return len(detections)
    except Exception as e:
        logger.error(f"保存结果过程中出错: {e}")
        return 0


def preprocess_image(image, model_input_size):
    """图像预处理，包括缩放、填充等

    Args:
        image: 输入图像（可以是文件路径或numpy数组）
        model_input_size: 模型输入尺寸

    Returns:
        tuple: (预处理后的张量, 缩放和填充信息, 原始图像尺寸, 原始图像)
    """
    try:
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise FileNotFoundError(f"图像文件未找到: {image}")
        else:
            img = image.copy()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        scale = min(model_input_size / width, model_input_size / height)
        new_size = (int(width * scale), int(height * scale))
        img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)
        canvas = np.full((model_input_size, model_input_size, 3), 114, dtype=np.uint8)
        pad_x = (model_input_size - new_size[0]) // 2
        pad_y = (model_input_size - new_size[1]) // 2
        canvas[pad_y:pad_y + new_size[1], pad_x:pad_x + new_size[0]] = img_resized

        input_tensor = np.ascontiguousarray((canvas / 255.0).astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...])
        return input_tensor, (scale, pad_x, pad_y), (height, width), img
    except Exception as e:
        logger.error(f"图像预处理出错: {e}")
        raise e


class DetectionManager:
    """检测管理器，封装检测相关的所有功能"""

    def __init__(self, config=None):
        """初始化检测管理器

        Args:
            config: 配置字典，如果为None则使用默认配置
        """
        if config is None:
            config = MODEL_CONFIG

        self.model_path = config["model_path"]
        self.categories = config["categories"]
        self.conf_threshold = config["conf_threshold"]
        self.nms_threshold = config["nms_threshold"]
        self.model_input_size = config["model_input_size"]
        self.engine = None
        self.detection_dir = None

    def initialize(self, base_dir):
        """初始化检测系统和保存目录

        Args:
            base_dir: 基础保存目录

        Returns:
            bool: 初始化是否成功
        """
        try:
            # 初始化CUDA
            initialize_cuda()

            # 初始化TensorRT引擎
            self.engine = TRTInferenceEngine(
                engine_path=self.model_path,
                categories=self.categories,
                conf_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold
            )

            # 设置保存目录
            self.detection_dir = os.path.join(base_dir, SAVE_CONFIG["sub_dirs"]["dangerous_det"])
            os.makedirs(self.detection_dir, exist_ok=True)

            logger.info("检测系统初始化成功")
            return True
        except Exception as e:
            logger.error(f"检测系统初始化失败: {e}")
            return False

    def detect_from_image(self, image_path):
        """从文件读取图像并进行检测

        Args:
            image_path: 图像文件路径

        Returns:
            tuple: (检测结果列表, 原始图像)
        """
        if not self.engine:
            logger.error("检测系统未初始化")
            return None, None

        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"无法读取图像: {image_path}")
                return None, None

            # 处理检测
            detections, original_image = process_detection(
                self.engine, image, self.model_input_size)

            return detections, original_image
        except Exception as e:
            logger.error(f"检测过程出错: {e}")
            return None, None

    def save_results(self, detections, original_image, car_bucket_corners=None):
        """保存检测结果

        Args:
            detections: 检测结果列表
            original_image: 原始图像
            car_bucket_corners: 车兜区域的 4 个角点 (可选)

        Returns:
            int: 检测到的物体数量
        """
        if not self.detection_dir:
            logger.error("保存目录未设置")
            return 0

        return save_and_visualize_results(
            original_image, detections, self.detection_dir, self.categories, car_bucket_corners)

    def release(self):
        """释放资源"""
        if self.engine:
            self.engine.release()
            self.engine = None

        release_cuda()
        logger.info("检测系统资源已释放")


def detect_car_bucket(frame):
    """
    检测黄色车兜区域（与 test_2_detection 相同的逻辑）
    Args:
        frame: 输入图像（注意：本模块中 original_image 是 RGB 格式）
    Returns:
        numpy.ndarray: 车兜区域的 4 个角点 (4, 2)，如果未检测到则返回 None
    """
    try:
        # original_image 是 RGB 格式，需先转回 BGR 再做 HSV 转换
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 80, 80])
        upper_yellow = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        max_contour = max(contours, key=cv2.contourArea)
        min_area = 10000
        if cv2.contourArea(max_contour) < min_area:
            return None

        peri = cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, 0.02 * peri, True)

        if len(approx) != 4:
            return None

        corners = approx.reshape(4, 2).astype(np.float32)
        return corners
    except Exception as e:
        logger.warning(f"车兜区域检测出错: {e}")
        return None


def filter_detections_in_car_bucket(detections, car_bucket_corners):
    """
    过滤检测框，只保留车兜区域内的检测目标
    Args:
        detections: 检测结果列表
        car_bucket_corners: 车兜区域的 4 个角点 (4, 2)
    Returns:
        list: 过滤后的检测结果
    """
    if car_bucket_corners is None or len(car_bucket_corners) != 4:
        return detections

    filtered = []
    polygon = car_bucket_corners.astype(np.int32).reshape((-1, 1, 2))

    for det in detections:
        bbox = det.get("bbox")
        if bbox is None or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        distance = cv2.pointPolygonTest(polygon, (cx, cy), False)
        if distance >= 0:
            filtered.append(det)

    if len(filtered) < len(detections):
        logger.info(f"车兜区域过滤: {len(detections)} -> {len(filtered)} 个目标")

    return filtered


def simple_detect(image_path, config=None):
    """简单的检测函数，可以直接调用

    Args:
        image_path: 图像路径
        config: 配置信息，如果为None则使用默认配置

    Returns:
        tuple: (检测结果列表, 是否成功)
    """
    # 图像路径必须提供
    if image_path is None:
        logger.error("必须提供图像路径")
        return None, False

    # 使用默认配置或传入的配置
    if config is None:
        config = {
            "model": MODEL_CONFIG,
            "save": SAVE_CONFIG
        }

    # 创建检测管理器
    detector = DetectionManager(config=config["model"])

    try:
        # 初始化
        if not detector.initialize(config["save"]["base_dir"]):
            return None, False

        # 从文件读取图像并检测
        detections, original_image = detector.detect_from_image(image_path)

        # 检测车兜区域并过滤结果
        car_bucket_corners = None
        if original_image is not None:
            car_bucket_corners = detect_car_bucket(original_image)
            if car_bucket_corners is not None:
                logger.info("检测到车兜区域，将过滤区域外的目标")
                if detections:
                    detections = filter_detections_in_car_bucket(detections, car_bucket_corners)

        if detections is not None:
            # 获取检测结果图片路径
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # 使用配置中的路径
            detection_path = os.path.join(
                config["save"]["base_dir"],
                config["save"]["sub_dirs"]["dangerous_det"],
                f"detection_{timestamp}.jpg"
            )
            # 确保目录存在
            os.makedirs(os.path.dirname(detection_path), exist_ok=True)

            # 如果没有检测到危险品，直接保存原始图像
            if not detections:
                # 保存原始图像（带车兜标注）
                img_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
                if car_bucket_corners is not None:
                    corners = car_bucket_corners.astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img_bgr, [corners], True, (0, 255, 255), 3)
                    for i, pt in enumerate(corners.reshape(-1, 2)):
                        x, y = int(pt[0]), int(pt[1])
                        cv2.circle(img_bgr, (x, y), 15, (0, 0, 255), -1)
                        cv2.putText(img_bgr, str(i + 1), (x - 15, y - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.imwrite(detection_path, img_bgr)
                detections = [{
                    "type": "detection",
                    "label": "NoDangerous",
                    "bbox": [0, 0, 0, 0],
                    "confidence": 0.0,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "result_path": image_path
                }]

            else:
                # 保存带检测框的结果（带车兜标注）
                detector.save_results(detections, original_image, car_bucket_corners)

            # 将检测结果路径添加到每个检测结果中
            for det in detections:
                det['result_path'] = detection_path

            return detections, True
        else:
            return [], True

    except Exception as e:
        logger.error(f"检测过程出错: {e}")
        return None, False
    finally:
        detector.release()


def main():
    """主函数，用于直接运行此脚本时的测试"""
    # 直接指定图像路径
    image_path = "test_image.jpg"  # 这里替换为您的图像路径

    # 检查图像文件是否存在
    if not os.path.exists(image_path):
        print(f"错误：图像文件不存在: {image_path}")
        return

    # 执行检测
    detections, success = simple_detect(image_path)
    if success:
        if detections:
            print(f"检测到 {len(detections)} 个危险品：")
            for det in detections:
                print(f"- {det['label']}: 置信度 {det['confidence']:.2f}")
        else:
            print("未检测到危险品")
    else:
        print("检测失败")


if __name__ == "__main__":
    main()