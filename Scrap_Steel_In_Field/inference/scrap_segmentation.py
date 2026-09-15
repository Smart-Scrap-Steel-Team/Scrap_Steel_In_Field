import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import cv2
from datetime import datetime
import os
import sys
import logging
import json
from config.segment_config import MODEL_CONFIG, SAVE_CONFIG, LOG_CONFIG
from utils.logger import setup_logger

# 添加父目录到系统路径，确保可以导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 创建日志记录器
logger = setup_logger('segmentation', LOG_CONFIG['base_dir'], LOG_CONFIG['sub_dirs']['segmentation'])

# CUDA相关全局变量
cuda_initialized = False  # CUDA初始化状态标志
device = None  # CUDA设备对象
context = None  # CUDA上下文对象


def initialize_cuda():
    """
    初始化CUDA环境，用于GPU加速
    设置全局CUDA设备、上下文和初始化状态
    """
    global cuda_initialized, device, context

    if not cuda_initialized:
        try:
            cuda.init()  # 初始化CUDA驱动
            device = cuda.Device(0)  # 获取第一个CUDA设备
            context = device.make_context()  # 创建CUDA上下文
            cuda_initialized = True
            logger.info("CUDA 初始化成功")
        except Exception as e:
            logger.error(f"CUDA 初始化失败: {e}")
            raise


def release_cuda():
    """
    释放全局CUDA资源
    清理CUDA上下文和设备对象
    """
    global cuda_initialized, context, device

    if not cuda_initialized:
        logger.info("CUDA 未初始化，无需释放")
        return

    if context is None:
        logger.warning("CUDA 上下文已为空")
        return

    try:
        context.detach()  # 分离CUDA上下文
        context = None
        device = None
        cuda_initialized = False
        logger.info("全局 CUDA 资源已释放")
    except Exception as e:
        logger.error(f"释放全局 CUDA 资源失败: {e}")
        raise


class TRTInferenceEngine:
    """
    TensorRT推理引擎类，用于加载和运行模型
    负责模型的加载、推理执行和资源管理
    """

    def __init__(self, engine_path, categories, conf_threshold, nms_threshold):
        """
        初始化推理引擎
        Args:
            engine_path: TensorRT引擎文件路径
            categories: 类别列表
            conf_threshold: 置信度阈值
            nms_threshold: 非极大值抑制阈值
        """
        self.categories = categories
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.engine = None
        self.context = None
        self.input_name = None
        self.output_detect_name = None
        self.output_mask_name = None
        self.engine, self.context, self.input_name, self.output_mask_name,self.output_detect_name = self.load_engine(
            engine_path)
        if self.engine is None or self.context is None:
            raise RuntimeError("Failed to load TensorRT engine.")

    def __del__(self):
        """
        析构函数，确保资源被正确释放
        在对象被销毁时自动调用
        """
        try:
            self.release()
        except Exception as e:
            logger.error(f"析构时释放资源失败: {e}")

    def release(self):
        """
        释放引擎实例资源
        清理TensorRT引擎和上下文
        """
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
        """
        加载TensorRT引擎
        Args:
            engine_path: 引擎文件路径
        Returns:
            tuple: (engine, context, input_name, output_detect_name, output_mask_name)
        """
        if not os.path.exists(engine_path):
            logger.error(f"引擎文件未找到: {engine_path}")
            return None, None, None, None, None

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        try:
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            with trt.Runtime(TRT_LOGGER) as runtime:
                engine = runtime.deserialize_cuda_engine(engine_data)
                if engine is None:
                    logger.error("引擎反序列化失败")
                    return None, None, None, None, None
                context = engine.create_execution_context()
                if context is None:
                    logger.error("创建执行上下文失败")
                    return None, None, None, None, None
                # 通过索引获取输入输出张量名称（兼容旧版本TensorRT）
                input_name = engine.get_tensor_name(0)
                output_detect_name = engine.get_tensor_name(1)
                output_mask_name = engine.get_tensor_name(2)

            logger.info(f"引擎加载成功: {os.path.basename(engine_path)}")
            return engine, context, input_name, output_detect_name, output_mask_name
        except Exception as e:
            logger.error(f"加载引擎出错: {e}")
            return None, None, None, None, None

    def infer(self, input_tensor, ratio_pad, orig_shape):
        """
        执行推理
        Args:
            input_tensor: 输入张量
            ratio_pad: 缩放比例和填充信息
            orig_shape: 原始图像尺寸
        Returns:
            tuple: (detections, mask, class_counts, area_stats)
        """
        if self.engine is None or self.context is None:
            logger.error("引擎或上下文未初始化")
            return [], None, {}, {}
        try:
            stream = cuda.Stream()  # 创建CUDA流
            input_shape = self.context.get_tensor_shape(self.input_name)

            if tuple(input_tensor.shape) != tuple(input_shape):
                raise ValueError(f"输入形状 {input_tensor.shape} 与引擎期望 {input_shape} 不匹配")
            # 分配GPU内存
            d_input = cuda.mem_alloc(input_tensor.nbytes)
            d_detect = cuda.mem_alloc(
                trt.volume(self.context.get_tensor_shape(self.output_detect_name)) * np.float32().itemsize)
            d_mask = cuda.mem_alloc(
                trt.volume(self.context.get_tensor_shape(self.output_mask_name)) * np.float32().itemsize)

            # 创建主机端输出缓冲区
            host_detect = np.empty(self.context.get_tensor_shape(self.output_detect_name), dtype=np.float32)
            host_mask = np.empty(self.context.get_tensor_shape(self.output_mask_name), dtype=np.float32)

            try:
                # 数据传输和推理执行
                cuda.memcpy_htod_async(d_input, input_tensor, stream)
                self.context.set_tensor_address(self.input_name, d_input)
                self.context.set_tensor_address(self.output_detect_name, d_detect)
                self.context.set_tensor_address(self.output_mask_name, d_mask)

                self.context.execute_async_v3(stream.handle)

                # 将结果从GPU复制回主机
                cuda.memcpy_dtoh_async(host_detect, d_detect, stream)
                cuda.memcpy_dtoh_async(host_mask, d_mask, stream)
                stream.synchronize()

                # 处理检测结果和掩码
                detections, class_counts = self.process_detections(host_detect, ratio_pad, orig_shape)
                mask, area_stats = self.process_mask(host_mask, ratio_pad, orig_shape)
                return detections, mask, class_counts, area_stats
            except Exception as e:
                logger.error(f"推理执行失败: {e}")
                return [], None, {}, {}
            finally:
                # 清理GPU内存
                stream.synchronize()
                d_input.free()
                d_detect.free()
                d_mask.free()
        except Exception as e:
            logger.error(f"推理过程发生异常: {e}")
            return [], None, {}, {}

    def process_detections(self, detect_output, ratio_pad, orig_shape):
        """
        处理检测输出
        Args:
            detect_output: 检测输出张量
            ratio_pad: 缩放比例和填充信息
            orig_shape: 原始图像尺寸
        Returns:
            tuple: (detections, class_counts)
        """
        try:
            # 初始化类别计数
            class_counts = {cls_id: 0 for cls_id in range(len(self.categories))}

            if detect_output is None or detect_output.size == 0:
                logger.warning("检测输出为空，返回默认值")
                return [], class_counts

        except Exception as e:
            logger.error(f"处理检测输出初始化失败: {e}")
            return [], class_counts

        # 确保输出维度统一
        class_ids = np.zeros((0,), dtype=int)
        scores = np.zeros((0,))

        # 重塑输出形状并处理
        detect_output = detect_output.reshape(1, 52, 8400).transpose(0, 2, 1)

        # 拆分边界框和类别分数
        box_data = detect_output[..., 0:4]  # [x_center, y_center, width, height]
        obj_scores = detect_output[..., 4]  # 目标置信度
        cls_scores = detect_output[..., 5:5 + len(self.categories)]  # 类别分数

        # 合并分数并获取最大类别分数
        scores = obj_scores[..., None] * cls_scores
        class_ids = np.argmax(scores, axis=-1)
        max_scores = np.max(scores, axis=-1)

        # 转换到检测结果格式 [x1, y1, x2, y2, score, class_id]
        detections = np.concatenate([
            box_data[..., 0:2] - box_data[..., 2:4] / 2,  # x1,y1
            box_data[..., 0:2] + box_data[..., 2:4] / 2,  # x2,y2
            max_scores[..., None],
            class_ids[..., None].astype(float)
        ], axis=-1)

        logger.debug(f"处理后的检测结果形状: {detections.shape}")
        if detections.shape[0] > 0 and detections.shape[1] > 0:
            logger.debug(f"示例检测结果: {detections[0][:3]}")

        # 过滤低置信度检测
        conf_mask = detect_output[..., 4] > self.conf_threshold
        detect_output = detect_output[conf_mask]

        logger.debug(f"处理后检测输出形状: {detect_output.shape}")
        logger.debug(f"有效检测数量: {len(detect_output)}")

        # 如果没有有效检测，则直接返回空结果
        if len(detect_output) == 0:
            logger.debug("检测输出为空")
            return [], class_counts

        boxes = detect_output[:, :4]
        scores = detect_output[:, 4]
        class_ids = detect_output[:, 5].astype(int)

        # 记录检测统计信息
        if len(detect_output) > 0:
            logger.debug(f"最大置信度: {np.max(scores):.4f}, 平均置信度: {np.mean(scores):.4f}")
        else:
            logger.debug("检测输出为空")
            return [], class_counts

        # 转换boxes到原始图像坐标
        r, (dw, dh) = ratio_pad
        h, w = orig_shape
        boxes[:, 0] = (boxes[:, 0] - dw) / r
        boxes[:, 1] = (boxes[:, 1] - dh) / r
        boxes[:, 2] = (boxes[:, 2] - dw) / r
        boxes[:, 3] = (boxes[:, 3] - dh) / r
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)

        # 应用NMS
        try:
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(
                    boxes.tolist(),
                    scores.tolist(),
                    score_threshold=0.25,  # 降低置信度阈值
                    nms_threshold=0.45  # 放宽NMS阈值
                )
                if len(indices) > 0:
                    indices = indices.flatten()
                    detections = []
                    for i in indices:
                        if i < len(boxes) and i < len(scores) and i < len(class_ids):
                            if boxes[i].shape == (4,) and 0 <= class_ids[i] < len(self.categories):
                                detections.append({
                                    "bbox": boxes[i].tolist(),
                                    "confidence": scores[i],
                                    "class_id": class_ids[i],
                                    "label": self.categories[class_ids[i]]
                                })
                                # 更新类别计数
                                if 0 <= class_ids[i] < len(self.categories):
                                    class_counts[class_ids[i]] += 1
                                    logger.debug(
                                        f"检测到有效类别: ID={class_ids[i]}, 类别名称={self.categories[class_ids[i]]}, 置信度={scores[i]:.2f}")
                                else:
                                    logger.warning(
                                        f"检测到无效类别ID: {class_ids[i]} (总类别数: {len(self.categories)})")
                else:
                    detections = []
            else:
                detections = []
            return detections, class_counts
        except Exception as e:
            logger.error(f"处理检测输出出错: {e}")
            return [], class_counts

    def process_segmentation(self, detect_output, mask_output, ratio_pad, orig_shape):
        detections, class_counts = self.process_detections(detect_output, ratio_pad, orig_shape)
        mask, area_stats = self.process_mask(mask_output, ratio_pad, orig_shape)
        return detections, mask, class_counts, area_stats

    def process_mask(self, mask_output, ratio_pad, orig_shape):
        """
        处理掩码输出
        Args:
            mask_output: 掩码输出张量
            ratio_pad: 缩放比例和填充信息
            orig_shape: 原始图像尺寸
        Returns:
            tuple: (mask, area_stats)
        """
        try:
            # 处理掩码维度
            mask_output = mask_output.squeeze(0)
            upsampled = cv2.resize(mask_output.transpose(1, 2, 0), (640, 640), interpolation=cv2.INTER_LINEAR)
            mask = np.argmax(upsampled, axis=-1).astype(np.uint8)
            r, (dw, dh) = ratio_pad
            h, w = orig_shape

            # 计算实际缩放比例
            pixel_area = (1 / r) ** 2  # 每个像素对应的实际面积（平方米）

            # 统计各分类像素数量并过滤掉超出类别范围的分类
            unique, counts = np.unique(mask, return_counts=True)

            # 过滤掉超出类别数量的分类索引
            valid_mask = np.array([cls_id < len(self.categories) for cls_id in unique])
            unique = unique[valid_mask]
            counts = counts[valid_mask]

            # 创建面积统计（统一使用Python int作为键）
            area_stats = {int(cls_id): float(cnt * pixel_area) for cls_id, cnt in zip(unique, counts)}

            logger.info(f"掩码中检测到的类别: {unique.tolist()}")

            # 调整掩码大小和位置
            dh_int, dw_int = int(round(dh)), int(round(dw))
            mask = mask[dh_int:640 - dh_int, dw_int:640 - dw_int]
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # 将超出类别范围的像素设置为0（背景）
            mask[mask >= len(self.categories)] = 0

            return mask, area_stats
        except Exception as e:
            logger.error(f"处理掩码输出出错: {e}")
            return None, {}


def visualize_segmentation(image, mask, output_path=None):
    """
    可视化分割结果
    Args:
        image: 原始图像 (假设为 RGB)
        mask: 分割掩码
        output_propath: 输出图像路径（可选）
    Returns:
        numpy.ndarray: 可视化后的图像 (BGR, 用于 OpenCV 显示)
    """
    # 修改开始 -------------------------------------------------
    # 不再需要转换，直接复制 RGB 图像
    img_rgb = image.copy()
    # --------------------------------------------------------

    color_mask = np.zeros_like(img_rgb)

    # 应用颜色映射
    for cls_id, color in COLOR_MAP.items():
        # 注意：COLOR_MAP 中的颜色是 BGR 格式
        # 我们需要将其转换为 RGB 以匹配 img_rgb
        color_rgb = (color[2], color[1], color[0])
        color_mask[mask == cls_id] = color_rgb

    # 叠加掩码 (权重各 0.5)
    # 此时 img_rgb 和 color_mask 都是 RGB 格式
    img_result = cv2.addWeighted(img_rgb, 0.5, color_mask, 0.5, 0)

    # 如果需要保存或使用 OpenCV 显示，最后再转换回 BGR
    img_bgr = cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR)

    if output_path:
        cv2.imwrite(output_path, img_bgr)
    else:
        cv2.imshow("分割结果", img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img_bgr


def preprocess_image(image, model_input_size):
    """
    图像预处理
    Args:
        image: 输入图像（可以是文件路径或图像数组）
        model_input_size: 模型输入尺寸
    Returns:
        tuple: (input_tensor, ratio_pad, orig_shape, original_image)
    """
    try:
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise FileNotFoundError(f"图像文件未找到: {image}")
        else:
            img = image.copy()
        h, w = img.shape[:2]
        r = model_input_size / max(h, w)
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw, dh = model_input_size - new_unpad[0], model_input_size - new_unpad[1]
        dw, dh = dw / 2, dh / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        resized = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        input_tensor = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        input_tensor = input_tensor.transpose(2, 0, 1)
        input_tensor = np.ascontiguousarray(input_tensor)  # 确保内存连续
        input_tensor = np.expand_dims(input_tensor, axis=0)
        ratio_pad = (r, (dw, dh))
        return input_tensor, ratio_pad, img.shape[:2], img
    except Exception as e:
        logger.error(f"图像预处理出错: {e}")
        raise e


def process_segmentation(engine, image, model_input_size):
    """
    处理分割流程
    Args:
        engine: TensorRT推理引擎
        image: 输入图像
        model_input_size: 模型输入尺寸
    Returns:
        tuple: (detections, mask, class_counts, area_stats, original_image)
    """
    input_tensor, ratio_pad, orig_shape, original_image = preprocess_image(image, model_input_size)
    logger.info(f"图像预处理完成，原始尺寸: {orig_shape}")
    logger.info("开始执行分割...")
    try:
        detections, mask, class_counts, area_stats = engine.infer(input_tensor, ratio_pad, orig_shape)
        logger.info(f"分割完成，检测到 {len(detections)} 个物体")

        # 处理掩码中的类别，将它们转换为检测结果
        if len(detections) == 0 and mask is not None and area_stats:
            logger.info("从分割掩码中提取检测结果")
            # 将面积较大的区域转换为检测框
            new_detections = []
            new_class_counts = {cls_id: 0 for cls_id in range(len(engine.categories))}

            # 只处理面积合理的区域（可以调整阈值）
            min_area_threshold = 1000  # 最小面积阈值

            for cls_id, area in area_stats.items():
                if area > min_area_threshold and cls_id < len(engine.categories):
                    # 找到这个类别的区域
                    class_mask = (mask == cls_id).astype(np.uint8)
                    if np.sum(class_mask) > 0:
                        # 找到这个区域的边界框
                        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            if cv2.contourArea(contour) > min_area_threshold / 10:  # 过滤太小的轮廓
                                x, y, w, h = cv2.boundingRect(contour)
                                new_detections.append({
                                    "bbox": [x, y, x + w, y + h],
                                    "confidence": 0.5,  # 置信度默认值
                                    "class_id": cls_id,
                                    "label": engine.categories[cls_id]
                                })
                                new_class_counts[cls_id] += 1

            logger.info(f"从掩码中提取了 {len(new_detections)} 个检测框")
            if new_detections:
                detections = new_detections
                class_counts = new_class_counts

        return detections, mask, class_counts, area_stats, original_image
    except Exception as e:
        logger.error(f"推理错误: {e}")
        return [], None, {}, {}, image


def save_segmentation_results(original_image, mask, detections, save_path, class_counts, area_stats):
    """
    保存分割和检测结果
    Args:
        original_image: 原始图像
        mask: 分割掩码
        detections: 检测结果列表
        save_path: 保存路径
        class_counts: 类别计数
        area_stats: 面积统计
    Returns:
        bool: 保存是否成功
    """
    try:
        # 创建统计报告
        report = {
            'counts': class_counts,
            'areas': area_stats,
            'total_count': sum(class_counts.values()),
            'total_area': sum(area_stats.values()),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'save_path': save_path
        }
        car_bucket_corners = detect_car_bucket(original_image)
        final_detections, final_mask = filter_results_by_roi(detections, mask, car_bucket_corners)
        # 更新你的变量或继续后续处理
        detections = final_detections
        mask = final_mask
        img_bgr = original_image.copy()

        # 绘制掩码
        color_mask = np.zeros_like(img_bgr)
        if mask is not None:
            for cls_id, color in COLOR_MAP.items():
                # color 是 BGR, img_bgr 是 BGR, 直接赋值
                color_mask[mask == cls_id] = color
            img_bgr = cv2.addWeighted(img_bgr, 0.5, color_mask, 0.5, 0)

        # 绘制检测框
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            label = det["label"]
            conf = det["confidence"]
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            area = area_stats.get(det["class_id"], 0)
            cv2.putText(img_bgr, f"{label} {conf:.2f} ({area:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

        cv2.imwrite(save_path, img_bgr)  # 保存 BGR 图像

        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # 设置JSON保存路径为 photos 文件夹
        json_path = os.path.join(project_root, "photos", "segmentation_result.json")

        # 确保目录存在
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        # 检查是否已存在JSON文件，如果存在则读取现有内容
        results = []
        try:
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    results = []
                    for line in f:
                        line = line.strip()
                        if line:  # 跳过空行
                            try:
                                results.append(json.loads(line))
                            except json.JSONDecodeError:
                                logger.warning(f"跳过无效的JSON行: {line}")
        except Exception as e:
            logger.warning(f"读取现有JSON文件失败: {e}，将创建新文件")
            results = []

        # 添加当前结果
        results.append(report)

        # 保存JSON报告，每行一条记录
        with open(json_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')

        logger.info(f"保存分割和检测结果到: {save_path}")
        logger.info(f"JSON结果保存到: {json_path}")
        logger.info(f"类别统计：{class_counts}")
        logger.info(f"面积统计：{area_stats}")
        return True
    except Exception as e:
        logger.error(f"保存结果失败: {e}")
        return False
    finally:
        # 释放相关资源
        if 'img_bgr' in locals():
            cv2.destroyAllWindows()


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


class SegmentationManager:
    """
    分割管理器类
    负责管理分割系统的初始化、执行和资源释放
    """

    def __init__(self, config=None):
        """
        初始化分割管理器
        Args:
            config: 配置字典，如果为None则使用默认配置
        """
        if config is None:
            config = MODEL_CONFIG
        self.model_path = config["model_path"]
        self.categories = config["categories"]
        self.model_input_size = config["model_input_size"]
        self.engine = None
        self.save_dir = None
        self.config = config

    def get_class_counts(self, class_ids):
        """
        统计各类别出现次数
        Args:
            class_ids: 类别ID数组
        Returns:
            dict: 类别ID到计数的映射
        """
        if class_ids.size == 0:
            return {}
        unique, counts = np.unique(class_ids, return_counts=True)
        return dict(zip(unique.astype(int), counts))

    def initialize(self, base_dir):
        """
        初始化分割系统
        Args:
            base_dir: 基础目录路径
        Returns:
            bool: 初始化是否成功
        """
        try:
            initialize_cuda()
            self.engine = TRTInferenceEngine(
                engine_path=self.model_path,
                categories=self.categories,
                conf_threshold=self.config["conf_threshold"],
                nms_threshold=self.config["nms_threshold"]
            )
            self.save_dir = os.path.join(base_dir, SAVE_CONFIG["sub_dirs"]["segmentation"])
            os.makedirs(self.save_dir, exist_ok=True)
            logger.info("分割系统初始化成功")
            return True
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            release_cuda()
            return False

    def segment_image(self, image):
        """
        执行图像分割
        Args:
            image_path: 图像文件路径
        Returns:
            tuple: (detections, mask, save_path, class_counts, area_stats)
        """
        try:
            if image is None:
                raise FileNotFoundError(f"无法读取图像: {image}")
            detections, mask, class_counts, area_stats, original_image = process_segmentation(
                self.engine, image, self.model_input_size)
            save_path = os.path.join(
                self.save_dir,
                f"segment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            save_segmentation_results(original_image, mask, detections, save_path, class_counts, area_stats)
            return detections, mask, save_path, class_counts, area_stats
        except Exception as e:
            logger.error(f"分割失败: {e}")
            return None, None, None, {}, {}

    def release(self):
        """
        释放资源
        清理引擎和CUDA资源
        """
        if self.engine:
            self.engine.release()
            self.engine = None
        release_cuda()
        logger.info("分割系统资源已释放")

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
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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


def filter_results_by_roi(detections, mask, car_bucket_corners):
    """
    根据车框坐标过滤检测结果和掩码
    """
    # 1. 计算车框的矩形边界 (x1, y1, x2, y2)
    # car_bucket_corners 是 float32 类型，需要转为 int
    points = car_bucket_corners.astype(np.int32)
    x, y, w, h = cv2.boundingRect(points)
    x1, y1, x2, y2 = x, y, x + w, y + h

    logger.info(f"计算出的车框矩形边界: ({x1}, {y1}) to ({x2}, {y2})")

    # --- 2. 过滤 detections (检测框) ---
    # 策略：保留与车框区域有重叠(IOU)或者中心点落在车框内的检测框
    filtered_detections = []
    for det in detections:
        # 从字典中提取坐标
        # 注意：截图显示 bbox 是 [x1, y1, x2, y2] 格式
        dx1, dy1, dx2, dy2 = map(int, det['bbox'])

        # 计算检测框中心点
        center_x = (dx1 + dx2) // 2
        center_y = (dy1 + dy2) // 2

        # 判定条件：如果检测框的中心点在车框矩形内，则保留
        # (你可以根据需求改为计算 IOU，但中心点法计算更快且通常够用)
        if (x1 <= center_x <= x2) and (y1 <= center_y <= y2):
            filtered_detections.append(det)
        else:
            # 可选：打印被过滤掉的框
            # logger.debug(f"过滤掉框: {det['label']} at ({center_x}, {center_y})")
            pass

    logger.info(f"原始检测数: {len(detections)}, 过滤后检测数: {len(filtered_detections)}")

    # --- 3. 处理 mask (掩码) ---
    # 策略：创建一个全黑的掩码，只保留车框矩形区域内的像素
    h_mask, w_mask = mask.shape
    roi_mask = np.zeros((h_mask, w_mask), dtype=np.uint8)

    # 在 roi_mask 上将车框区域设为 255 (白色)
    # 注意边界检查，防止坐标超出图像范围
    x1_safe = max(0, x1)
    y1_safe = max(0, y1)
    x2_safe = min(w_mask, x2)
    y2_safe = min(h_mask, y2)

    roi_mask[y1_safe:y2_safe, x1_safe:x2_safe] = 255

    # 使用位运算将原 mask 限制在 roi_mask 范围内
    # 这意味着车框外的所有像素都会变成 0 (背景)
    masked_result = cv2.bitwise_and(mask, mask, mask=roi_mask)

    return filtered_detections, masked_result


def simple_segment(image_path, config=None):
    """
    简单的分割接口
    Args:
        image_path: 图像文件路径
        config: 配置字典（可选）
    Returns:
        tuple: (detections, mask, save_path, class_counts, area_stats)
    """
    class_counts = {}
    area_stats = {}

    if image_path is None:
        logger.error("必须提供图像路径")
        return None, None, None, class_counts, area_stats
    #危险物检测
    segmenter = SegmentationManager(config or MODEL_CONFIG)
    try:
        if not segmenter.initialize(SAVE_CONFIG["base_dir"]):
            logger.error("初始化失败: Failed to load TensorRT engine.")
            return None, None, None, class_counts, area_stats
        image = cv2.imread(image_path)
        detections, mask, save_path, class_counts, area_stats = segmenter.segment_image(image)

        # 只要mask不为None，就视为分割成功，即使没有检测到物体
        if mask is not None:
            logger.info("分割成功" + (", 但没有检测到物体" if len(detections) == 0 else ""))
            area_stats = {k: v / 10000 for k, v in area_stats.items()}
            return detections, mask, save_path, class_counts, area_stats
        else:
            logger.error("分割失败: 未能生成有效的分割掩码")
            return None, None, None, class_counts, area_stats
    except Exception as e:
        logger.error(f"发生异常: {str(e)}")
        return None, None, None, class_counts, area_stats
    finally:
        segmenter.release()


def main():
    """
    测试主函数
    用于测试分割系统的功能
    """
    image_path = "test_image.jpg"
    if not os.path.exists(image_path):
        print(f"错误：图像文件不存在: {image_path}")
        return

    result = simple_segment(image_path)
    if result is not None:
        detections, mask, save_path, class_counts, area_stats = result
        if detections is not None and mask is not None:
            print(f"分割完成，结果保存至: {save_path}")
            print(f"检测到 {len(detections)} 个物体")
            print(f"类别统计: {class_counts}")
            print(f"面积统计: {area_stats}")
        else:
            print("分割失败")
    else:
        print("分割失败")


if __name__ == "__main__":
    main()