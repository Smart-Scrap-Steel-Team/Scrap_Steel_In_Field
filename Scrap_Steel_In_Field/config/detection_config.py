import os
import sys
# 添加父目录到系统路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 日志配置
LOG_CONFIG = {
    "base_dir": os.path.join(PROJECT_ROOT, "logs"),  # 日志基础目录
    "sub_dirs": {
        "detection": "detection",  # 检测日志
        "image": "image",         # 图像处理日志
        "system": "system"        # 系统日志
    }
}

# 模型配置
MODEL_CONFIG = {
    "model_path": os.path.join(PROJECT_ROOT, "models", "dangerous.engine"),
    "categories": ['GasCyl1', 'GasCyl2', 'FireExt1', 'FireExt2'],
    "conf_threshold": 0.35,
    "nms_threshold": 0.4,
    "model_input_size": 640
}

# 图片保存配置
SAVE_CONFIG = {
    "base_dir": os.path.join(PROJECT_ROOT, "photos"),  # 基础保存目录
    "sub_dirs": {
        "original": "original",    # 原始图像
        "vehicle": "vehicle",      # 车体图像
        "dangerous_det": "dangerous_det",  # 危险品检测结果
#        "scrap_seg": "scrap_seg",   # 废钢分割结果
#        "test_original": os.path.join("test", "original"),    # 测试用原始图像
#        "test_detection": os.path.join("test", "detection")   # 测试用检测结果
    }
} 