import os
import sys
# 添加父目录到系统路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 日志配置
LOG_CONFIG = {
    "base_dir": os.path.join(PROJECT_ROOT, "logs"),  # 日志基础目录
    "level": "DEBUG",  # 启用DEBUG级别日志
    "sub_dirs": {
        "segmentation": "segmentation",  # 检测日志
        "image": "image",         # 图像处理日志
        "system": "system"        # 系统日志
    }
}

# 模型配置
MODEL_CONFIG = {
    "model_path": os.path.join(PROJECT_ROOT, "models", "scrap_steel.engine"),
    # "categories": ["方管1","方管2","方管3","圆管1","圆管2","圆管3","平整钢板","长条钢板1",
    #                 "长钢板2","长条钢板3","角钢","钢筋1","钢筋2","管道连接件","金属连接件","圆铁片"],
    "categories": ["Sqr Pipe 1", "Sqr Pipe 2", "Sqr Pipe 3","Round Pipe 1", "Round Pipe 2", "Round Pipe 3",
                    "Flat Plate", "Long Plate 1","Long Plate 2", "Long Plate 3","Angle", "Rebar 1", "Rebar 2",
                    "Pipe Fitting", "Metal Fitting", "Round Plate"],                
    "conf_threshold": 0.05,
    "nms_threshold": 0.1,
    "model_input_size": 640
}

# 图片保存配置
SAVE_CONFIG = {
    "base_dir": os.path.join(PROJECT_ROOT, "photos"),  # 基础保存目录
    "sub_dirs": {
        "original": "original",    # 原始图像
        "vehicle": "vehicle",      # 车体图像
        "dangerous_det": "dangerous_det",  # 危险品检测结果
        "segmentation": "segmentation"
#        "scrap_seg": "scrap_seg",   # 废钢分割结果
#        "test_original": os.path.join("test", "original"),    # 测试用原始图像
#        "test_detection": os.path.join("test", "detection")   # 测试用检测结果
    }
}