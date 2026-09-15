import logging
import os
from datetime import datetime

def setup_logger(logger_name, base_dir, sub_dir):
    """设置日志记录器
    
    Args:
        logger_name: 日志记录器名称
        base_dir: 基础保存目录
        sub_dir: 日志子目录名称
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 创建logs目录
    log_dir = os.path.join(base_dir, sub_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成日志文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{logger_name}_{timestamp}.log")
    
    # 配置日志格式
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 配置日志记录器
    logger = logging.getLogger(logger_name)
    # 设置默认日志级别为DEBUG
    logger.setLevel(logging.DEBUG)
    
    # 如果logger已经存在处理器，先移除
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger