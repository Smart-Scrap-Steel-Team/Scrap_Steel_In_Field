import os
import logging
from logging.handlers import RotatingFileHandler
import time

def setup_logger():
    """
    配置日志系统
    返回配置好的logger对象
    """
    # 创建logs目录（如果不存在）
    log_dir = 'logs/vehicle_detection'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 创建日志文件名（包含时间戳）
    log_filename = os.path.join(log_dir, f'vehicle_detection_{time.strftime("%Y%m%d%H")}.log')
    
    # 创建日志记录器
    logger = logging.getLogger('VehicleDetection')
    logger.setLevel(logging.DEBUG)
    
    # 如果已经有处理器，则不再添加
    if logger.handlers:
        return logger
    
    # 创建文件处理器（使用RotatingFileHandler实现日志轮转）
    file_handler = RotatingFileHandler(
        log_filename,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=2,  # 保留2个备份文件
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 为处理器设置格式化器
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 为日志记录器添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 创建全局logger实例
logger = setup_logger() 