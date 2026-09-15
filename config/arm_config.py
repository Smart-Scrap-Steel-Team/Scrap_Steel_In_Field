"""
机械臂配置文件
包含机械臂相关的配置参数
"""

# 日志配置
LOG_CONFIG = {
    # 日志文件保存目录结构
    "arm1": {
        "base_dir": None,  # 将在运行时设置为当前文件所在目录
        "sub_dir": "logs/arm1"
    },
}

# 机械臂操作配置
ARM_CONFIG = {
    "arm1": {
        "grab_count": 5,  # 默认抓取次数
        "photo_retry": {
            "max_retries": 10,  # 拍照最大重试次数
            "retry_delay": 2.0  # 拍照重试间隔时间(秒)
        }
    }
} 