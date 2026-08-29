import yaml
from logger_config import logger

# 加载配置文件
# 返回 config 字典对象
def load_config():
    """从 config.yaml 文件加载配置"""
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        return None

# 全局配置变量
config = load_config() 