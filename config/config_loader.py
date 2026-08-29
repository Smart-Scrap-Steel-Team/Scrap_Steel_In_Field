import os
import yaml

# Path to the config.yaml file (assume it's in the project root)
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')

class ConfigLoader:
    _config = None

    @classmethod
    def _load_config(cls):
        if cls._config is None:
            try:
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    cls._config = yaml.safe_load(f)
            except Exception as e:
                print(f"Failed to load config.yaml: {e}")
                cls._config = {}

    @classmethod
    def is_module_enabled(cls, module_name):
        cls._load_config()
        return cls._config.get('modules', {}).get(module_name, False)

    @classmethod
    def get_all_module_status(cls):
        cls._load_config()
        return cls._config.get('modules', {}) 