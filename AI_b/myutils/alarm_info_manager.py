import datetime
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from logger_config import logger
from myutils.config_util import config

class AlarmInfoManager:
    def __init__(self):
        # 从配置文件读取报警设置
        if config is None:
            logger.warning("Configuration not loaded, using default alarm settings")
            self.enabled = True
            self.max_history = 100
            self.helmet_enabled = True
            self.alert_on_no_helmet = True
            self.fire_enabled = True
            self.alert_on_fire = True
            self.alert_on_smoke = True
        else:
            alarm_config = config.get('alarm', {})
            self.enabled = alarm_config.get('enabled', True)
            self.max_history = alarm_config.get('max_alarm_history', 100)
            
            helmet_config = alarm_config.get('helmet', {})
            self.helmet_enabled = helmet_config.get('enabled', True)
            self.alert_on_no_helmet = helmet_config.get('alert_on_no_helmet', True)
            
            fire_config = alarm_config.get('fire', {})
            self.fire_enabled = fire_config.get('enabled', True)
            self.alert_on_fire = fire_config.get('alert_on_fire', True)
            self.alert_on_smoke = fire_config.get('alert_on_smoke', True)
        
        self.latest_helmet_alarm = None
        self.latest_fire_alarm = None
        self.alarm_history = []  # 存储报警历史记录

    def update_helmet_alarm(self, results):
        """更新安全帽报警信息"""
        if not self.enabled or not self.helmet_enabled:
            return
            
        if results:
            # 筛选需要报警的结果
            alarm_results = []
            for result in results:
                if result['label'] == '未正确佩戴安全帽' and self.alert_on_no_helmet:
                    alarm_results.append(result)
            
            if alarm_results:
                # 获取最新的报警记录
                latest_alarm = max(alarm_results, key=lambda x: x['timestamp'])
                self.latest_helmet_alarm = latest_alarm
                
                # 添加到历史记录
                self._add_to_history(latest_alarm, 'helmet')
                logger.info(f"安全帽报警: {latest_alarm['label']}, 置信度: {latest_alarm['confidence']:.2f}")

    def update_fire_alarm(self, results):
        """更新火焰烟雾报警信息"""
        if not self.enabled or not self.fire_enabled:
            return
            
        if results:
            # 筛选需要报警的结果
            alarm_results = []
            for result in results:
                if result['label'] == '火焰' and self.alert_on_fire:
                    alarm_results.append(result)
                elif result['label'] == '烟雾' and self.alert_on_smoke:
                    alarm_results.append(result)
            
            if alarm_results:
                # 获取最新的报警记录
                latest_alarm = max(alarm_results, key=lambda x: x['timestamp'])
                self.latest_fire_alarm = latest_alarm
                
                # 添加到历史记录
                self._add_to_history(latest_alarm, 'fire')
                logger.info(f"火灾报警: {latest_alarm['label']}, 置信度: {latest_alarm['confidence']:.2f}")

    def _add_to_history(self, alarm, alarm_type):
        """添加报警到历史记录"""
        alarm_record = {
            'timestamp': alarm['timestamp'],
            'type': alarm_type,
            'label': alarm['label'],
            'confidence': alarm['confidence'],
            'bbox': alarm.get('bbox', [])
        }
        
        self.alarm_history.append(alarm_record)
        
        # 限制历史记录数量
        if len(self.alarm_history) > self.max_history:
            self.alarm_history = self.alarm_history[-self.max_history:]

    def get_latest_alarm(self):
        """返回最新一条报警，无论是安全帽还是火焰烟雾"""
        if not self.enabled:
            return None
            
        alarms = []
        if self.latest_helmet_alarm:
            alarms.append(self.latest_helmet_alarm)
        if self.latest_fire_alarm:
            alarms.append(self.latest_fire_alarm)
        if not alarms:
            return None
        # 按时间戳排序，返回最新
        return max(alarms, key=lambda x: x['timestamp'])

    def get_alarm_history(self, alarm_type=None, limit=10):
        """获取报警历史记录"""
        if not self.enabled:
            return []
            
        history = self.alarm_history
        if alarm_type:
            history = [h for h in history if h['type'] == alarm_type]
        
        # 按时间戳排序，返回最新的记录
        history.sort(key=lambda x: x['timestamp'], reverse=True)
        return history[:limit]

    def get_alarm_statistics(self, date=None):
        """获取报警统计信息"""
        if not self.enabled:
            return {'helmet_count': 0, 'fire_count': 0, 'smoke_count': 0}
            
        if date is None:
            date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        helmet_count = sum(1 for h in self.alarm_history 
                          if h['type'] == 'helmet' and h['timestamp'].startswith(date))
        fire_count = sum(1 for h in self.alarm_history 
                        if h['type'] == 'fire' and h['label'] == '火焰' and h['timestamp'].startswith(date))
        smoke_count = sum(1 for h in self.alarm_history 
                         if h['type'] == 'fire' and h['label'] == '烟雾' and h['timestamp'].startswith(date))
        
        return {
            'helmet_count': helmet_count,
            'fire_count': fire_count,
            'smoke_count': smoke_count
        }

    @staticmethod
    def draw_chinese_text(img, text, position, color=(255,0,0), font_size=20, font_path="simhei.ttf"):
        """
        在OpenCV的BGR图像上用Pillow绘制中文
        """
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
        draw.text(position, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)