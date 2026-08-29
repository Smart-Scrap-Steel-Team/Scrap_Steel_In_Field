import os
import json
from flask import Blueprint, request, jsonify
import datetime
from services.safety_detection_service import SafetyDetectionService

safety_bp = Blueprint('safety', __name__)
safety_service = SafetyDetectionService()

@safety_bp.route('/')
def safety_index():
    """Render safety index page."""
    return safety_service.safety_index()

@safety_bp.route('/video_feed/<int:cam_index>')
def safety_video_feed(cam_index):
    """Video stream route for safety detection."""
    return safety_service.video_feed(cam_index)

@safety_bp.route('/api/latest_results/<int:cam_index>', methods=['GET'])
def get_latest_results(cam_index):
    """Get latest detection results API."""
    return safety_service.get_latest_results(cam_index)

@safety_bp.route('/api/history/<int:cam_index>', methods=['GET'])
def get_history(cam_index):
    """Get detection history API."""
    return safety_service.get_history(cam_index)

@safety_bp.route('/api/stats/<int:cam_index>', methods=['GET'])
def get_stats(cam_index):
    """Get detection statistics API."""
    return safety_service.get_stats(cam_index) 

@safety_bp.route('/safety/api/today_statistics', methods=['GET'])
def get_today_statistics():
    """获取今日统计信息"""
    helmet_file = "detection_results/helmet_detections.json"
    fire_file = "detection_results/fire_detections.json"
    helmet_data = []
    fire_data = []
    
    # 读取安全帽检测数据
    if os.path.exists(helmet_file):
        with open(helmet_file, 'r', encoding='utf-8') as f:
            try:
                helmet_data = json.load(f)
            except:
                helmet_data = []
                
    # 读取火灾检测数据
    if os.path.exists(fire_file):
        with open(fire_file, 'r', encoding='utf-8') as f:
            try:
                fire_data = json.load(f)
            except:
                fire_data = []

    today = datetime.datetime.now().strftime("%Y-%m-%d")

    # 安全帽统计
    total_person = sum(1 for d in helmet_data if d['timestamp'].startswith(today))
    helmet_count = sum(1 for d in helmet_data if d['label'] == '佩戴安全帽' and d['timestamp'].startswith(today))
    no_helmet = sum(1 for d in helmet_data if d['label'] == '未正确佩戴安全帽' and d['timestamp'].startswith(today))
    helmet_rate = f"{(helmet_count/total_person*100):.1f}%" if total_person else "0%"

    # 火灾检测统计
    fire_detection_count = sum(1 for d in fire_data if d['timestamp'].startswith(today))
    fire_alarm_count = sum(1 for d in fire_data if d['label'] == '火焰' and d['timestamp'].startswith(today))
    
    # 获取最近报警时间
    fire_alarms = [d for d in fire_data if d['label'] == '火焰' and d['timestamp'].startswith(today)]
    last_fire_alarm_time = max(fire_alarms, key=lambda x: x['timestamp'])['timestamp'] if fire_alarms else '--'

    return jsonify({
        "date": today,
        "totalPerson": total_person,
        "helmetCount": helmet_count,
        "noHelmet": no_helmet,
        "helmetRate": helmet_rate,
        "fireDetectionCount": fire_detection_count,
        "fireAlarmCount": fire_alarm_count,
        "lastFireAlarmTime": last_fire_alarm_time
    })
