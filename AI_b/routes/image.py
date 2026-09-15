import os
import json
from flask import Blueprint, request, jsonify, send_file
from services.image_service import ImageService

image_bp = Blueprint('image', __name__)
image_service = ImageService()

# Real-time monitor image
@image_bp.route('/get_image')
def get_image():
    return image_service.get_image()

@image_bp.route('/set_image_path', methods=['POST'])
def set_image_path():
    return image_service.set_image_path()

@image_bp.route('/update_image', methods=['POST'])
def update_image():
    return image_service.update_image()

# Dangerous detection image
@image_bp.route('/get_detection')
def get_detection():
    return image_service.get_detection()

@image_bp.route('/set_detection_path', methods=['POST'])
def set_detection_path():
    return image_service.set_detection_path()

@image_bp.route('/update_detection', methods=['POST'])
def update_detection():
    return image_service.update_detection()

# Segmentation image
@image_bp.route('/get_segmentation')
def get_segmentation():
    return image_service.get_segmentation()

@image_bp.route('/set_segmentation_path', methods=['POST'])
def set_segmentation_path():
    return image_service.set_segmentation_path()

@image_bp.route('/update_segmentation', methods=['POST'])
def update_segmentation():
    return image_service.update_segmentation()

# JSON data
@image_bp.route('/get_json_data')
def get_json_data():
    return image_service.get_json_data()

@image_bp.route('/get_json_history')
def get_json_history():
    return image_service.get_json_history()

@image_bp.route('/get_segmentation_stats')
def get_segmentation_stats():
    return image_service.get_segmentation_stats()

@image_bp.route('/get_latest_danger_counts')
def get_latest_danger_counts():
    return image_service.get_latest_danger_counts()

@image_bp.route('/get_danger_total')
def get_danger_total():
    return image_service.get_danger_total()

# File status check
@image_bp.route('/check_segmentation_file_time')
def check_segmentation_file_time():
    return image_service.check_file_time('segmentation')

@image_bp.route('/check_danger_file_time')
def check_danger_file_time():
    return image_service.check_file_time('danger')

@image_bp.route('/check_danger_file_new_data')
def check_danger_file_new_data():
    return image_service.check_file_new_data('danger')

@image_bp.route('/check_segmentation_file_new_data')
def check_segmentation_file_new_data():
    return image_service.check_file_new_data('segmentation') 

@image_bp.route('/get_vehicle_history')
def get_vehicle_history():
    """获取车辆历史信息，调用 ImageService 的方法"""
    return image_service.get_vehicle_history()