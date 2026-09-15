from flask import Blueprint
from services.vehicle_detection_service import VehicleDetectionService

vehicle_bp = Blueprint('vehicle', __name__)
vehicle_service = VehicleDetectionService()

@vehicle_bp.route('/video_feed_entrance')
def video_feed_entrance():
    """Entrance video stream route."""
    return vehicle_service.video_feed_entrance()

@vehicle_bp.route('/video_feed_exit')
def video_feed_exit():
    """Exit video stream route."""
    return vehicle_service.video_feed_exit()

@vehicle_bp.route('/get_detection_history_entrance')
def get_detection_history_entrance():
    """Return entrance detection history."""
    return vehicle_service.get_detection_history_entrance()

@vehicle_bp.route('/get_detection_history_exit')
def get_detection_history_exit():
    """Return exit detection history."""
    return vehicle_service.get_detection_history_exit()

@vehicle_bp.route('/get_current_vehicles')
def get_current_vehicles():
    """Return current vehicles info."""
    return vehicle_service.get_current_vehicles() 