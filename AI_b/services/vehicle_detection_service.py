from myutils.vehicle_detection import VehicleDetection
from flask import Response, jsonify
import threading

class VehicleDetectionService:
    """车辆检测服务逻辑封装。"""
    def __init__(self):
        # 实例化车辆检测主类
        self.detector = VehicleDetection()
        # 如果车辆检测和进场检测都启用，则自动启动相关线程
        if self.detector.enabled and self.detector.entrance_enabled:
            # 启动进场摄像头采集线程（不断采集摄像头画面，更新self.current_frame）
            entrance_camera_thread = threading.Thread(
                target=self.detector.camera_stream,
                args=(self.detector.entrance_channel, 'entrance'),
                daemon=True
            )
            entrance_camera_thread.start()
            # 启动进场帧处理线程（不断处理采集到的帧，做检测和业务逻辑）
            entrance_process_thread = threading.Thread(
                target=self.detector.process_frame_entrance,
                daemon=True
            )
            entrance_process_thread.start()

    def video_feed_entrance(self):
        # 推送进场视频流（MJPEG流），前端可直接显示
        if not self.detector.enabled:
            return jsonify({"error": "Vehicle detection system not initialized"}), 500
        return Response(self.detector.generate_frames_entrance(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def video_feed_exit(self):
        # 推送出场视频流（如有需要可补全出场线程启动）
        if not self.detector.enabled:
            return jsonify({"error": "Vehicle detection system not initialized"}), 500
        return Response(self.detector.generate_frames_exit(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    def get_detection_history_entrance(self):
        # 获取进场检测历史记录
        return self.detector.get_detection_history_entrance()

    def get_detection_history_exit(self):
        # 获取出场检测历史记录
        return self.detector.get_detection_history_exit()

    def get_current_vehicles(self):
        # 获取当前在场车辆信息
        return self.detector.get_current_vehicles_info() 