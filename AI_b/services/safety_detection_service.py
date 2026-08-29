from myutils.safety_detection_system import SafetyDetectionSystem
from flask import Response, jsonify, request, render_template
import datetime

class SafetyDetectionService:
    """安全检测服务逻辑封装。"""
    def __init__(self):
        # 实例化安全检测主类（包含安全帽、火焰等检测）
        self.system = SafetyDetectionSystem()

    def safety_index(self):
        # 渲染安全检测主页
        return render_template('index.html')

    def video_feed(self, cam_index):
        # 推送安全检测视频流（MJPEG流），前端可直接显示
        if not self.system.enabled:
            return jsonify({"error": "Safety detection system not initialized or disabled"}), 500
        def generate():
            while True:
                frame = self.system.get_latest_frame(cam_index)
                if frame is not None:
                    results = self.system.get_latest_results(cam_index)
                    # 可选：在帧上绘制检测结果（如检测框、标签）
                    import cv2
                    for result in results:
                        x1, y1, x2, y2 = result['bbox']
                        color = (0, 255, 0)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        label = f"{result['label']} {result['confidence']:.2f}"
                        cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                import time; time.sleep(0.1)
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def get_latest_results(self, cam_index):
        # 获取指定摄像头的最新检测结果
        if not self.system.enabled:
            return jsonify({"error": "Safety detection system not initialized or disabled"}), 500
        results = self.system.get_latest_results(cam_index)
        return jsonify(results)

    def get_history(self, cam_index):
        # 获取指定摄像头的历史检测记录（可按时间范围筛选）
        if not self.system.enabled:
            return jsonify({"error": "Safety detection system not initialized or disabled"}), 500
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        history = self.system.get_history(cam_index, start_date, end_date)
        return jsonify(history)

    def get_stats(self, cam_index):
        # 获取指定摄像头的检测统计信息
        if not self.system.enabled:
            return jsonify({"error": "Safety detection system not initialized or disabled"}), 500
        results = self.system.get_latest_results(cam_index)
        stats = {
            'total_detections': len(results),  # 总检测次数
            'by_category': {},                # 按类别统计
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        for result in results:
            category = result['label']
            if category not in stats['by_category']:
                stats['by_category'][category] = 0
            stats['by_category'][category] += 1
        return jsonify(stats) 