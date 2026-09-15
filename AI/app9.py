from flask import Flask, send_from_directory, jsonify, request, send_file, render_template, Response
import time
import os
import json
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from get_image.get_img import Camera
from config.camera_config import CAMERA_CONFIG

app = Flask(__name__, static_folder='static', template_folder='templates')

# 初始化摄像头对象
camera = None
def get_camera():
    global camera
    if camera is None:
        camera = Camera(
            CAMERA_CONFIG["camera_ip"],
            CAMERA_CONFIG["username"],
            CAMERA_CONFIG["password"]
        )
    return camera

# 默认图片路径（当摄像头不可用时显示）
def get_default_image():
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new('RGB', (640, 480), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    try:
        font_large = ImageFont.truetype("simhei.ttf", 32)
        font_medium = ImageFont.truetype("simhei.ttf", 24)
        font_small = ImageFont.truetype("simhei.ttf", 18)
    except:
        try:
            font_large = ImageFont.truetype("msyh.ttc", 32)
            font_medium = ImageFont.truetype("msyh.ttc", 24)
            font_small = ImageFont.truetype("msyh.ttc", 18)
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()

    text1 = "Camera Offline"
    text2 = "摄像头未连接"
    text3 = f"IP: {CAMERA_CONFIG.get('camera_ip', 'Unknown')}"

    bbox1 = draw.textbbox((0, 0), text1, font=font_large)
    bbox2 = draw.textbbox((0, 0), text2, font=font_medium)
    bbox3 = draw.textbbox((0, 0), text3, font=font_small)

    x1 = (640 - (bbox1[2] - bbox1[0])) // 2
    x2 = (640 - (bbox2[2] - bbox2[0])) // 2
    x3 = (640 - (bbox3[2] - bbox3[0])) // 2

    draw.text((x1, 180), text1, fill=(255, 255, 255), font=font_large)
    draw.text((x2, 230), text2, fill=(200, 200, 200), font=font_medium)
    draw.text((x3, 280), text3, fill=(150, 150, 150), font=font_small)

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img_cv

image_paths = {
    'original': None,
    'detection': None,
    'pie_chart': None,
    'json_data': None,
    'segmentation': None
}

@app.route('/')
def index():
    update_segmentation()
    update_image()
    update_detection()
    return render_template('index_bak.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

'''
实时监控图像
'''
@app.route('/get_image')
def get_image():
    path = image_paths['original']
    if path is None or not os.path.exists(path):
        return jsonify({'error': 'Image not found'}), 404
    return send_file(path, mimetype='image/jpeg')

@app.route('/set_image_path', methods=['POST'])
def set_image_path():
    data = request.json
    new_path = data.get('path')
    if new_path and os.path.exists(new_path):
        image_paths['original'] = new_path
        return jsonify({'success': True, 'message': 'Image path updated successfully'})
    return jsonify({'success': False, 'message': 'Invalid image path'})

@app.route('/update_image', methods=['POST'])
def update_image():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_dir = os.path.join(project_root, 'photos', 'original')
    if not os.path.exists(image_dir):
        return jsonify({'success': False, 'message': 'Original image directory not found'})
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]
    if not image_files:
        return jsonify({'success': False, 'message': 'No original images found'})
    latest_image_file = max(image_files, key=lambda f: os.path.getmtime(os.path.join(image_dir, f)))
    latest_image_path = os.path.join(image_dir, latest_image_file)
    image_paths['original'] = latest_image_path
    return jsonify({'success': True, 'path': '/get_image'})

'''
危险品图像
'''
@app.route('/get_detection')
def get_detection():
    path = image_paths['detection']
    if path is None or not os.path.exists(path):
        return jsonify({'error': 'Detection image not found'}), 404
    return send_file(path, mimetype='image/jpeg')

@app.route('/set_detection_path', methods=['POST'])
def set_detection_path():
    data = request.json
    new_path = data.get('path')
    if new_path and os.path.exists(new_path):
        image_paths['detection'] = new_path
        return jsonify({'success': True, 'message': 'Detection path updated'})
    return jsonify({'success': False, 'message': 'Invalid detection path'})

@app.route('/update_detection', methods=['POST'])
def update_detection():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    detection_dir = os.path.join(project_root, 'photos', 'dangerous_det')
    if not os.path.exists(detection_dir):
        return jsonify({'success': False, 'message': 'Detection directory not found'})
    image_files = [f for f in os.listdir(detection_dir) if f.lower().endswith(('.jpg', '.png'))]
    if not image_files:
        return jsonify({'success': False, 'message': 'No detection images found'})
    latest_image_file = max(image_files, key=lambda f: os.path.getmtime(os.path.join(detection_dir, f)))
    latest_image_path = os.path.join(detection_dir, latest_image_file)
    image_paths['detection'] = latest_image_path
    return jsonify({'success': True, 'path': '/get_detection'})

@app.route('/update_json_data', methods=['POST'])
def update_json_data():
    path = os.path.join(os.path.dirname(__file__), '..', 'photos', 'detection_results.json')
    if path is None:
        return jsonify({'success': False, 'message': 'No JSON data available'})
    return jsonify({'success': True, 'path': '/get_json_data'})

@app.route('/update_json_history', methods=['POST'])
def update_json_history():
    path = os.path.join(os.path.dirname(__file__), '..', 'photos', 'detection_results.json')
    if path is None:
        return jsonify({'success': False, 'message': 'No JSON data available'})
    return jsonify({'success': True, 'path': '/get_json_history'})

@app.route('/get_latest_danger_counts')
def get_latest_danger_counts():
    path = os.path.join(os.path.dirname(__file__), '..', 'photos', 'detection_results.json')
    if not os.path.exists(path):
        return jsonify({'error': 'JSON data not found'}), 404
    try:
        last_line = ""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    last_line = line.strip()
        if last_line:
            data = json.loads(last_line)
            counts = data.get('counts', {})
            danger_types = ['GasCyl1', 'GasCyl2', 'FireExt1', 'FireExt2']
            result = {dtype: counts.get(dtype, 0) for dtype in danger_types}
            return jsonify({'success': True, 'counts': result})
        else:
            return jsonify({'success': False, 'message': 'No valid data found'})
    except Exception as e:
        return jsonify({'success': False, 'message': '读取JSON数据失败', 'error': str(e)})

@app.route('/get_danger_total')
def get_danger_total():
    path = os.path.join(os.path.dirname(__file__), '..', 'photos', 'detection_results.json')
    if not os.path.exists(path):
        return jsonify({'error': 'JSON data not found'}), 404
    try:
        last_line = ""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    last_line = line.strip()
        if last_line:
            data = json.loads(last_line)
            counts = data.get('counts', {})
            total_count = sum(counts.values()) if counts else 0
            return jsonify({'success': True, 'total_count': total_count})
        else:
            return jsonify({'success': False, 'message': 'No valid data found'})
    except Exception as e:
        return jsonify({'success': False, 'message': '读取JSON数据失败', 'error': str(e)})

@app.route('/check_segmentation_file_time')
def check_segmentation_file_time():
    try:
        file_path = os.path.join(os.path.dirname(__file__), '..', 'photos', 'segmentation_result.json')
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        last_modified = os.path.getmtime(file_path)
        return jsonify({'success': True, 'last_modified': last_modified})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/check_danger_file_time')
def check_danger_file_time():
    try:
        file_path = os.path.join(os.path.dirname(__file__), '..', 'photos', 'detection_results.json')
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        last_modified = os.path.getmtime(file_path)
        return jsonify({'success': True, 'last_modified': last_modified})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/check_danger_file_new_data')
def check_danger_file_new_data():
    try:
        file_path = os.path.join(os.path.dirname(__file__), '..', 'photos', 'detection_results.json')
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        last_line = ""
        data_count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data_count += 1
                    last_line = line.strip()
        if not last_line:
            return jsonify({'success': False, 'error': 'No data in file'}), 404
        data = json.loads(last_line)
        return jsonify({'success': True, 'last_line_data': data, 'data_count': data_count, 'last_modified': os.path.getmtime(file_path)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/check_segmentation_file_new_data')
def check_segmentation_file_new_data():
    try:
        file_path = os.path.join(os.path.dirname(__file__), '..', 'photos', 'segmentation_result.json')
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        last_line = ""
        data_count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data_count += 1
                    last_line = line.strip()
        if not last_line:
            return jsonify({'success': False, 'error': 'No data in file'}), 404
        data = json.loads(last_line)
        return jsonify({'success': True, 'last_line_data': data, 'data_count': data_count, 'last_modified': os.path.getmtime(file_path)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

@app.route('/arm_status')
def get_arm_status():
    try:
        status_path = os.path.join(os.path.dirname(__file__), '..', 'photos', 'arm_status.json')
        with open(status_path, 'r', encoding='utf-8') as f:
            status_data = json.load(f)
            return jsonify({'arm1_status': status_data.get('arm1_status', False)})
    except Exception as e:
        return jsonify({'arm1_status': None, 'error': str(e)}), 500

"""
车辆信息
"""
@app.route('/set_json_data_path', methods=['POST'])
def set_json_data_path():
    data = request.json
    new_path = data.get('path')
    if new_path and os.path.exists(new_path):
        image_paths['json_data'] = new_path
        return jsonify({'success': True, 'message': 'JSON data path updated'})
    return jsonify({'success': False, 'message': 'Invalid JSON data path'})

@app.route('/vehicle_records/vehicles.json')
def serve_vehicles_json():
    file_path = os.path.join(os.path.dirname(__file__), '..', 'vehicle_records', 'vehicles.json')
    if os.path.exists(file_path):
        return send_file(file_path)
    else:
        return jsonify({'error': 'Vehicles JSON file not found'}), 404

@app.route('/get_json_data')
def get_json_data():
    path = os.path.join(os.path.dirname(__file__), '..', 'photos', 'detection_results.json')
    if not os.path.exists(path):
        return jsonify({'error': 'JSON data not found'}), 404
    try:
        last_line = ""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    last_line = line.strip()
        if last_line:
            data = json.loads(last_line)
            return jsonify({'success': True, 'data': data})
        else:
            return jsonify({'error': 'No valid JSON data found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_json_history')
def get_json_history():
    path = os.path.join(os.path.dirname(__file__), '..', 'photos', 'detection_results.json')
    if not os.path.exists(path):
        return jsonify({'error': 'JSON data not found'}), 404
    try:
        all_lines = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_lines.append(line.strip())
        if not all_lines:
            return jsonify({'error': 'No valid JSON data found'}), 404
        records = []
        start_index = len(all_lines) - 1 if len(all_lines) > 1 else 0
        for i in range(start_index, -1, -1):
            try:
                data = json.loads(all_lines[i])
                records.append(data)
                if len(records) >= 10:
                    break
            except:
                continue
        return jsonify({'success': True, 'data': records})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_segmentation')
def get_segmentation():
    path = image_paths['segmentation']
    if path is None or not os.path.exists(path):
        return jsonify({'error': 'Segmentation image not found'}), 404
    return send_file(path, mimetype='image/jpeg')

@app.route('/set_segmentation_path', methods=['POST'])
def set_segmentation_path():
    data = request.json
    new_path = data.get('path')
    if new_path and os.path.exists(new_path):
        image_paths['segmentation'] = new_path
        return jsonify({'success': True, 'message': 'Segmentation path updated'})
    return jsonify({'success': False, 'message': 'Invalid segmentation path'})

@app.route('/update_segmentation', methods=['POST'])
def update_segmentation():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    segmentation_dir = os.path.join(project_root, 'photos', 'segmentation')
    if not os.path.exists(segmentation_dir):
        return jsonify({'success': False, 'message': 'Segmentation directory not found'})
    image_files = [f for f in os.listdir(segmentation_dir) if f.lower().endswith(('.jpg', '.png'))]
    if not image_files:
        return jsonify({'success': False, 'message': 'No segmentation images found'})
    latest_image_file = max(image_files, key=lambda f: os.path.getmtime(os.path.join(segmentation_dir, f)))
    latest_image_path = os.path.join(segmentation_dir, latest_image_file)
    image_paths['segmentation'] = latest_image_path
    return jsonify({'success': True, 'path': '/get_segmentation'})

@app.route('/get_segmentation_stats')
def get_segmentation_stats():
    try:
        file_path = os.path.join(os.path.dirname(__file__), '..', 'photos', 'segmentation_result.json')
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'Segmentation result file not found'}), 404
        if os.path.getsize(file_path) == 0:
            return jsonify({'success': False, 'error': 'File is empty'}), 404
        last_line = ""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    last_line = line.strip()
        if not last_line:
            return jsonify({'success': False, 'error': 'No valid data found'}), 404
        data = json.loads(last_line)
        counts = data.get('counts', {})
        areas = data.get('areas', {})
        categories = {
            '方管类': ['0', '1', '2'],
            '圆管类': ['3', '4', '5'],
            '钢板类': ['6'],
            '镀锌件': ['7', '8', '9', '10'],
            '钢筋': ['11', '12'],
            '管道件': ['13'],
            '连接件': ['14'],
            '打孔废料': ['15']
        }
        stats = {}
        for category, indices in categories.items():
            count = sum(int(counts.get(idx, 0)) for idx in indices)
            area = sum(float(areas.get(idx, 0)) for idx in indices)
            stats[category] = {'count': count, 'area': area}
        return jsonify({
            'success': True,
            'stats': stats,
            'total_count': data.get('total_count', 0),
            'total_area': data.get('total_area', 0)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== 【稳定版】RTSP视频流（永不崩溃）====================
import threading
RTSP_URL = 'rtsp://admin:@192.168.3.10:554/live/ch00_0'
latest_frame = None
frame_lock = threading.Lock()
keep_capturing = True

def rtsp_background_worker():
    global latest_frame
    cap = None
    while keep_capturing:
        try:
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, frame = cap.read()
            if ret:
                with frame_lock:
                    latest_frame = frame.copy()
            else:
                cap.release()
                cap = None
                time.sleep(0.5)
        except:
            cap = None
            time.sleep(0.5)

threading.Thread(target=rtsp_background_worker, daemon=True).start()

def generate_stream():
    global latest_frame
    while True:
        if latest_frame is None:
            time.sleep(0.01)
            continue
        with frame_lock:
            f = latest_frame.copy()
        try:
            _, jpeg = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 50])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        except:
            continue

@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ==================== 【删除所有冲突接口】 ====================
# 旧接口全部删除，不再提供
# /get_live_frame、/get_live_stream 已彻底移除

@app.route('/check_camera_status')
def check_camera_status():
    try:
        cam = get_camera()
        exists = cam.check_camera_existence([CAMERA_CONFIG["channel"]])
        return jsonify({
            'success': True,
            'camera_online': exists.get(CAMERA_CONFIG["channel"], False),
            'channel': CAMERA_CONFIG["channel"],
            'ip': CAMERA_CONFIG["camera_ip"]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)