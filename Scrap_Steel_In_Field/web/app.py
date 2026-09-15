from flask import Flask, send_from_directory, jsonify, request, send_file, render_template, Response
import time
import os
import json
import sys
import cv2
import numpy as np

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from get_image.get_img import Camera
from config.camera_config import CAMERA_CONFIG

app = Flask(__name__)

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

# 为每个图片类型维护独立的路径变量
image_paths = {
    'original': None,  # 原始图片路径
    'detection': None,  # 检测结果图片路径
    'pie_chart': None,  # 饼图路径
    'json_data': None   # JSON数据文件路径
}

@app.route('/')
def index():
    return render_template('index_bak.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/get_image')
def get_image():
    path = image_paths['original']
    if path is None or not os.path.exists(path):
        return jsonify({'error': 'Image not found'}), 404
    return send_file(path, mimetype='image/jpeg')

@app.route('/get_detection')
def get_detection():
    path = image_paths['detection']
    if path is None or not os.path.exists(path):
        return jsonify({'error': 'Detection image not found'}), 404
    return send_file(path, mimetype='image/jpeg')

@app.route('/get_pie_chart')
def get_pie_chart():
    path = image_paths['pie_chart']
    if path is None or not os.path.exists(path):
        return jsonify({'error': 'Pie chart not found'}), 404
    return send_file(path, mimetype='image/png')

@app.route('/get_json_data')
def get_json_data():
    path = image_paths['json_data']
    if path is None or not os.path.exists(path):
        return jsonify({'error': 'JSON data not found'}), 404
    
    try:
        # 读取文件最后一行
        last_line = ""
        with open(path, 'r', encoding='utf-8') as f:
            # 从文件末尾读取数据
            for line in f:
                if line.strip():  # 忽略空行
                    last_line = line.strip()
        
        # 如果找到了有效的最后一行，解析它
        if last_line:
            try:
                data = json.loads(last_line)
                return jsonify({
                    'success': True,
                    'data': data
                })
            except json.JSONDecodeError:
                return jsonify({
                    'error': 'Failed to parse JSON data'
                }), 500
        else:
            return jsonify({
                'error': 'No valid JSON data found in file'
            }), 404
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/get_json_history')
def get_json_history():
    path = image_paths['json_data']
    if path is None or not os.path.exists(path):
        return jsonify({'error': 'JSON data not found'}), 404
    
    try:
        # 读取文件的所有行
        all_lines = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # 忽略空行
                    all_lines.append(line.strip())
        
        # 如果文件为空，返回错误
        if not all_lines:
            return jsonify({
                'error': 'No valid JSON data found in file'
            }), 404
        
        # 倒序并从倒数第二条开始（如果只有一条则从第一条开始）
        records = []
        
        # 从倒数第二条开始，或者从第一条开始（如果只有一条）
        start_index = len(all_lines) - 2 if len(all_lines) > 1 else 0
            
        # 从选定索引倒序读取
        for i in range(start_index, -1, -1):
            try:
                data = json.loads(all_lines[i])
                records.append(data)
                if len(records) >= 10:  # 最多取10条
                    break
            except json.JSONDecodeError:
                continue
                
        return jsonify({
            'success': True,
            'data': records
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/update_image', methods=['POST'])
def update_image():
    path = image_paths['original']
    if path is None:
        return jsonify({
            'success': False,
            'message': 'No image available'
        })
    return jsonify({
        'success': True,
        'path': '/get_image'
    })

@app.route('/set_image_path', methods=['POST'])
def set_image_path():
    data = request.json
    new_path = data.get('path')
    if new_path and os.path.exists(new_path):
        image_paths['original'] = new_path
        return jsonify({
            'success': True,
            'message': 'Image path updated successfully'
        })
    return jsonify({
        'success': False,
        'message': 'Invalid image path'
    })

@app.route('/set_detection_path', methods=['POST'])
def set_detection_path():
    data = request.json
    new_path = data.get('path')
    if new_path and os.path.exists(new_path):
        image_paths['detection'] = new_path
        return jsonify({
            'success': True,
            'message': 'Detection path updated successfully'
        })
    return jsonify({
        'success': False,
        'message': 'Invalid detection path'
    })

@app.route('/set_pie_chart_path', methods=['POST'])
def set_pie_chart_path():
    data = request.json
    new_path = data.get('path')
    if new_path and os.path.exists(new_path):
        image_paths['pie_chart'] = new_path
        return jsonify({
            'success': True,
            'message': 'Pie chart path updated successfully'
        })
    return jsonify({
        'success': False,
        'message': 'Invalid pie chart path'
    })

@app.route('/set_json_data_path', methods=['POST'])
def set_json_data_path():
    data = request.json
    new_path = data.get('path')
    if new_path and os.path.exists(new_path):
        image_paths['json_data'] = new_path
        return jsonify({
            'success': True,
            'message': 'JSON data path updated successfully'
        })
    return jsonify({
        'success': False,
        'message': 'Invalid JSON data path'
    })

@app.route('/update_detection', methods=['POST'])
def update_detection():
    path = image_paths['detection']
    if path is None:
        return jsonify({
            'success': False,
            'message': 'No detection image available'
        })
    return jsonify({
        'success': True,
        'path': '/get_detection'
    })

@app.route('/update_pie_chart', methods=['POST'])
def update_pie_chart():
    path = image_paths['pie_chart']
    if path is None:
        return jsonify({
            'success': False,
            'message': 'No pie chart available'
        })
    return jsonify({
        'success': True,
        'path': '/get_pie_chart'
    })

@app.route('/update_json_data', methods=['POST'])
def update_json_data():
    path = image_paths['json_data']
    if path is None:
        return jsonify({
            'success': False,
            'message': 'No JSON data available'
        })
    return jsonify({
        'success': True,
        'path': '/get_json_data'
    })

@app.route('/update_json_history', methods=['POST'])
def update_json_history():
    path = image_paths['json_data']
    if path is None:
        return jsonify({
            'success': False,
            'message': 'No JSON data available'
        })
    return jsonify({
        'success': True,
        'path': '/get_json_history'
    })

@app.route('/vehicle_records/vehicles.json')
def serve_vehicles_json():
    # 返回上级目录中的vehicle_records/vehicles.json文件
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vehicle_records', 'vehicles.json')
    if os.path.exists(file_path):
        return send_file(file_path)
    else:
        return jsonify({'error': 'Vehicles JSON file not found'}), 404


# ==================== 实时视频流接口 ====================

# 默认图片路径（当摄像头不可用时显示）
def get_default_image():
    """生成默认提示图片"""
    # 创建一个黑色背景的图片，显示提示文字
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img.fill(30)  # 深灰色背景
    
    # 添加提示文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    text1 = "Camera Offline"
    text2 = "摄像头未连接"
    text3 = f"IP: {CAMERA_CONFIG.get('camera_ip', 'Unknown')}"
    
    # 计算文字位置（居中）
    (text_width1, _), _ = cv2.getTextSize(text1, font, 1, 2)
    (text_width2, _), _ = cv2.getTextSize(text2, font, 0.8, 2)
    (text_width3, _), _ = cv2.getTextSize(text3, font, 0.6, 1)
    
    x1 = (640 - text_width1) // 2
    x2 = (640 - text_width2) // 2
    x3 = (640 - text_width3) // 2
    
    # 绘制文字
    cv2.putText(img, text1, (x1, 200), font, 1, (255, 255, 255), 2)
    cv2.putText(img, text2, (x2, 250), font, 0.8, (200, 200, 200), 2)
    cv2.putText(img, text3, (x3, 300), font, 0.6, (150, 150, 150), 1)
    
    return img


@app.route('/get_live_frame')
def get_live_frame():
    """
    获取摄像头实时单帧图像
    用于车辆图像区域的实时显示
    摄像头离线时返回默认图片
    """
    try:
        cam = get_camera()
        frame = None
        
        # 如果有自定义RTSP路径，优先使用RTSP方式
        if CAMERA_CONFIG.get("rtsp_path"):
            frame = cam.get_camera_image_by_rtspurl(
                CAMERA_CONFIG["channel"], 
                CAMERA_CONFIG["rtsp_path"]
            )
        else:
            # 使用HTTP方式获取图像（更快更稳定）
            frame = cam.get_camera_image(CAMERA_CONFIG["channel"])
            if frame is None:
                # 如果HTTP方式失败，尝试RTSP方式
                frame = cam.get_camera_image_by_rtspurl(CAMERA_CONFIG["channel"])
        
        # 如果获取失败，使用默认图片
        if frame is None:
            frame = get_default_image()
        
        # 将OpenCV图像转换为JPEG格式
        _, buffer = cv2.imencode('.jpg', frame)
        response = Response(buffer.tobytes(), mimetype='image/jpeg')
        # 添加缓存控制头，防止浏览器缓存
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
        
    except Exception as e:
        # 发生异常时返回默认图片
        frame = get_default_image()
        _, buffer = cv2.imencode('.jpg', frame)
        response = Response(buffer.tobytes(), mimetype='image/jpeg')
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        return response


@app.route('/get_live_stream')
def get_live_stream():
    """
    MJPEG实时视频流
    用于连续播放实时视频
    摄像头离线时显示默认图片
    """
    def generate_frames():
        cam = get_camera()
        consecutive_errors = 0  # 连续错误计数
        max_consecutive_errors = 5  # 最大连续错误次数
        
        while True:
            try:
                frame = None
                
                # 如果有自定义RTSP路径，优先使用RTSP方式
                if CAMERA_CONFIG.get("rtsp_path"):
                    frame = cam.get_camera_image_by_rtspurl(
                        CAMERA_CONFIG["channel"],
                        CAMERA_CONFIG["rtsp_path"]
                    )
                else:
                    # 获取单帧图像
                    frame = cam.get_camera_image(CAMERA_CONFIG["channel"])
                    if frame is None:
                        frame = cam.get_camera_image_by_rtspurl(CAMERA_CONFIG["channel"])
                
                # 如果获取失败，使用默认图片
                if frame is None:
                    frame = get_default_image()
                    consecutive_errors += 1
                else:
                    consecutive_errors = 0  # 成功获取则重置错误计数
                
                # 编码为JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                # 使用MJPEG格式推送
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # 如果连续多次错误，降低刷新频率（节省资源）
                if consecutive_errors >= max_consecutive_errors:
                    time.sleep(2)  # 摄像头离线时每2秒刷新一次
                else:
                    time.sleep(0.2)  # 正常时约5fps
                    
            except Exception as e:
                print(f"视频流错误: {e}")
                # 发生异常时发送默认图片
                try:
                    frame = get_default_image()
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except:
                    pass
                time.sleep(2)
    
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/check_camera_status')
def check_camera_status():
    """
    检查摄像头连接状态
    """
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
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False) 
