from urllib import response

from flask import Flask, send_from_directory, jsonify, request, send_file, render_template, Response
import time
import os
import json
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import threading
import socket

matplotlib.use('Agg')  # 设置后端以支持无GUI环境
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from get_image.get_img import Camera
from config.camera_config import CAMERA_CONFIG

from inference.detection_inference_simplified import DetectionManager,simple_detect#from inference.detection_inference_simplified import DetectionManager,simple_detect#导入推理模块
from inference.scrap_segmentation import SegmentationManager,simple_segment#from inference.scrap_segmentation import SegmentationManager,simple_segment#用于目标检测
from config.detection_config import MODEL_CONFIG as DETECT_CONFIG#from config.detection_config import MODEL_CONFIG as DETECT_CONFIG#用于分割任务
from config.segment_config import MODEL_CONFIG as SEGMENT_CONFIG#from config.segment_config import MODEL_CONFIG as SEGMENT_CONFIG#导入检测模型的配置参数
#导入分割模型的配置参数，用 SEGMENT_CONFIG 来引用
#这些代码是为视频流添加 AI 视觉识别能力（检测+分割）所做的导入准备增强
# 导入车兜检测函数
from inference.detection_inference_simplified import detect_car_bucket as detect_car_bucket_raw

app = Flask(__name__, static_folder='static', template_folder='templates')

# 推理相关全局变量
inference_lock = threading.Lock()
detector = None
segmenter = None
last_inference_time = 0
inference_interval = 5  # 推理间隔（秒）

# 目录配置
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
photos_dir = os.path.join(PROJECT_ROOT, 'photos')
original_dir = os.path.join(photos_dir, 'original')
det_dir = os.path.join(photos_dir, 'dangerous_det')
seg_dir = os.path.join(photos_dir, 'segmentation')

# 确保目录存在
os.makedirs(original_dir, exist_ok=True)
os.makedirs(det_dir, exist_ok=True)
os.makedirs(seg_dir, exist_ok=True)

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
    """生成默认提示图片"""
    from PIL import Image, ImageDraw, ImageFont

    # 创建PIL图像
    img = Image.new('RGB', (640, 480), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)

    # 尝试加载中文字体
    try:
        # 尝试常见的Windows中文字体
        font_large = ImageFont.truetype("simhei.ttf", 32)
        font_medium = ImageFont.truetype("simhei.ttf", 24)
        font_small = ImageFont.truetype("simhei.ttf", 18)
    except:
        try:
            font_large = ImageFont.truetype("msyh.ttc", 32)
            font_medium = ImageFont.truetype("msyh.ttc", 24)
            font_small = ImageFont.truetype("msyh.ttc", 18)
        except:
            # 如果找不到中文字体，使用默认字体
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()

    text1 = "Camera Offline"
    text2 = "摄像头未连接"
    text3 = f"IP: {CAMERA_CONFIG.get('camera_ip', 'Unknown')}"

    # 计算文字位置（居中）
    bbox1 = draw.textbbox((0, 0), text1, font=font_large)
    bbox2 = draw.textbbox((0, 0), text2, font=font_medium)
    bbox3 = draw.textbbox((0, 0), text3, font=font_small)

    x1 = (640 - (bbox1[2] - bbox1[0])) // 2
    x2 = (640 - (bbox2[2] - bbox2[0])) // 2
    x3 = (640 - (bbox3[2] - bbox3[0])) // 2

    # 绘制文字
    draw.text((x1, 180), text1, fill=(255, 255, 255), font=font_large)
    draw.text((x2, 230), text2, fill=(200, 200, 200), font=font_medium)
    draw.text((x3, 280), text3, fill=(150, 150, 150), font=font_small)

    # 转换为OpenCV格式
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img_cv


# 为每个图片类型维护独立的路径变量
image_paths = {
    'original': None,  # 原始图片路径
    'detection': None,  # 检测结果图片路径
    'pie_chart': None,  # 饼图路径
    'json_data': None,  # JSON数据文件路径
    'segmentation': None  # 分割图片路径
}

# 最新推理结果缓存
latest_inference_results = {
    'detection': None,
    'segmentation': None
}


def initialize_inference_engines():
    """初始化推理引擎"""
    global detector, segmenter
    try:
        print("正在初始化危险检测引擎...")
        detector = DetectionManager(config=DETECT_CONFIG)
        if not detector.initialize(photos_dir):
            print("警告: 危险检测引擎初始化失败")

        print("正在初始化废钢分割引擎...")
        segmenter = SegmentationManager(config=SEGMENT_CONFIG)
        if not segmenter.initialize(photos_dir):
            print("警告: 废钢分割引擎初始化失败")

        print("推理引擎初始化完成！")
    except Exception as e:
        print(f"初始化推理引擎失败: {e}")
        import traceback
        traceback.print_exc()


def run_inference_loop():
    """后台推理循环，每隔5秒执行一次推理"""
    global last_inference_time,detector,segmenter#global last_inference_time,detector,segmenter

    while True:
        try:
            current_time = time.time()
            if current_time - last_inference_time >= inference_interval:
                with inference_lock:
                    last_inference_time = current_time
                    perform_inference()
            time.sleep(0.5)  # 减少轮询频率
        except Exception as e:
            print(f"推理循环出错: {e}")
            time.sleep(1)


def perform_inference():
    """执行一次完整的推理流程"""
    global detector, segmenter

    try:
        # 从摄像头获取图像
        frame = capture_image_from_camera()
        if frame is None:
            print("无法获取摄像头图像")
            return

        # 保存原始图像
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_path = os.path.join(original_dir, f"original_{timestamp}.jpg")
        cv2.imwrite(original_path, frame)
        image_paths['original'] = original_path

        print(f"[{datetime.now().strftime('%H:%M:%S')}] 执行推理...")

        # 运行危险检测
        if detector:
            try:
                detections, success = simple_detect(original_path)

                if success:
                    # 获取检测结果图片路径（simple_detect会保存检测结果）
                    if detections and len(detections) > 0 and 'result_path' in detections[0]:
                        det_path = detections[0]['result_path']
                        image_paths['detection'] = det_path

                    # 保存检测结果到JSON
                    counts = {}
                    for det in detections:
                        label = det.get('label', '')
                        counts[label] = counts.get(label, 0) + 1

                    det_json = {
                        'counts': counts,
                        'total_count': len(detections),
                        'timestamp': timestamp
                    }

                    json_path = os.path.join(photos_dir, 'detection_results.json')
                    with open(json_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(det_json, ensure_ascii=False) + '\n')

                    latest_inference_results['detection'] = det_json
                    print(f"  危险检测完成，检测到 {len(detections)} 个危险物")
                else:
                    # 保存空结果
                    det_json = {
                        'counts': {},
                        'total_count': 0,
                        'timestamp': timestamp
                    }
                    json_path = os.path.join(photos_dir, 'detection_results.json')
                    with open(json_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(det_json, ensure_ascii=False) + '\n')
                    latest_inference_results['detection'] = det_json
            except Exception as e:
                print(f"  危险检测出错: {e}")

        # 运行废钢分割
        if segmenter:
            try:
                result = simple_segment(original_path)

                if result and len(result) >= 5:
                    detections, mask, save_path, class_counts, area_stats = result

                    if mask is not None:
                        image_paths['segmentation'] = save_path

                        # 保存分割结果到JSON
                        seg_json = {
                            'counts': class_counts,
                            'areas': area_stats,
                            'total_count': sum(class_counts.values()) if class_counts else 0,
                            'total_area': sum(area_stats.values()) if area_stats else 0,
                            'timestamp': timestamp
                        }

                        json_path = os.path.join(photos_dir, 'segmentation_result.json')
                        with open(json_path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(seg_json, ensure_ascii=False) + '\n')

                        latest_inference_results['segmentation'] = seg_json
                        print(f"  废钢分割完成，检测到 {len(detections)} 个物体")
                    else:
                        # 保存空结果
                        seg_json = {
                            'counts': {},
                            'areas': {},
                            'total_count': 0,
                            'total_area': 0,
                            'timestamp': timestamp
                        }
                        json_path = os.path.join(photos_dir, 'segmentation_result.json')
                        with open(json_path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(seg_json, ensure_ascii=False) + '\n')
                        latest_inference_results['segmentation'] = seg_json
            except Exception as e:
                print(f"  废钢分割出错: {e}")

        print("  推理完成")

    except Exception as e:
        print(f"执行推理失败: {e}")
        import traceback
        traceback.print_exc()


def capture_image_from_camera():
    """从摄像头捕获图像"""
    try:
        cam = get_camera()
        frame = None
        if CAMERA_CONFIG.get("rtsp_path"):
            frame = cam.get_camera_image_by_rtspurl(
                CAMERA_CONFIG["channel"],
                CAMERA_CONFIG["rtsp_path"]
            )
        else:
            frame = cam.get_camera_image(CAMERA_CONFIG["channel"])
            if frame is None:
                frame = cam.get_camera_image_by_rtspurl(CAMERA_CONFIG["channel"])
        return frame
    except Exception as e:
        print(f"捕获图像失败: {e}")
        return None


# 网页路由
@app.route('/')
def index():
    # 在页面加载时尝试更新分割图片路径
    update_segmentation()
    # 在页面加载时尝试更新实时监控图片路径
    update_image()
    # 在页面加载时尝试更新危险品检测图片路径
    update_detection()
    return render_template('index_bak.html')


# 静态文件路由
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


'''
实时监控图像
'''


# 上传图片并处理
@app.route('/get_image')
def get_image():
    path = image_paths['original']
    if path is None or not os.path.exists(path):
        return jsonify({'error': 'Image not found'}), 404
    return send_file(path, mimetype='image/jpeg')


# 设置图片路径
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


# 更新图片
@app.route('/update_image', methods=['POST'])
def update_image():
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_dir = os.path.join(project_root, 'photos', 'original')

    if not os.path.exists(image_dir):
        return jsonify({
            'success': False,
            'message': 'Original image directory not found'
        })

    # 查找最新的图片文件 (假设是 .jpg 或 .png)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]

    if not image_files:
        return jsonify({
            'success': False,
            'message': 'No original images found in directory'
        })

    # 按修改时间排序，找到最新的文件
    latest_image_file = max(image_files, key=lambda f: os.path.getmtime(os.path.join(image_dir, f)))
    latest_image_path = os.path.join(image_dir, latest_image_file)

    # 获取当前存储的图片路径
    current_image_path = image_paths['original']

    # 如果最新的图片与当前显示的图片不同，更新路径并返回成功
    if latest_image_path != current_image_path:
        image_paths['original'] = latest_image_path
        return jsonify({
            'success': True,
            'path': '/get_image'
        })
    else:
        # 即使没有新的图片，也确保image_paths['original']是最新的路径
        image_paths['original'] = latest_image_path
        return jsonify({
            'success': False,
            'message': 'No new original image available'
        })


'''
危险品图像
'''


# 获取检测结果图片
@app.route('/get_detection')
def get_detection():
    path = image_paths['detection']
    if path is None or not os.path.exists(path):
        return jsonify({'error': 'Detection image not found'}), 404
    return send_file(path, mimetype='image/jpeg')


# 设置检测结果图片路径
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


# 更新图片
@app.route('/update_detection', methods=['POST'])
def update_detection():
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    detection_dir = os.path.join(project_root, 'photos', 'dangerous_det')

    if not os.path.exists(detection_dir):
        return jsonify({
            'success': False,
            'message': 'Detection image directory not found'
        })

    # 查找最新的图片文件 (假设是 .jpg 或 .png)
    image_files = [f for f in os.listdir(detection_dir) if f.lower().endswith(('.jpg', '.png'))]

    if not image_files:
        return jsonify({
            'success': False,
            'message': 'No detection images found in directory'
        })

    # 按修改时间排序，找到最新的文件
    latest_image_file = max(image_files, key=lambda f: os.path.getmtime(os.path.join(detection_dir, f)))
    latest_image_path = os.path.join(detection_dir, latest_image_file)

    # 获取当前存储的图片路径
    current_image_path = image_paths['detection']

    # 如果最新的图片与当前显示的图片不同，更新路径并返回成功
    if latest_image_path != current_image_path:
        image_paths['detection'] = latest_image_path
        return jsonify({
            'success': True,
            'path': '/get_detection'
        })
    else:
        # 即使没有新的图片，也确保image_paths['detection']是最新的路径
        image_paths['detection'] = latest_image_path
        return jsonify({
            'success': False,
            'message': 'No new detection image available'
        })


# 更新JSON数据
@app.route('/update_json_data', methods=['POST'])
def update_json_data():
    path = os.path.join(os.path.dirname(__file__), '..', 'photos', 'detection_results.json')
    if path is None:
        return jsonify({
            'success': False,
            'message': 'No JSON data available'
        })
    return jsonify({
        'success': True,
        'path': '/get_json_data'
    })


# 更新JSON历史数据
@app.route('/update_json_history', methods=['POST'])
def update_json_history():
    path = os.path.join(os.path.dirname(__file__), '..', 'photos', 'detection_results.json')
    if path is None:
        return jsonify({
            'success': False,
            'message': 'No JSON data available'
        })
    return jsonify({
        'success': True,
        'path': '/get_json_history'
    })


# 获取最新危险物数量
@app.route('/get_latest_danger_counts')
def get_latest_danger_counts():
    path = os.path.join(os.path.dirname(__file__), '..', 'photos', 'detection_results.json')
    if path is None or not os.path.exists(path):
        return jsonify({'error': 'JSON data not found'}), 404

    try:
        # 读取文件最后一行
        last_line = ""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    last_line = line.strip()

        if last_line:
            data = json.loads(last_line)
            counts = data.get('counts', {})
            # 确保四种危险物都有默认值0
            danger_types = ['GasCyl1', 'GasCyl2', 'FireExt1', 'FireExt2']
            result = {dtype: counts.get(dtype, 0) for dtype in danger_types}
            return jsonify({'success': True, 'counts': result})
        else:
            return jsonify({'success': False, 'message': 'No valid data found'})
    except Exception as e:
        return jsonify({'success': False, 'message': '读取JSON数据失败', 'error': str(e)})


# 获取危险物总数
@app.route('/get_danger_total')
def get_danger_total():
    path = os.path.join(os.path.dirname(__file__), '..', 'photos', 'detection_results.json')
    if path is None or not os.path.exists(path):
        return jsonify({'error': 'JSON data not found'}), 404

    try:
        # 读取文件最后一行
        last_line = ""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    last_line = line.strip()

        if last_line:
            data = json.loads(last_line)
            counts = data.get('counts', {})
            # 计算总数 - 所有危险物类型的数量之和
            total_count = sum(counts.values()) if counts else 0

            return jsonify({
                'success': True,
                'total_count': total_count
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No valid data found'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': '读取JSON数据失败',
            'error': str(e)
        })


# 检查segmentation_result.json文件修改时间
@app.route('/check_segmentation_file_time')
def check_segmentation_file_time():
    try:
        file_path = os.path.join(os.path.dirname(__file__), '..', 'photos', 'segmentation_result.json')
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'error': 'File not found'
            }), 404

        last_modified = os.path.getmtime(file_path)
        return jsonify({
            'success': True,
            'last_modified': last_modified
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# 检查detection_results.json文件修改时间
@app.route('/check_danger_file_time')
def check_danger_file_time():
    try:
        file_path = os.path.join(os.path.dirname(__file__), '..', 'photos', 'detection_results.json')
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'error': 'File not found'
            }), 404

        last_modified = os.path.getmtime(file_path)

        return jsonify({
            'success': True,
            'last_modified': last_modified
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# 检查detection_results.json文件是否有新数据
@app.route('/check_danger_file_new_data')
def check_danger_file_new_data():
    try:
        file_path = os.path.join(os.path.dirname(__file__), '..', 'photos', 'detection_results.json')
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'error': 'File not found'
            }), 404

        # 获取文件的最后一行，检查是否有新的JSON数据
        last_line = ""
        data_count = 0

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # 忽略空行
                    data_count += 1
                    last_line = line.strip()

        if not last_line:
            return jsonify({
                'success': False,
                'error': 'No data in file'
            }), 404

        # 尝试解析最后一行JSON
        try:
            data = json.loads(last_line)
            return jsonify({
                'success': True,
                'last_line_data': data,
                'data_count': data_count,
                'last_modified': os.path.getmtime(file_path)
            })
        except json.JSONDecodeError:
            return jsonify({
                'success': False,
                'error': 'Invalid JSON format'
            }), 500

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# 检查segmentation_result.json文件是否有新数据
@app.route('/check_segmentation_file_new_data')
def check_segmentation_file_new_data():
    try:
        file_path = os.path.join(os.path.dirname(__file__), '..', 'photos', 'segmentation_result.json')
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'error': 'File not found'
            }), 404

        # 获取文件的最后一行，检查是否有新的JSON数据
        last_line = ""
        data_count = 0

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # 忽略空行
                    data_count += 1
                    last_line = line.strip()

        if not last_line:
            return jsonify({
                'success': False,
                'error': 'No data in file'
            }), 404

        # 尝试解析最后一行JSON
        try:
            data = json.loads(last_line)
            return jsonify({
                'success': True,
                'last_line_data': data,
                'data_count': data_count,
                'last_modified': os.path.getmtime(file_path)
            })
        except json.JSONDecodeError:
            return jsonify({
                'success': False,
                'error': 'Invalid JSON format'
            }), 500

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# 辅助函数：确保目录存在
def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


# 获取机械臂状态
@app.route('/arm_status')
def get_arm_status():
    try:
        # 调整路径为：Scrap_Steel_In_Field\photos\arm_status.json
        status_path = os.path.join(os.path.dirname(__file__), '..', 'photos', 'arm_status.json')
        with open(status_path, 'r', encoding='utf-8') as f:
            status_data = json.load(f)
            # 确保返回布尔值（例如从文件中读取的状态应为 true/false）
            return jsonify({'arm1_status': status_data.get('arm1_status', False)})  # 默认未启动
    except Exception as e:
        # 异常时返回 arm1_status 为 null，表示未知
        return jsonify({'arm1_status': None, 'error': str(e)}), 500


"""
车辆信息
"""


# 设置JSON数据文件路径
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


# 提供车辆JSON数据文件
@app.route('/vehicle_records/vehicles.json')
def serve_vehicles_json():
    # 返回上级目录中的vehicle_records/vehicles.json文件
    file_path = os.path.join(os.path.dirname(__file__), '..', 'vehicle_records', 'vehicles.json')
    if os.path.exists(file_path):
        return send_file(file_path)
    else:
        return jsonify({'error': 'Vehicles JSON file not found'}), 404


# 获取车辆JSON数据，只返回最后一条数据，即最新数据
@app.route('/get_json_data')
def get_json_data():
    path = os.path.join(os.path.dirname(__file__), '..', 'photos', 'detection_results.json')
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


# 获取车辆JSON数据，返回所有数据，最多10条
@app.route('/get_json_history')
def get_json_history():
    path = os.path.join(os.path.dirname(__file__), '..', 'photos', 'detection_results.json')
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
        start_index = len(all_lines) - 1 if len(all_lines) > 1 else 0

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


# 获取分割图片
@app.route('/get_segmentation')
def get_segmentation():
    path = image_paths['segmentation']
    if path is None or not os.path.exists(path):
        return jsonify({'error': 'Segmentation image not found'}), 404
    return send_file(path, mimetype='image/jpeg')


# 设置分割图片路径
@app.route('/set_segmentation_path', methods=['POST'])
def set_segmentation_path():
    data = request.json
    new_path = data.get('path')
    if new_path and os.path.exists(new_path):
        image_paths['segmentation'] = new_path
        return jsonify({
            'success': True,
            'message': 'Segmentation path updated successfully'
        })
    return jsonify({
        'success': False,
        'message': 'Invalid segmentation path'
    })


# 更新分割图片
@app.route('/update_segmentation', methods=['POST'])
def update_segmentation():
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    segmentation_dir = os.path.join(project_root, 'photos', 'segmentation')

    if not os.path.exists(segmentation_dir):
        return jsonify({
            'success': False,
            'message': 'Segmentation directory not found'
        })

    # 查找最新的图片文件 (假设是 .jpg 或 .png)
    image_files = [f for f in os.listdir(segmentation_dir) if f.lower().endswith(('.jpg', '.png'))]

    if not image_files:
        return jsonify({
            'success': False,
            'message': 'No segmentation images found in directory'
        })

    # 按修改时间排序，找到最新的文件
    latest_image_file = max(image_files, key=lambda f: os.path.getmtime(os.path.join(segmentation_dir, f)))
    latest_image_path = os.path.join(segmentation_dir, latest_image_file)

    # 获取当前存储的图片路径
    current_image_path = image_paths['segmentation']

    # 如果最新的图片与当前显示的图片不同，更新路径并返回成功
    if latest_image_path != current_image_path:
        image_paths['segmentation'] = latest_image_path
        return jsonify({
            'success': True,
            'path': '/get_segmentation'
        })
    else:
        # 即使没有新的图片，也确保image_paths['segmentation']是最新的路径
        image_paths['segmentation'] = latest_image_path
        return jsonify({
            'success': False,
            'message': 'No new segmentation image available'
        })


# 获取分割结果统计
@app.route('/get_segmentation_stats')
def get_segmentation_stats():
    try:
        # 读取segmentation_result.json文件的最后一行
        file_path = os.path.join(os.path.dirname(__file__), '..', 'photos', 'segmentation_result.json')

        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'error': 'Segmentation result file not found'
            }), 404

        # 检查文件是否为空
        if os.path.getsize(file_path) == 0:
            return jsonify({
                'success': False,
                'error': 'File is empty'
            }), 404

        last_line = ""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    last_line = line.strip()

        if not last_line:
            return jsonify({
                'success': False,
                'error': 'No valid data found in file'
            }), 404

        try:
            data = json.loads(last_line)
        except json.JSONDecodeError as e:
            return jsonify({
                'success': False,
                'error': f'Invalid JSON format: {str(e)}'
            }), 500

        counts = data.get('counts', {})
        areas = data.get('areas', {})

        # 定义分类映射
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

        # 计算每个类别的数量和面积
        stats = {}
        for category, indices in categories.items():
            count = sum(int(counts.get(idx, 0)) for idx in indices)
            area = sum(float(areas.get(idx, 0)) for idx in indices)
            stats[category] = {
                'count': count,
                'area': area
            }

        response_data = {
            'success': True,
            'stats': stats,
            'total_count': data.get('total_count', 0),
            'total_area': data.get('total_area', 0)
        }
        return jsonify(response_data)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ==================== 实时视频流接口 ====================

# RTSP视频流配置
RTSP_URL = 'rtsp://admin:@192.168.3.10:554/live/ch00_0'

# 全局变量：用于存储最新的视频帧，实现"采集"与"编码发送"解耦
_latest_frame = None
_frame_lock = threading.Lock()
_frame_ready = threading.Event()  # 帧准备就绪事件


def _capture_thread(rtsp_url):
    """后台采集线程：只负责从RTSP流获取最新画面"""
    global _latest_frame

    # 使用FFmpeg选项优化延迟
    # - rtsp_transport tcp: 使用TCP传输更稳定
    # - buffer_size: 减少缓冲区
    # - max_delay: 最大延迟设为0
    # - stimeout:  socket超时
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    # 关键优化1：设置缓冲区大小必须在循环开始前，且设为1确保只取最新帧
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 设置低分辨率预览以减少带宽和编码延迟（可选，根据需要调整）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 设置帧率
    cap.set(cv2.CAP_PROP_FPS, 25)

    if not cap.isOpened():
        print(f"无法打开RTSP流: {rtsp_url}")
        return

    # 通知第一帧已准备好
    first_frame_received = False

    while True:
        ret, frame = cap.read()
        if ret:
            with _frame_lock:
                _latest_frame = frame
            if not first_frame_received:
                _frame_ready.set()  # 通知帧已就绪
                first_frame_received = True
        else:
            # 连接异常时稍作等待，避免空转占用CPU
            time.sleep(0.01)  # 减少等待时间从0.1秒到0.01秒


def gen_rtsp_frames():
    """生成RTSP视频流帧"""
    global _latest_frame

    # 启动后台采集线程（确保只启动一次，或使用单例模式管理）
    if not hasattr(gen_rtsp_frames, 'thread_started'):
        t = threading.Thread(target=_capture_thread, args=(RTSP_URL,), daemon=True)
        t.start()
        gen_rtsp_frames.thread_started = True
        # 等待第一帧到达（使用事件等待替代固定1秒延迟）
        _frame_ready.wait(timeout=2.0)  # 最多等待2秒

    while True:
        if _latest_frame is None:
            time.sleep(0.001)  # 减少到1ms等待
            continue

        # 获取当前最新帧的副本
        with _frame_lock:
            frame = _latest_frame.copy()

        try:
            # 编码为JPEG，提高质量到75以获得更好的视觉效果，同时保持合理带宽
            # 使用优化编码参数
            encode_params = [
                cv2.IMWRITE_JPEG_QUALITY, 75,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1
            ]
            _, buffer = cv2.imencode('.jpg', frame, encode_params)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Cache-Control: no-cache, no-store, must-revalidate\r\n'
                   b'Pragma: no-cache\r\n'
                   b'Expires: 0\r\n\r\n' + frame_bytes + b'\r\n')

            # 控制帧率约20-25fps，避免过度占用带宽
            time.sleep(0.04)  # 40ms = 25fps

        except Exception as e:
            print(f"RTSP视频流编码错误: {e}")
            continue


@app.route('/video_feed')
def video_feed():
    """RTSP实时视频流路由"""
#response=Response(gen_rtsp_frames(),
                      #mimetype='multipart/x-mixed-replace;boundary=frame')#(可以在普通的网页浏览器中直接打开观看，且无需安装任何插件。)
    # 添加缓存控制头，防止代理和浏览器缓存
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['X-Accel-Buffering'] = 'no'  # 禁用Nginx缓冲
    return response


@app.route('/get_live_frame')
def get_live_frame():
    """获取摄像头实时单帧图像"""
    try:
        cam = get_camera()
        frame = None
        if CAMERA_CONFIG.get("rtsp_path"):
            frame = cam.get_camera_image_by_rtspurl(
                CAMERA_CONFIG["channel"],
                CAMERA_CONFIG["rtsp_path"]
            )
        else:
            frame = cam.get_camera_image(CAMERA_CONFIG["channel"])
            if frame is None:
                frame = cam.get_camera_image_by_rtspurl(CAMERA_CONFIG["channel"])
        if frame is None:
            frame = get_default_image()
        _, buffer = cv2.imencode('.jpg', frame)
        response = Response(buffer.tobytes(), mimetype='image/jpeg')
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    except Exception as e:
        frame = get_default_image()
        _, buffer = cv2.imencode('.jpg', frame)
        response = Response(buffer.tobytes(), mimetype='image/jpeg')
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        return response


@app.route('/get_live_stream')
def get_live_stream():
    """MJPEG实时视频流"""

    def generate_frames():
        cam = get_camera()
        consecutive_errors = 0
        max_consecutive_errors = 5
        while True:
            try:
                frame = None
                if CAMERA_CONFIG.get("rtsp_path"):
                    frame = cam.get_camera_image_by_rtspurl(
                        CAMERA_CONFIG["channel"],
                        CAMERA_CONFIG["rtsp_path"]
                    )
                else:
                    frame = cam.get_camera_image(CAMERA_CONFIG["channel"])
                    if frame is None:
                        frame = cam.get_camera_image_by_rtspurl(CAMERA_CONFIG["channel"])
                if frame is None:
                    frame = get_default_image()
                    consecutive_errors += 1
                else:
                    consecutive_errors = 0
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                if consecutive_errors >= max_consecutive_errors:
                    time.sleep(2)
                else:
                    time.sleep(0.2)
            except Exception as e:
                print(f"视频流错误: {e}")
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
    """检查摄像头连接状态"""
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


# ==================== 机械臂控制 ====================

# 机械臂相关全局变量
ARM_SERVER_PORT = 8770  # 机械臂连接端口（与 test_arm_online.py 一致）
arm_server_socket = None
arm_client_socket = None
arm_connected = False  # 机械臂是否已连接
arm_server_running = False  # 服务器线程是否在运行
arm_cycle_active = False  # 当前是否正在执行抓取循环
arm_lock = threading.Lock()
arm_last_grasp_points = []  # 最近一次计算的抓取点
arm_ready_event = threading.Event()  # 服务器就绪事件

# 变换矩阵文件路径
TRANSFORM_MATRIX_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'test_suite', 'transform_matrix.json'
)


def _sort_corners(box):
    """按左上、右上、右下、左下排序四个角点"""
    sorted_by_y = box[box[:, 1].argsort()]
    top_two = sorted_by_y[:2]
    bottom_two = sorted_by_y[2:]
    top_left = top_two[top_two[:, 0].argmin()]
    top_right = top_two[top_two[:, 0].argmax()]
    bottom_left = bottom_two[bottom_two[:, 0].argmin()]
    bottom_right = bottom_two[bottom_two[:, 0].argmax()]
    return np.array([top_left, top_right, bottom_right, bottom_left])


def transform_corners_to_world(corners, matrix_file=None):
    """将像素坐标转换为世界坐标"""
    if matrix_file is None:
        matrix_file = TRANSFORM_MATRIX_PATH

    if not os.path.exists(matrix_file):
        print(f"[机械臂] 错误: 变换矩阵文件不存在 {matrix_file}")
        return None

    try:
        with open(matrix_file, 'r') as f:
            matrix_list = json.load(f)
        transform_matrix = np.array(matrix_list)

        world_corners = []
        for corner in corners:
            pixel_homogeneous = np.array([corner[0], corner[1], 1])
            transformed_point = np.dot(transform_matrix, pixel_homogeneous)
            world_point = transformed_point[:2] / transformed_point[2]
            world_corners.append(tuple(world_point))

        return world_corners
    except Exception as e:
        print(f"[机械臂] 坐标转换错误: {e}")
        return None


def calculate_grasp_points(world_corners):
    if len(world_corners) != 4:
        return None

    try:
        corners = np.array(world_corners)
        center = np.mean(corners, axis=0)

        centered = corners - center
        cov = np.dot(centered.T, centered)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        main_dir = eigenvectors[:, np.argmax(eigenvalues)]
        secondary_dir = eigenvectors[:, np.argmin(eigenvalues)]

        main_dist = np.abs(np.dot(centered, main_dir))
        sec_dist = np.abs(np.dot(centered, secondary_dir))
        length = 2 * np.max(main_dist)
        width = 2 * np.max(sec_dist)

        grasp_points = []
        for ratio in [0.25, 0.5, 0.75]:
            offset = (ratio - 0.5) * length * main_dir
            point = center + offset
            grasp_points.append(tuple(point))

        return grasp_points

    except Exception as e:
        print("Grasp points error:", e)
        return None


def compute_grasp_points_from_corners(corners):
    """从四个车兜角点计算三个机械臂抓取点"""
    try:
        # 1. 确保角点已排序
        if corners.shape[0] == 4:
            corners = _sort_corners(corners)
        print(f"[机械臂] 车兜角点 (像素): {corners.tolist()}")

        # 2. 转换到世界坐标
        world_corners = transform_corners_to_world(corners)
        if world_corners is None:
            return None

        print(f"[机械臂] 车兜角点 (世界坐标): {world_corners}")

        # 3. 计算三个抓取点
        grasp_points = calculate_grasp_points(world_corners)
        if grasp_points is None:
            return None

        print(f"[机械臂] 计算的抓取点: {grasp_points}")
        return grasp_points

    except Exception as e:
        print(f"[机械臂] 计算抓取点过程出错: {e}")
        return None


def _arm_server_thread():
    """机械臂TCP服务器线程，持续运行，保持与机械臂的连接"""
    global arm_server_socket, arm_client_socket, arm_connected, arm_cycle_active

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server_socket.bind(('0.0.0.0', ARM_SERVER_PORT))
        server_socket.listen(1)
        print(f"\n[机械臂] 服务器已启动，监听端口 {ARM_SERVER_PORT}")
        print("[机械臂] 等待机械臂连接...")
        arm_ready_event.set()  # 标记服务器已就绪

        # 持续接受连接（支持断线重连）
        while True:
            try:
                client_sock, addr = server_socket.accept()
                arm_client_socket = client_sock
                with arm_lock:
                    arm_connected = True
                print(f"[机械臂] 机械臂已连接: {addr}")

                # 发送 START 指令
                client_sock.sendall("START".encode())
                print("[机械臂] 已发送: START")

                # 进入消息循环，只监听机械臂状态
                while True:
                    try:
                        client_sock.settimeout(1.0)
                        data = client_sock.recv(1024)

                        if not data:
                            print("[机械臂] 机械臂断开连接")
                            break

                        msg = data.decode('utf-8', errors='ignore').strip()
                        print(f"[机械臂] 收到: [{msg}]")

                        if msg == "yes_grab":
                            # yes_grab 只是心跳/连接正常信号，不触发任何操作
                            print("[机械臂] 连接正常")

                        elif msg in ["A", "B"]:
                            print(f"[机械臂] 机械臂确认抓取完成: {msg}")

                        elif msg == "C":
                            print("[机械臂] 机械臂确认全部抓取完成: C")
                            with arm_lock:
                                arm_cycle_active = False
                            update_arm_status_file(False)
                            print("[机械臂] 状态已释放，等待下一次点击")

                    except socket.timeout:
                        continue

                # 机械臂断开，清理状态
                with arm_lock:
                    arm_connected = False
                    arm_cycle_active = False
                update_arm_status_file(False)

            except Exception as e:
                print(f"[机械臂] 接受连接出错: {e}")
                time.sleep(1)

    except Exception as e:
        print(f"[机械臂] 服务器错误: {e}")
    finally:
        server_socket.close()
        with arm_lock:
            arm_server_socket = None
            arm_client_socket = None
            arm_connected = False
        print("[机械臂] 服务器已关闭")


@app.route('/start_arm', methods=['POST'])#将核心运行代码与数据大屏相连)
def start_arm():
    """启动机械臂：拍照 → 计算抓取坐标 → 发送坐标给机械臂"""
    global arm_connected,arm_cycle_active,arm_client_socket#global arm_connected, arm_cycle_active, arm_client_socket

    with arm_lock:
        if arm_cycle_active:
            return jsonify({'success': False, 'message': '机械臂正在执行抓取循环，请等待完成'})

        if not arm_connected:
            return jsonify({'success': False, 'message': '机械臂未连接，请等待机械臂连接服务器'})

        client = arm_client_socket

    try:
        with arm_lock:
            arm_cycle_active = True

        # 1. 从摄像头拍照
        print("[机械臂] 正在拍照获取图像...")
        frame = capture_image_from_camera()
        if frame is None:
            with arm_lock:
                arm_cycle_active = False
            return jsonify({'success': False, 'message': '拍照失败，请检查摄像头连接'})

        # frame 是 BGR 格式，检测车兜需要 RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2. 检测车兜四个角点
        corners = detect_car_bucket_raw(image_rgb)
        if corners is None:
            with arm_lock:
                arm_cycle_active = False
            update_arm_status_file(False)
            return jsonify({'success': False, 'message': '未检测到车兜区域'})

        # 3. 通过角点计算抓取点（已包含像素→世界坐标转换）
        grasp_points = compute_grasp_points_from_corners(corners)
        if not grasp_points:
            with arm_lock:
                arm_cycle_active = False
            update_arm_status_file(False)
            return jsonify({'success': False, 'message': '计算抓取点失败'})

        print(f"[机械臂] 计算了 {len(grasp_points)} 个抓取点: {grasp_points}")

        # 更新状态文件
        update_arm_status_file(True)

        # 4. 依次发送三个坐标点
        for i, (x, y) in enumerate(grasp_points):
            send_msg = "{:.1f},{:.1f},0;".format(x, y)
            time.sleep(0.5)
            try:
                client.sendall(send_msg.encode())
                print(f"[机械臂]   已发送坐标 {i + 1}: {send_msg}")
            except Exception as e:
                print(f"[机械臂]   发送失败: {e}")
                with arm_lock:
                    arm_cycle_active = False
                update_arm_status_file(False)
                return jsonify({'success': False, 'message': f'发送坐标失败: {str(e)}'})

        print("[机械臂] 所有坐标已发送完成，等待机械臂确认...")

        response = {
            'success': True,
            'message': '坐标已发送，等待机械臂抓取完成',
            'points': [{'x': round(x, 1), 'y': round(y, 1), 'z': 0} for x, y in grasp_points]
        }

        return jsonify(response)

    except Exception as e:
        print(f"[机械臂] 启动抓取循环出错: {e}")
        with arm_lock:
            arm_cycle_active = False
        update_arm_status_file(False)
        return jsonify({'success': False, 'message': f'启动失败: {str(e)}'})


@app.route('/stop_arm', methods=['POST'])
def stop_arm():
    """停止机械臂当前抓取循环（不关闭连接）"""
    global arm_cycle_active

    with arm_lock:
        if not arm_cycle_active:
            return jsonify({'success': False, 'message': '机械臂当前未在执行抓取循环'})

        arm_cycle_active = False
        update_arm_status_file(False)

    return jsonify({'success': True, 'message': '机械臂抓取循环已停止'})


def update_arm_status_file(is_running):
    """更新机械臂状态文件"""
    try:
        status_path = os.path.join(os.path.dirname(__file__), '..', 'photos', 'arm_status.json')
        with open(status_path, 'w', encoding='utf-8') as f:
            json.dump({'arm1_status': is_running}, f)
    except Exception as e:
        print(f"更新机械臂状态文件失败: {e}")


# ==================== 推理结果接口 ====================

@app.route('/get_latest_inference_results')
def get_latest_inference_results():
    """获取最新的推理结果"""
    return jsonify({
        'success': True,
        'detection': latest_inference_results['detection'],
        'segmentation': latest_inference_results['segmentation'],
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })


@app.route('/trigger_inference')
def trigger_inference():
    """手动触发一次推理"""
    global last_inference_time
    with inference_lock:
        last_inference_time = 0  # 重置时间，让推理循环立即执行
    return jsonify({
        'success': True,
        'message': '推理已触发'
    })


if __name__ == '__main__':
    # 初始化推理引擎
    initialize_inference_engines()

    # 启动后台推理线程
    inference_thread = threading.Thread(target=run_inference_loop, daemon=True)
    inference_thread.start()
    print("后台推理线程已启动")

    # 启动机械臂TCP服务器线程（持续运行）
    arm_server_thread = threading.Thread(target=_arm_server_thread, daemon=True)
    arm_server_thread.start()
    print("机械臂TCP服务器线程已启动")

    # 启动Flask应用
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)