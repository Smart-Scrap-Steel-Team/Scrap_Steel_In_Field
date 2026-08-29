from flask import Flask, send_from_directory, jsonify, request, send_file, render_template
import time
import os
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 设置后端以支持无GUI环境
import numpy as np
from datetime import datetime

app = Flask(__name__, static_folder='static', template_folder='templates')

# 为每个图片类型维护独立的路径变量
image_paths = {
    'original': None,  # 原始图片路径
    'detection': None,  # 检测结果图片路径
    'pie_chart': None,  # 饼图路径
    'json_data': None,   # JSON数据文件路径
    'segmentation': None  # 分割图片路径
}

# 网页路由
@app.route('/')
def index():
    # 在页面加载时尝试更新分割图片路径
    update_segmentation()
    # 在页面加载时尝试更新实时监控图片路径
    update_image()
    # 在页面加载时尝试更新危险品检测图片路径
    update_detection()
    return render_template('index.html')

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
            return jsonify({'success': True,'counts': result})
        else:
            return jsonify({'success': False,'message': 'No valid data found'})
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

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5003, use_reloader=False)