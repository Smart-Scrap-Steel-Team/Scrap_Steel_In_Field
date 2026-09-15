from flask import Flask, send_from_directory, jsonify, request, send_file, render_template
import time
import os
import json

app = Flask(__name__)

# 为每个图片类型维护独立的路径变量
image_paths = {
    'original': None,  # 原始图片路径
    'detection': None,  # 检测结果图片路径
    'pie_chart': None,  # 饼图路径
    'json_data': None   # JSON数据文件路径
}

@app.route('/')
def index():
    return render_template('index.html')

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

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False) 
