import os
import json
from flask import send_file, jsonify, request

class ImageService:
    """图片和JSON数据相关的服务逻辑。"""
    def __init__(self):
        # 维护各类图片的路径
        self.image_paths = {
            'original': None,      # 原始图片路径
            'detection': None,     # 检测结果图片路径
            'segmentation': None,  # 分割图片路径
            'json_data': None      # JSON数据文件路径
        }
        # 项目根目录（修正：使用当前工作目录，确保查找路径正确）
        self.project_root = os.path.abspath(os.getcwd())

    def get_image(self):
        # 获取原始图片
        path = self.image_paths['original']
        # 使用日志输出路径信息
        print(f"获取原始图片路径: {path}")
        if path is None or not os.path.exists(path):
            return jsonify({'error': 'Image not found'}), 404
        return send_file(path, mimetype='image/jpeg')

    def set_image_path(self):
        # 设置原始图片路径
        data = request.json
        new_path = data.get('path')
        if new_path and os.path.exists(new_path):
            self.image_paths['original'] = new_path
            return jsonify({'success': True, 'message': 'Image path updated'})
        return jsonify({'success': False, 'message': 'Invalid image path'})

    def update_image(self):
        # 查找最新的原始图片并更新路径
        image_dir = os.path.join(self.project_root, 'photos', 'original')
        print(f"查找最新的原始图片路径: {image_dir}")
        if not os.path.exists(image_dir):
            return jsonify({'success': False, 'message': 'Original image directory not found'})
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]
        if not image_files:
            return jsonify({'success': False, 'message': 'No original image found'})
        latest_image_file = max(image_files, key=lambda f: os.path.getmtime(os.path.join(image_dir, f)))
        latest_image_path = os.path.join(image_dir, latest_image_file)
        current_image_path = self.image_paths['original']
        if latest_image_path != current_image_path:
            self.image_paths['original'] = latest_image_path
            return jsonify({'success': True, 'path': '/get_image'})
        else:
            self.image_paths['original'] = latest_image_path
            return jsonify({'success': False, 'message': 'No new original image'})

    def get_detection(self):
        # 获取检测结果图片
        path = self.image_paths['detection']
        print(f"获取检测结果图片路径: {path}")
        if path is None or not os.path.exists(path):
            return jsonify({'error': 'Detection image not found'}), 404
        return send_file(path, mimetype='image/jpeg')

    def set_detection_path(self):
        # 设置检测结果图片路径
        data = request.json
        new_path = data.get('path')
        print(f"设置检测结果图片路径: {new_path}")
        if new_path and os.path.exists(new_path):
            self.image_paths['detection'] = new_path
            return jsonify({'success': True, 'message': 'Detection image path updated'})
        return jsonify({'success': False, 'message': 'Invalid detection image path'})

    def update_detection(self):
        # 查找最新的检测结果图片并更新路径
        detection_dir = os.path.join(self.project_root, 'photos', 'dangerous_det')
        if not os.path.exists(detection_dir):
            return jsonify({'success': False, 'message': 'Detection image directory not found'})
        image_files = [f for f in os.listdir(detection_dir) if f.lower().endswith(('.jpg', '.png'))]
        if not image_files:
            return jsonify({'success': False, 'message': 'No detection image found'})
        latest_image_file = max(image_files, key=lambda f: os.path.getmtime(os.path.join(detection_dir, f)))
        latest_image_path = os.path.join(detection_dir, latest_image_file)
        current_image_path = self.image_paths['detection']
        if latest_image_path != current_image_path:
            self.image_paths['detection'] = latest_image_path
            return jsonify({'success': True, 'path': '/get_detection'})
        else:
            self.image_paths['detection'] = latest_image_path
            return jsonify({'success': False, 'message': 'No new detection image'})

    def get_segmentation(self):
        # 获取分割图片
        path = self.image_paths['segmentation']
        if path is None or not os.path.exists(path):
            return jsonify({'error': 'Segmentation image not found'}), 404
        return send_file(path, mimetype='image/jpeg')

    def set_segmentation_path(self):
        # 设置分割图片路径
        data = request.json
        new_path = data.get('path')
        if new_path and os.path.exists(new_path):
            self.image_paths['segmentation'] = new_path
            return jsonify({'success': True, 'message': 'Segmentation image path updated'})
        return jsonify({'success': False, 'message': 'Invalid segmentation image path'})

    def update_segmentation(self):
        # 查找最新的分割图片并更新路径
        segmentation_dir = os.path.join(self.project_root, 'photos', 'segmentation')
        if not os.path.exists(segmentation_dir):
            return jsonify({'success': False, 'message': 'Segmentation image directory not found'})
        image_files = [f for f in os.listdir(segmentation_dir) if f.lower().endswith(('.jpg', '.png'))]
        if not image_files:
            return jsonify({'success': False, 'message': 'No segmentation image found'})
        latest_image_file = max(image_files, key=lambda f: os.path.getmtime(os.path.join(segmentation_dir, f)))
        latest_image_path = os.path.join(segmentation_dir, latest_image_file)
        current_image_path = self.image_paths['segmentation']
        if latest_image_path != current_image_path:
            self.image_paths['segmentation'] = latest_image_path
            return jsonify({'success': True, 'path': '/get_segmentation'})
        else:
            self.image_paths['segmentation'] = latest_image_path
            return jsonify({'success': False, 'message': 'No new segmentation image'})

    def get_json_data(self):
        # 获取检测结果JSON文件的最新一条数据
        path = os.path.join(self.project_root, 'photos', 'detection_results.json')
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

    def get_json_history(self):
        # 获取检测结果JSON文件的历史数据（最多10条）
        path = os.path.join(self.project_root, 'photos', 'detection_results.json')
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
                except json.JSONDecodeError:
                    continue
            return jsonify({'success': True, 'data': records})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def get_segmentation_stats(self):
        # 获取分割结果统计信息
        try:
            file_path = os.path.join(self.project_root, 'photos', 'segmentation_result.json')
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
                return jsonify({'success': False, 'error': 'No valid data in file'}), 404
            data = json.loads(last_line)
            counts = data.get('counts', {})
            areas = data.get('areas', {})
            categories = {
                'SquareTube': ['0', '1', '2'],
                'RoundTube': ['3', '4', '5'],
                'SteelPlate': ['6'],
                'Galvanized': ['7', '8', '9', '10'],
                'Rebar': ['11', '12'],
                'PipeFitting': ['13'],
                'Connector': ['14'],
                'PerforatedScrap': ['15']
            }
            stats = {}
            for category, indices in categories.items():
                count = sum(int(counts.get(idx, 0)) for idx in indices)
                area = sum(float(areas.get(idx, 0)) for idx in indices)
                stats[category] = {'count': count, 'area': area}
            response_data = {
                'success': True,
                'stats': stats,
                'total_count': data.get('total_count', 0),
                'total_area': data.get('total_area', 0)
            }
            return jsonify(response_data)
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    def get_latest_danger_counts(self):
        # 获取最新危险物数量统计
        path = os.path.join(self.project_root, 'photos', 'detection_results.json')
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
            return jsonify({'success': False, 'message': 'Failed to read JSON data', 'error': str(e)})

    def get_danger_total(self):
        # 获取危险物总数
        path = os.path.join(self.project_root, 'photos', 'detection_results.json')
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
            return jsonify({'success': False, 'message': 'Failed to read JSON data', 'error': str(e)})

    def check_file_time(self, file_type):
        # 检查指定类型文件的最后修改时间
        if file_type == 'segmentation':
            file_path = os.path.join(self.project_root, 'photos', 'segmentation_result.json')
        else:
            file_path = os.path.join(self.project_root, 'photos', 'detection_results.json')
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        last_modified = os.path.getmtime(file_path)
        return jsonify({'success': True, 'last_modified': last_modified})

    def check_file_new_data(self, file_type):
        # 检查指定类型文件是否有新数据
        if file_type == 'segmentation':
            file_path = os.path.join(self.project_root, 'photos', 'segmentation_result.json')
        else:
            file_path = os.path.join(self.project_root, 'photos', 'detection_results.json')
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
            return jsonify({'success': False, 'error': 'File is empty'}), 404
        try:
            data = json.loads(last_line)
            return jsonify({'success': True, 'last_line_data': data, 'data_count': data_count, 'last_modified': os.path.getmtime(file_path)})
        except json.JSONDecodeError:
            return jsonify({'success': False, 'error': 'Invalid JSON format'}), 500
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    def get_vehicle_history(self):
        """获取车辆历史信息，返回 vehicle_records/vehicles.json 文件中最新的10条记录"""
        import os, json
        from flask import jsonify
        file_path = os.path.abspath(os.path.join(os.getcwd(), 'vehicle_records', 'vehicles.json'))
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': '车辆历史文件未找到'}), 404
        try:
            lines = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        lines.append(line.strip())
            if not lines:
                return jsonify({'success': False, 'error': '无历史记录'}), 404
            # 取最新的10条
            records = []
            for i in range(len(lines)-1, -1, -1):
                try:
                    record = json.loads(lines[i])
                    records.append(record)
                    if len(records) >= 10:
                        break
                except json.JSONDecodeError:
                    continue
            records.reverse()  # 保证时间顺序从旧到新
            return jsonify({'success': True, 'data': records})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500 