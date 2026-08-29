import threading
import time
import os
import cv2
import queue
import requests
from datetime import datetime
from network import ServerConnection, SERVER_CONFIG, shared_queue
from myutils.logger import setup_logger
from config.arm_config import LOG_CONFIG, ARM_CONFIG
from config.detection_config import SAVE_CONFIG
from config.camera_config import CAMERA_CONFIG
from get_image.get_clear_picture import get_clear_image
from get_image.get_img import Camera
from inference.detection_inference_simplified import simple_detect
from inference.scrap_segmentation import simple_segment
from get_image.field_light import on_field_light
# from Scrap_Steel_Loading.corner_coord_detect.arm_coords import detect_and_transform
from corner_coord_detect.detect_and_transform import calculate_grasp_points,detect_and_transform_rectangle
import json
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

# 全局日志记录器
logger = None
# 全局相机对象
camera = None
# 确保保存目录存在
for dir_name in SAVE_CONFIG["sub_dirs"].values():
    os.makedirs(os.path.join(SAVE_CONFIG["base_dir"], dir_name), exist_ok=True)

def initialize_logger():
    """初始化日志记录器"""
    global logger
    # 获取基础目录（当前文件所在目录）
    base_dir = os.path.dirname(__file__)
    # 更新配置文件中的base_dir
    LOG_CONFIG["arm1"]["base_dir"] = base_dir
    # 使用配置文件中的设置
    logger = setup_logger(
        "arm1", 
        LOG_CONFIG["arm1"]["base_dir"], 
        sub_dir=LOG_CONFIG["arm1"]["sub_dir"]
    )

def initialize_camera():
    """初始化相机对象"""
    global camera
    camera = Camera(CAMERA_CONFIG["camera_ip"], 
                    CAMERA_CONFIG["username"], 
                    CAMERA_CONFIG["password"]
                    )
    logger.info("相机对象初始化完成")

# 初始化日志记录器
initialize_logger()
# 初始化相机
initialize_camera()

def send_photo_request(photo_type="original", max_retries=None, retry_delay=None):
    """发送拍照请求并等待响应，支持重试与错误处理
    
    使用clear_picture.py中的get_clear_image函数获取清晰图像
    
    Args:
        photo_type (str): 照片类型，"vehicle"表示车体照片，"original"表示原始照片，默认"original"
        max_retries (int, optional): 最大重试次数，默认使用配置文件中的值
        retry_delay (float, optional): 重试间隔，默认使用配置文件中的值
        
    Returns:
        tuple: (拍照是否成功, 图片保存路径)
    """
    # 使用配置文件中的设置，如果未提供则使用默认值
    if max_retries is None:
        max_retries = ARM_CONFIG["arm1"]["photo_retry"]["max_retries"]
    if retry_delay is None:
        retry_delay = ARM_CONFIG["arm1"]["photo_retry"]["retry_delay"]
    
    global camera
    if camera is None:
        initialize_camera()
    
    try:
        logger.info("开始获取清晰图像...")
        # 使用get_clear_image函数获取清晰图像
        frame = get_clear_image(
            camera, 
            CAMERA_CONFIG["channel"], 
            max_attempts=max_retries,
            wait_time=retry_delay,
            timeout=max_retries * retry_delay * 2, 
            save_config=SAVE_CONFIG
        )
        
        if frame is not None:
            # 保存图像
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 根据照片类型选择保存目录
            sub_dir = SAVE_CONFIG["sub_dirs"].get(photo_type, "original")
            
            # 确保目录存在
            save_dir = os.path.join(SAVE_CONFIG["base_dir"], sub_dir)
            os.makedirs(save_dir, exist_ok=True)
            
            # 构建文件名和路径
            save_path = os.path.join(
                save_dir,
                f"{photo_type}_image_{timestamp}.jpg"
            )
            
            cv2.imwrite(save_path, frame)
            logger.info(f"已保存{photo_type}图像: {save_path}")
            
            return True, save_path
        else:
            logger.error("未能获取清晰图像")
            return False, None
            
    except Exception as e:
        logger.error(f"获取清晰图像过程中发生错误: {str(e)}")
        return False, None

def receive_arm_message(arm_name):
    """接收机械臂的消息"""
    try:
        server_name, msg = shared_queue.get()
        if server_name == arm_name:
            # 只有当消息来自目标机械臂时才标记为完成
            shared_queue.task_done()
            logger.info(f"收到机械臂消息: {msg}")
            return msg
        else:
            # 如果不是目标消息，放回队列并返回None
            shared_queue.put((server_name, msg))
            return None
    except queue.Empty:
        logger.warning(f"等待机械臂消息超时")
        return None
    except Exception as e:
        logger.error(f"接收消息时出错: {str(e)}")
        return None

def start_arm1_cycle(arm_name, connections):
    """机械臂1的操作流程，触发拍照功能"""
        
    # 每次循环开始时创建新的日志文件
    initialize_logger()
    
    # 创建危险物识别结果JSON文件
    detection_json_path = os.path.join(SAVE_CONFIG["base_dir"], "detection_results.json")
    
    try:
        # logger.info(f"\n===== 开始 {arm_name} 的处理流程 =====")
        # logger.info(f"\n[步骤1] 请求拍摄车体照片")
        # success, image_path = send_photo_request(photo_type="vehicle")

        # if success:
        #     logger.info(f"[完成] 车体拍照完成: {image_path}")
        # else:
        #     logger.warning(f"[警告] 车体拍照可能未完成，但将继续流程")

        # logger.info(f"\n[步骤2] 进行废钢检测")
        i=0
        while True:
            i+=1
            msg = receive_arm_message(arm_name)
            if msg=="yes_grab":
                success, image_path = send_photo_request(photo_type="original")
                on_field_light(True)
                time.sleep(2)
                on_field_light(False)
                if success:
                    logger.info(f"[完成] 第{i}次原始图片拍照完成: {image_path}")

                    
                    # 对拍摄的原始图片进行危险品推理
                    logger.info(f"[步骤{i+2}.1] 进行危险品推理")
                    try:
                        # 直接推理
                        detections, detect_success = simple_detect(image_path=image_path, config=None)
                        if detect_success: 
                            if detections and detections[0]["label"] != "NoDangerous":
                                logger.info(f"[完成] 危险品检测成功，发现 {len(detections)} 个危险品")
                                for det_idx, det in enumerate(detections):
                                    logger.info(f"危险品 {det_idx+1}: {det['label']} (置信度: {det['confidence']:.2f})")
                            else:
                                logger.info("[完成] 危险品检测完成，未发现危险品")
                                
                            # 将检测结果写入JSON文件
                            try:
                                # 统计各类危险品的数量
                                danger_types = ['GasCyl1', 'GasCyl2', 'FireExt1', 'FireExt2']
                                counts = {dtype: 0 for dtype in danger_types}
                                for det in detections:
                                    if det['label'] in danger_types:
                                        counts[det['label']] += 1
                                
                                # 将统计结果添加到JSON中
                                detection_result = {
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "image_path": image_path,
                                    "detections": detections,
                                    "counts": counts,  # 添加各类别的计数
                                    "total_dangerous": sum(counts.values())  # 添加危险品总数
                                }
                                with open(detection_json_path, 'a', encoding='utf-8') as f:
                                    json.dump(detection_result, f, ensure_ascii=False)
                                    f.write('\n')  # 每行一个JSON对象
                                logger.info(f"危险品检测结果已写入JSON文件: {detection_json_path}")
                                
                            #     # 生成饼状图
                            #     try:
                            #         # 使用之前已经统计好的counts数据
                            #         # 计算总数和比例
                            #         total = sum(counts.values())
                            #         # 创建饼状图
                            #         plt.figure(figsize=(8, 6))
                                    
                            #         if total > 0:
                            #             # 有危险品时只显示检测到的类型
                            #             labels = []
                            #             sizes = []
                            #             colors_map = {'GasCyl1': '#ff9999', 'GasCyl2': '#66b3ff', 'FireExt1': '#99ff99', 'FireExt2': '#ffcc99'}
                            #             colors = []
                                        
                            #             # 只添加计数不为0的危险品类型
                            #             for dtype in danger_types:
                            #                 if counts[dtype] > 0:
                            #                     labels.append(f"{dtype} ({counts[dtype]})")
                            #                     sizes.append(counts[dtype])
                            #                     colors.append(colors_map[dtype])
                                        
                            #             plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                            #                 startangle=90)
                            #             plt.title('危险品检测结果分布')
                            #         else:
                            #             # 没有危险品时显示"未检测到危险品"
                            #             labels = ['未检测到危险品']
                            #             sizes = [100]
                            #             colors = ['#eeeeee']  # 浅灰色
                            #             plt.pie(sizes, labels=labels, colors=colors, startangle=90)
                            #             plt.title('危险品检测结果：未检测到危险品')
                                    
                            #         plt.axis('equal')  # 使饼图为正圆形
                                    
                            #         # 保存饼状图
                            #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            #         # 确保dangerous_pie文件夹存在
                            #         dangerous_pie_dir = os.path.join(SAVE_CONFIG["base_dir"], "dangerous_pie")
                            #         os.makedirs(dangerous_pie_dir, exist_ok=True)
                            #         pie_chart_path = os.path.join(dangerous_pie_dir, f"dangerous_pie_chart_{timestamp}.png")
                            #         plt.savefig(pie_chart_path)
                            #         plt.close()
                                    
                            #         # 更新前端显示
                            #         try:
                            #             # 更新饼状图路径
                            #             response = requests.post('http://localhost:5000/set_pie_chart_path', 
                            #                                 json={'path': pie_chart_path})
                            #             if response.json()['success']:
                            #                 logger.info("饼状图已更新到 Flask 应用")
                            #             else:
                            #                 logger.warning("更新饼状图到 Flask 应用失败")
                            #         except Exception as e:
                            #             logger.error(f"更新饼状图到 Flask 应用时出错: {str(e)}")
                            #     except Exception as e:
                            #         logger.error(f"生成饼状图时出错: {str(e)}")
                                    
                            except Exception as e:
                                logger.error(f"写入危险品检测结果到JSON文件时出错: {str(e)}")

                            # # 将检测结果推送到前端
                            # try:
                            #     # 从detections中获取检测结果图片路径
                            #     detection_path = detections[0].get('result_path')
                            #     logger.info(f"检测结果的路径为：{detection_path}")
                            #     if detection_path and os.path.exists(detection_path):
                            #         logger.info(f"文件存在，大小: {os.path.getsize(detection_path)} 字节")
                            #         response = requests.post('http://localhost:5000/set_detection_path', 
                            #                             json={'path': detection_path})
                            #         if response.json()['success']:
                            #             logger.info("危险品检测结果已更新到 Flask 应用")
                            #         else:
                            #             logger.warning("更新危险品检测结果到 Flask 应用失败")
                            #     else:
                            #         logger.warning(f"未找到检测结果图片路径或文件不存在: {detection_path}")
                            # except Exception as e:
                            #     logger.error(f"更新危险品检测结果到 Flask 应用时出错: {str(e)}")
                        else:
                            logger.warning("[警告] 危险品检测失败，将按默认流程继续")
                    except Exception as e:
                        logger.error(f"[错误] 危险品检测过程中发生异常: {e}")
                        logger.warning("[警告] 由于检测异常，将按默认流程继续")
                        
                    # 对同一张原始图片进行废钢分割推理
                    logger.info(f"[步骤{i+2}.2] 对原始图片进行废钢分割")
                    try:
                        # 使用相同的图片路径进行废钢分割
                        detections, mask, save_path, class_counts, area_stats = simple_segment(image_path=image_path)
                        if mask is not None:
                            logger.info(f"[完成] 废钢分割成功")
                            if len(detections) > 0:
                                logger.info(f"检测到 {len(detections)} 个物体")
                                logger.info(f"类别统计: {class_counts}")
                                logger.info(f"面积统计: {area_stats}")
                            else:
                                logger.info("未检测到物体")
                            logger.info(f"废钢分割结果保存路径: {save_path}")
                        else:
                            logger.warning("[警告] 废钢分割失败，将按默认流程继续")
                    except Exception as e:
                        logger.error(f"[错误] 废钢分割过程中发生异常: {e}")
                        logger.warning("[警告] 由于分割异常，将按默认流程继续")
                else:
                    logger.warning(f"[警告] 第{i}次原始图片拍照可能未完成，但将继续流程")

                logger.info(f"\n[步骤{i+2}.3] 请求第{i}次抓取循环")

                """
                机械臂操作，通过角点检测与坐标转换，获取机械臂在真实世界的抓取点，进行抓取操作
            
                每发送一个坐标点，机械臂进行抓取，等待机械臂返回一个坐标点
                """
                # 获取机械臂坐标系抓取点
                _, world_corners = detect_and_transform_rectangle(show_result=False, max_attempts=5, image_path=image_path)
                grab_points = calculate_grasp_points(world_corners)

                logger.info(f"机械臂坐标系抓取点: {grab_points}")

                try:
                    # 提取抓取点坐标并添加z=0
                    if grab_points is not None:
                        formatted_points = []
                        for x, y in grab_points:
                            formatted_points.append(f"{x},{y},0")
                        
                        logger.info("抓取点坐标（z=0）:")
                        for point in formatted_points:
                            logger.info(point)

                    # connections[arm_name].safe_send("300,300,10;")
                    # 分别发送三个坐标点
                    if grab_points is not None and len(grab_points) >= 3:
                        # 发送第一个坐标点
                        first_point = f"{grab_points[0][0]},{grab_points[0][1]},0"
                        logger.info(f"发送第一个抓取点坐标: {first_point}")
                        connections[arm_name].safe_send(first_point + ";")

                        msg = receive_arm_message(arm_name)
                        if msg=="A":
                            logger.info(f"收到第{i}次的抓取点位: {msg}")
                        else:
                            time.sleep(2)  # 等待2000ms后重试

                        # 发送第二个坐标点
                        second_point = f"{grab_points[1][0]},{grab_points[1][1]},0"
                        logger.info(f"发送第二个抓取点坐标: {second_point}")
                        connections[arm_name].safe_send(second_point + ";")


                        msg = receive_arm_message(arm_name)
                        if msg=="B":
                            logger.info(f"收到第{i}次的抓取点位: {msg}")
                        else:
                            time.sleep(2)  # 等待2000ms后重试
                        
                        # 发送第三个坐标点
                        third_point = f"{grab_points[2][0]},{grab_points[2][1]},0"
                        logger.info(f"发送第三个抓取点坐标: {third_point}")
                        connections[arm_name].safe_send(third_point + ";")

                        msg = receive_arm_message(arm_name)
                        if msg=="C":
                            logger.info(f"收到第{i}次的抓取点位: {msg}")
                        else:
                            time.sleep(2)  # 等待2000ms后重试

                    logger.info(f"[完成] 第{i}次抓取指令发送成功")
                    
                except Exception as e:
                    logger.error(f"[错误] 第{i}次抓取指令发送失败: {str(e)}")
                # 等待传回抓取完成，进行下一个循环
            else:
                logger.info(f"\n===== {arm_name} 的处理流程已完成 =====\n")
                break
    except Exception as e:
        logger.error(f"[错误] 车体拍照处理过程中发生异常: {e}")

def run_arm1_server():
    # 创建服务器连接
    connections = {
        "arm1": ServerConnection(SERVER_CONFIG["arm1"], "arm1")
    }
    
    logger.info("服务器已启动，等待连接...")
    
    # 等待连接就绪
    while True:
        with connections["arm1"].conn_lock:
            if connections["arm1"].client_socket is not None:
                break
        time.sleep(0.5)
    
    logger.info("机械臂已连接，开始处理...")

    try:
        # 创建并启动机械臂1处理线程
        process_thread = threading.Thread(
            target=start_arm1_cycle,
            args=("arm1", connections)
        )
        process_thread.start()
        
        # 等待处理线程完成
        process_thread.join()
        
        logger.info("机械臂处理流程已完成")
    
    except KeyboardInterrupt:
        logger.info("\n接收到终止信号，正在关闭...")
#    finally:
#        # 关闭所有连接
#        for conn in connections.values():
#            conn.stop()
#        logger.info("所有连接已关闭")

if __name__ == "__main__":
    run_arm1_server()