import threading
import time
import datetime
import json
import os
from network import SERVER_CONFIG, ServerConnection
from arm1 import start_arm1_cycle
from MQTT.mqtt_copy import MessageProcessor
from config.config_loader import ConfigLoader
from myutils.image_capture_runner import start_image_capture
from myutils.dangerous_detection_runner import start_dangerous_detection
from myutils.scrap_segmentation_runner import start_scrap_segmentation

# # 创建全局MQTT处理器实例
# mqtt_processor = MessageProcessor(
#     broker="192.168.3.144",  # MQTT服务器地址
#     port=1883,               # MQTT服务器端口
#     topic=["Channel1"],      # 主题列表
#     client_id="arm1"         # 客户端ID
# )

mqtt_processor = MessageProcessor(
    broker="127.0.0.1",  # MQTT服务器地址
    port=1883,               # MQTT服务器端口
    topic=["Channel1"],      # 主题列表
    client_id="arm1"         # 客户端ID
)

# 定义小车列表，支持任意多个车辆
number_plate_list=["XD00001", "XD00002"]

# 创建一个信号量用于控制机械臂启动
arm1_signal = threading.Event()

# 存储当前处理的车牌号
current_plate_number = None

# 存储机械臂状态
arm1_status = False

# 线程安全的锁
device_lock = threading.Lock()

# 定义消息处理函数
# 处理MQTT消息，解析车牌号和状态码
def process_message(content):
    global current_plate_number, arm1_status
    
    print(f"收到MQTT消息: {content}")
    print(f"------- 开始处理MQTT消息 -------")

    # 特别处理：启动时的空消息
    if not content or content == "":
        print("忽略MQTT客户端启动时的初始空消息")
        print(f"------- 结束处理MQTT消息 -------")
        return content

    try:    
        # 尝试解析消息格式: "车牌号:状态码"
        parts = content.split(":", 1)
        
        if len(parts) == 2:
            plate_number = parts[0].strip()
            status_code = parts[1].strip()
                
            print(f"解析消息成功: 车牌号: {plate_number}，状态码: {status_code}")
        else:
            print(f"消息格式不正确: {content}")
            return content
    except Exception as e:
        print(f"解析消息失败: {e}，消息内容: {content}")

    
    # 检查是否为到达5号点位的信号
    if status_code == "4":
        print(f"检测到状态码4，准备处理小车到达信号")   

        # 更新机械臂状态
        arm1_status = True
        write_arm_status(arm1_status)
            
        # 只有当发送者是已知小车时才处理
        if plate_number in number_plate_list:
            print(f"收到启动信号4，车牌号: {plate_number}")
            
            # 记录车辆到达
            current_time = datetime.datetime.now().strftime("%Y年%m月%d日%H时%M分%S秒")
            print(f"{plate_number}，{current_time}")
            add_vehicle_record(plate_number, current_time, "arrive")
            
            with device_lock:
                current_plate_number = plate_number  # 保存当前车牌号
            
            # 设置启动信号
            arm1_signal.set()
        else:
            print(f"收到状态码4，但发送者'{plate_number}'不在已知小车列表中，忽略该消息")
    else:
        print(f"状态码非4 ({status_code})，不触发机械臂")
    
    print(f"------- 结束处理MQTT消息 -------")
    return content


# 将车辆记录添加到JSON文件
def add_vehicle_record(plate_number, time_str, status):
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建记录目录
    records_dir = os.path.join(current_dir, "vehicle_records")
    if not os.path.exists(records_dir):
        os.makedirs(records_dir)
    
    # JSON文件路径 - 修改为保存在vehicle_records目录中
    json_file = os.path.join(records_dir, "vehicles.json")
    
    # 创建记录
    record = {
        "plate_number": plate_number,
        "time": time_str,
        "status": status
    }
    
    # 将记录转为JSON字符串并追加到文件（每行一条记录）
    with open(json_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# 写入机械臂状态到文件
def write_arm_status(status):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 目标路径：Scrap_Steel_In_Field\photos\arm_status.json
        photos_dir = os.path.join(current_dir, "photos")
        # 自动创建photos目录（若不存在）
        os.makedirs(photos_dir, exist_ok=True)
        status_file = os.path.join(photos_dir, "arm_status.json")
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({"arm1_status": status}, f)
    except Exception as e:
        print(f"写入机械臂状态失败: {e}")

# MQTT处理器线程函数
def mqtt_receive():
    # 连接MQTT服务器但不启动消息循环
    print("连接MQTT服务器...")
    mqtt_processor.client.connect()
    
    # 先尝试清除保留消息
    # print("尝试清除MQTT主题上的保留消息...")
    try:
        # 清除Channel1主题上的保留消息
        for topic in mqtt_processor.client.topics:
            mqtt_processor.client.client.publish(topic, "", qos=mqtt_processor.client.qos, retain=True)
            # print(f"已发送清除主题 {topic} 的保留消息命令")
        time.sleep(1)  # 等待清除操作完成
    except Exception as e:
        print(f"清除保留消息失败: {e}")
    
    # 然后注册消息处理函数
    mqtt_processor.register_handler("Channel1", process_message)
    
    # 最后启动消息处理循环
    print("启动MQTT消息处理...")
    mqtt_processor.start()  # 这是阻塞调用，会一直运行

# MQTT发送消息函数
def mqtt_send(message, retain=False):
    global current_plate_number, arm1_status
    
    try:
        # 如果是处理完成消息且有设备ID，添加设备ID到消息中
        if message == "start":
            # 使用plaintext格式: "机械臂ID:命令"
            full_message = f"arm1:start"
            mqtt_processor.publish(full_message, retain=retain)
            
            # 记录车辆离开
            current_time = datetime.datetime.now().strftime("%Y年%m月%d日%H时%M分%S秒")
            plate_number = current_plate_number
            print(f"{plate_number}，{current_time}")
            add_vehicle_record(plate_number, current_time, "leave")
            
            # 更新机械臂状态为未启动
            arm1_status = False
            write_arm_status(arm1_status)
        else:
            # 直接发送消息
            mqtt_processor.publish(message, retain=retain)
            
    except Exception as e:
        print(f"发送MQTT消息时出错: {e}")

# 绑定组和锁
bound_groups = {
    "group1": {"arm": "arm1", "car": None, "lock": threading.Lock(), "processing": False},
}

# 机械臂控制线程函数
# 等待启动信号后运行机械臂控制程序
def arm1_control_thread(connections):
    """等待启动信号后运行机械臂控制程序"""
    global current_plate_number,arm1_status
    
    print("机械臂控制线程已创建，等待启动信号...")
    
    while True:
        # 等待启动信号
        arm1_signal.wait()
        
        # 获取当前设备ID
        with device_lock:
            plate_number = current_plate_number
        
        if plate_number:
            print(f"收到启动信号，车辆: {plate_number}到达，开始运行机械臂控制程序...")

            # 更新绑定状态
            target_group = "group1"
            group = bound_groups[target_group]
            
            with group["lock"]:
                # 只有当小车未绑定且组未在处理中时才进行绑定
                if group["car"] is None and not group["processing"]:
                    group["car"] = plate_number
                    group["processing"] = True
                    
                    print(f"{plate_number} 成功绑定到 {group['arm']}")
                    
                    # 运行机械臂控制程序
                    try:
                        # 使用当前的连接字典传递给 start_arm1_cycle
                        start_arm1_cycle("arm1", connections)
                        
                        # 发送MQTT消息
                        mqtt_send("start")
                        
                    except Exception as e:
                        print(f"运行机械臂程序时出错: {e}")
                    finally:
                        # 无论成功或失败，都清空所有状态
                        with device_lock:
                            current_plate_number = None
                        group["car"] = None
                        group["processing"] = False
                else:
                    print(f"{plate_number} 绑定失败：{target_group} 已被占用")
        
        # 重置信号，以便下次使用
        arm1_signal.clear()
        # 确保机械臂状态重置
        arm1_status = False
        print("机械臂控制程序执行完毕，等待下一次启动信号")

def main():
    print("启动废钢装箱系统...")

    # 启用机械臂通讯模块时创建服务器连接
    connections = {}

    # 如果启用抓取点与机械臂控制模块，只启动机械臂控制线程和MQTT服务。
    # 其他所有功能模块（图像采集、危险物检测、废钢分割）都不会启动，因为这些功能已包含在该模块中。
    if ConfigLoader.is_module_enabled('grasp_point_and_arm_control'):
        mqtt_thread = threading.Thread(target=mqtt_receive, daemon=True)
        mqtt_thread.start()
        print("MQTT监控已启动。")
        # 启用抓取点与机械臂控制模块会自动启用机械臂通讯和所有子模块。
        connections["arm1"] = ServerConnection(SERVER_CONFIG["arm1"], "arm1")
        arm1_thread = threading.Thread(target=arm1_control_thread, args=(connections,), daemon=True)
        arm1_thread.start()
    else:
        # 如果未启用抓取点与机械臂控制模块，则根据配置分别启动各功能模块
        if ConfigLoader.is_module_enabled('image_capture'):
            image_thread = threading.Thread(target=start_image_capture, daemon=True)
            image_thread.start()
            print("图像采集模块已启动。")

        if ConfigLoader.is_module_enabled('dangerous_detection'):
            danger_thread = threading.Thread(target=start_dangerous_detection, daemon=True)
            danger_thread.start()
            print("危险物检测模块已启动。")

        if ConfigLoader.is_module_enabled('scrap_segmentation'):
            seg_thread = threading.Thread(target=start_scrap_segmentation, daemon=True)
            seg_thread.start()
            print("废钢分割模块已启动。")

        # 如果只启用了机械臂通讯模块（未启用抓取点与机械臂控制），打印配置信息，启动MQTT和机械臂控制线程
        if ConfigLoader.is_module_enabled('arm_communication'):
            mqtt_thread = threading.Thread(target=mqtt_receive, daemon=True)
            mqtt_thread.start()
            print("MQTT监控已启动。")
            connections["arm1"] = ServerConnection(SERVER_CONFIG["arm1"], "arm1")
            arm1_thread = threading.Thread(target=arm1_control_thread, args=(connections,), daemon=True)
            arm1_thread.start()

    try:
        # 主线程持续运行
        while True:
            time.sleep(10)
            print("系统运行中...")
    except KeyboardInterrupt:
        print("\n收到终止信号，正在关闭...")
    print("系统已关闭。")

if __name__ == "__main__":
    main()