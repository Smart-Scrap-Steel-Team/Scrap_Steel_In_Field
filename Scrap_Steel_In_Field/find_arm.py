import socket
import threading
import time
from queue import Queue

# 要扫描的网段
NETWORK = "192.168.3"
# 扫描端口列表
COMMON_PORTS = [
    80,  # HTTP
    554,  # RTSP (摄像头)
    8770,  # 机械臂通讯端口
    23,  # Telnet
    21,  # FTP
    22,  # SSH
    443,  # HTTPS
    8000,  # 常见服务
    8080,  # 常见服务
]

# 线程数
THREADS = 50

# 结果存储
found_devices = []
lock = threading.Lock()


def ping_ip(ip):
    """Ping一个IP地址"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        result = sock.connect_ex((ip, 1))  # 尝试连接一个不太可能开的端口，只是判断IP是否在线
        sock.close()

        # 或者用ICMP ping（但Windows下需要管理员权限）
        # 这里简化处理，只要能连接任意一个端口就算在线
        return True
    except:
        return False


def scan_port(ip, port):
    """扫描单个端口"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.3)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except:
        return False


def scan_worker(queue):
    """工作线程"""
    while not queue.empty():
        ip = queue.get()
        try:
            # 先检查IP是否在线
            is_online = False
            open_ports = []

            # 扫描常见端口
            for port in COMMON_PORTS:
                if scan_port(ip, port):
                    is_online = True
                    open_ports.append(port)

            # 如果有任意端口开放，认为设备在线
            if is_online:
                with lock:
                    found_devices.append({
                        "ip": ip,
                        "open_ports": open_ports
                    })
                    print(f"✓ 发现设备: {ip}")
                    if open_ports:
                        print(f"  开放端口: {open_ports}")

        except Exception as e:
            pass

        queue.task_done()


def main():
    print("=" * 60)
    print("扫描 192.168.3.x 网段")
    print("=" * 60)
    print(f"扫描端口: {COMMON_PORTS}")
    print("\n正在扫描，请稍候...\n")

    # 创建任务队列
    queue = Queue()

    # 填充IP地址 (排除已知设备)
    known_ips = ["192.168.3.10", "192.168.3.151", "192.168.3.193"]
    for i in range(1, 255):
        ip = f"{NETWORK}.{i}"
        if ip not in known_ips:
            queue.put(ip)

    # 创建并启动线程
    threads = []
    for _ in range(THREADS):
        t = threading.Thread(target=scan_worker, args=(queue,), daemon=True)
        t.start()
        threads.append(t)

    # 等待所有任务完成
    queue.join()

    # 显示结果
    print("\n" + "=" * 60)
    print("扫描结果")
    print("=" * 60)

    if found_devices:
        print(f"\n发现 {len(found_devices)} 个未知设备：\n")
        for device in found_devices:
            ip = device["ip"]
            ports = device["open_ports"]

            # 判断设备类型
            device_type = "未知设备"
            if 8770 in ports:
                device_type = "⚠️ 可能是机械臂！（开放8770端口）"
            elif 554 in ports:
                device_type = "📷 摄像头/视频设备（有RTSP端口）"
            elif 80 in ports:
                device_type = "🌐 有Web管理界面"
            elif 23 in ports:
                device_type = "🔌 支持Telnet管理"

            print(f"  IP: {ip}")
            print(f"  类型: {device_type}")
            print(f"  开放端口: {ports}")
            print()
    else:
        print("\n未发现其他设备！")
        print("\n请检查：")
        print("  1. 机械臂是否已开机")
        print("  2. 机械臂是否连接到交换机")
        print("  3. 机械臂的IP是否在192.168.3.x网段")

    print("=" * 60)
    print("\n提示：")
    print("- 机械臂可能使用静态IP，检查说明书")
    print("- 查看机械臂的显示屏/标签，找默认IP")
    print("- 扫描完成后，找到的设备尝试用浏览器或telnet连接")


if __name__ == "__main__":
    main()
