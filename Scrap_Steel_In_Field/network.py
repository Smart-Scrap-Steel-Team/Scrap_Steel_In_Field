import socket
import threading
import queue
import time
import random
from Scrap_Steel_In_Field.config.camera_config import CAMERA_CONFIG
from Scrap_Steel_In_Field.get_image.get_img import Camera

# 全局队列
shared_queue = queue.Queue()

# 服务器配置
SERVER_CONFIG = {
    "arm1": {"host": "0.0.0.0", "port": 8769, "retry_interval": 5},
#    "camera": {"host": "0.0.0.0", "port": 8771, "retry_interval": 5},
}

class ServerConnection:
    def __init__(self, config, name):
        self.name = name
        self.host = config["host"]
        self.port = config["port"]
        self.retry_interval = config.get("retry_interval", 5)
        self.socket = None
        self.client_socket = None
        self.conn_lock = threading.Lock()
        self.should_run = True
        
        self.server_thread = threading.Thread(target=self.run_server, daemon=True)
        self.server_thread.start()
    
    def run_server(self):
        retry_delay = self.retry_interval
        max_retry_delay = 30

        # 控制服务器运行，尝试启动服务器       
        while self.should_run:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # 允许地址重用
                self.socket.bind((self.host, self.port))
                self.socket.listen(5)
                print(f"[{self.name}] 启动成功，监听 {self.host}:{self.port}")
                # 尝试日志监听
#                logging.basicConfig(filename="server.log", level=logging.INFO)
#                logging.info(f"[{self.name}] 服务器启动成功，监听 {self.host}:{self.port}")
                
                retry_delay = self.retry_interval

                # 监听客户端连接，处理动态连接                
                while self.should_run:
                    self.socket.settimeout(1.0)
                    try:
                        client_sock, addr = self.socket.accept() # 接受连接，返回客户端socket和地址
                        print(f"[{self.name}] 接受来自 {addr} 的连接")
                        
                        with self.conn_lock:
                            self.client_socket = client_sock

                        # 多客户端连接
                        # self.client_sockets = {}  # {addr: socket}
                        # with self.conn_lock:
                        #     self.client_sockets[addr] = client_socket
                        
                        try:
                            client_sock.sendall("START".encode()) # 发个消息测试下
                        except Exception as e:
                            print(f"[{self.name}] 发送消息失败: {e}")
                        
                        # 每个设备开启单独线程              
                        threading.Thread(
                            target=self.handle_client,
                            args=(client_sock, addr),
                            daemon=True
                        ).start()
                        
                    except socket.timeout:
                        continue
                    except Exception as e:
                        print(f"[{self.name}] 接受连接时出错: {e}")
                        time.sleep(0.5)
                
            except Exception as e:
                print(f"[{self.name}] 服务器启动失败: {e}")
                time.sleep(min(retry_delay * random.uniform(0.8, 1.2), max_retry_delay))
                retry_delay = min(retry_delay * 1.5, max_retry_delay)
                    
            finally:
                if self.socket:
                    try:
                        self.socket.close()
                    except:
                        pass
    
    # 处理单个客户端连接
    def handle_client(self, client_sock, addr):
        try:
            while self.should_run:
                try:
                    client_sock.settimeout(5.0)
                    data = client_sock.recv(1024)
                    
                    if not data:
                        break
                        
                    msg = data.decode()
                    print(f"[{self.name}] 收到: {msg}")
                    shared_queue.put((self.name, msg))
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"[{self.name}] 处理客户端 {addr} 时出错: {e}")
                    break
        finally:
            try:
                client_sock.close()
            except:
                pass
                
            with self.conn_lock:
                if self.client_socket == client_sock:
                    self.client_socket = None
            
            print(f"[{self.name}] 客户端 {addr} 连接已关闭")

    # 安全发送消息
    def safe_send(self, msg):
        with self.conn_lock:
            if not self.client_socket:
                raise RuntimeError(f"[{self.name}] 没有连接的客户端")                
            try:
                self.client_socket.sendall(msg.encode() if isinstance(msg, str) else msg)
                return True
            except (ConnectionResetError, BrokenPipeError) as e:
                print(f"[{self.name}] 发送失败: {e}")
                return False

    # 停止服务器
    def stop(self):
        self.should_run = False
        
        with self.conn_lock:
            if self.client_socket:
                try:
                    self.client_socket.close()
                except:
                    pass
                self.client_socket = None
        
        self.server_thread.join(timeout=1.0)
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass

def check_camera_status():
    """
    检查NVR通道的相机状态，使用Camera类的check_camera_existence函数
    
    Returns:
        bool: 相机是否正常运行
    """
    try:
        # 创建Camera实例
        camera = Camera(CAMERA_CONFIG["camera_ip"], 
                        CAMERA_CONFIG["username"], 
                        CAMERA_CONFIG["password"])
        
        # 检查相机通道状态
        camera_status = camera.check_camera_existence(channels=CAMERA_CONFIG["channels"])
        
        # 如果所有通道都可用，返回True，否则返回False
        return all(camera_status.values())
    except Exception as e:
        print(f"检查NVR相机状态时出错: {e}")
        return False

def monitor_connections(server_config, heartbeat_interval=5):
    """
    监控并维护所有服务器连接和NVR相机状态
    
    Args:
        server_config (dict): 服务器配置字典
        heartbeat_interval (int): 心跳发送间隔，单位秒，默认为5秒
    
    Returns:
        None
    """

    connections = {}
#    for name, config in server_config.items():
#        if name == "camera":
#            connections[name] = CameraConnection(config, name)
#        else:
#            connections[name] = ServerConnection(config, name)

    for name, config in server_config.items():
        connections[name] = ServerConnection(config, name)    
    try:
        while True:
            time.sleep(heartbeat_interval)  # 心跳间隔
            name_list = []
            for name, conn in connections.items():
                try:
                    conn.safe_send("HEARTBEAT") # 发送即可，不需要等待回复
                    name_list.append(name)
                except Exception as e:
                    print(f"心跳发送失败 {name}: {e}")
            
            # 检查NVR相机状态
            camera_status = check_camera_status()
            camera_status_text = "已开启" if camera_status else "未开启"
            
            print(f"设备状态: {name_list}等设备已连接，NVR相机{camera_status_text}")
            
    except KeyboardInterrupt:
        print("正在关闭服务器...")
    finally:
        for conn in connections.values():
            conn.stop()

if __name__ == "__main__":
    monitor_connections(SERVER_CONFIG)