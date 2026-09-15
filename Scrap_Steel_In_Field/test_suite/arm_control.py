
import socket
import threading
import time
import queue


class ArmController:
    def __init__(self, host="0.0.0.0", port=8770):
        self.host = host
        self.port = port
        self.client_socket = None
        self.client_address = None
        self.running = False
        self.server_socket = None
        self.message_queue = queue.Queue()
        self.conn_lock = threading.Lock()
    
    def start_server(self):
        """启动服务器等待连接"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)
        self.running = True
        
        print(f"机械臂服务器已启动: {self.host}:{self.port}")
        print("等待机械臂连接...")
        
        # 启动接收线程
        recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
        recv_thread.start()
        
        while self.running:
            try:
                client, addr = self.server_socket.accept()
                with self.conn_lock:
                    self.client_socket = client
                    self.client_address = addr
                
                print(f"\n机械臂已连接: {addr[0]}:{addr[1]}")
                
                # 发送启动信号
                self._send("START")
                break
            except socket.timeout:
                continue
            except Exception as e:
                print(f"连接错误: {e}")
                if self.running:
                    time.sleep(1)
    
    def is_connected(self):
        """检查是否已连接"""
        with self.conn_lock:
            return self.client_socket is not None
    
    def _send(self, msg):
        """内部发送消息"""
        with self.conn_lock:
            if self.client_socket:
                try:
                    self.client_socket.sendall(msg.encode())
                    print(f"发送: {msg}")
                except Exception as e:
                    print(f"发送失败: {e}")
    
    def send_grasp_point(self, x, y, z=0):
        """发送单个抓取点"""
        msg = f"{x},{y},{z};"
        self._send(msg)
    
    def send_three_points(self, points):
        """发送三个抓取点（等待确认）"""
        if len(points) != 3:
            print("需要三个点")
            return False
        
        confirm_chars = ['A', 'B', 'C']
        
        for i, (x, y) in enumerate(points):
            self.send_grasp_point(x, y, 0)
            print(f"等待机械臂确认 {confirm_chars[i]}...")
            # 这里简化，实际应该等待确认
            time.sleep(0.5)
        
        return True
    
    def _receive_loop(self):
        """接收线程"""
        while self.running:
            with self.conn_lock:
                if not self.client_socket:
                    time.sleep(0.1)
                    continue
                
                try:
                    self.client_socket.settimeout(0.5)
                    data = self.client_socket.recv(1024)
                    if not data:
                        print("\n机械臂断开连接")
                        self.client_socket = None
                        break
                    
                    msg = data.decode()
                    print(f"收到: {msg}")
                    self.message_queue.put(msg)
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"接收错误: {e}")
                    break
    
    def get_message(self, timeout=5):
        """获取一条消息"""
        try:
            return self.message_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        """停止服务器"""
        self.running = False
        with self.conn_lock:
            if self.client_socket:
                try:
                    self.client_socket.close()
                except:
                    pass
                self.client_socket = None
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        print("\n服务器已停止")


# 测试用的机械臂模拟器
class ArmSimulator:
    def __init__(self, server_ip, server_port=8770):
        self.server_ip = server_ip
        self.server_port = server_port
        self.sock = None
        self.running = False
    
    def connect(self):
        """连接到服务器"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.server_ip, self.server_port))
            self.running = True
            print(f"已连接到 {self.server_ip}:{self.server_port}")
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False
    
    def send(self, msg):
        """发送消息"""
        if self.sock:
            try:
                self.sock.sendall(msg.encode())
                print(f"模拟发送: {msg}")
            except Exception as e:
                print(f"发送失败: {e}")
    
    def receive(self):
        """接收消息"""
        if self.sock:
            try:
                self.sock.settimeout(1.0)
                data = self.sock.recv(1024)
                if data:
                    msg = data.decode()
                    print(f"模拟收到: {msg}")
                    return msg
            except socket.timeout:
                return None
            except Exception as e:
                print(f"接收错误: {e}")
        return None
    
    def disconnect(self):
        """断开连接"""
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        print("已断开连接")

