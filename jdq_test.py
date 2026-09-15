import socket
import time
import keyboard
import struct  # 引入 struct 模块用于处理二进制数据

SERVER_PORT = 8770

def start_arm_server():
    print("=" * 40)
    print("Arm Server Waiting...")
    print(f"Port: {SERVER_PORT}")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', SERVER_PORT))
    server_socket.listen(1)

    client_sock, addr = server_socket.accept()
    print(f"Connected from: {addr}")
    client_sock.sendall("START".encode())

    current_state = False

    while True:
        # 1. 判断是否按了 q，按了就退出
        if keyboard.is_pressed('q'):
            print("\nQuitting... Sending exit code (2)...")
            try:
                # 发送二进制整数 2
                client_sock.sendall(struct.pack('i', 2))
                time.sleep(0.2)  # 稍微等一下，确保数据发出
            except Exception as e:
                print(f"Send error: {e}")
            break
            
        # 2. 判断是否按了 空格，按了就切换状态
        elif keyboard.is_pressed('space'):
            current_state = not current_state
            send_code = 0 if current_state else 1  # 直接使用整数
            
            print(f"Sent: {send_code} ({'ON' if current_state else 'OFF'})")
            # 发送二进制整数 ('i' 表示标准的 4 字节整型)
            client_sock.sendall(struct.pack('i', send_code))
            
            # 防止手指按着没松开导致瞬间发几百次
            time.sleep(0.3) 
            
        # 3. 什么都没按，就稍微休息一下，防止CPU占用100%
        else:
            time.sleep(0.01) 

    client_sock.close()
    server_socket.close()

if __name__ == "__main__":
    start_arm_server()