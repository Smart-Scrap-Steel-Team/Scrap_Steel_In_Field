import socket
import threading
import time
import sys


class ArmTestServer:
    def __init__(self, host='0.0.0.0', port=8770):
        self.host = host
        self.port = port
        self.client_socket = None
        self.client_address = None
        self.running = False
        self.server_socket = None

    def start(self):
        print("=" * 60)
        print("机械臂测试服务器")
        print("=" * 60)
        print(f"本机IP (机械臂要连接的IP): 192.168.3.193")
        print(f"监听端口: {self.port}")
        print("\n请在机械臂上配置：")
        print("  - 服务器IP: 192.168.3.193")
        print("  - 端口: 8770")
        print("=" * 60)

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)

        self.running = True

        print("\n等待机械臂连接...")
        while self.running:
            try:
                client_sock, addr = self.server_socket.accept()
                self.client_socket = client_sock
                self.client_address = addr
                print(f"\n{'=' * 60}")
                print(f"机械臂已连接！IP: {addr[0]}")
                print(f"{'=' * 60}")

                # 发送START
                self._send("START")

                # 启动接收线程
                recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
                recv_thread.start()

                # 启动交互界面
                self._interactive_menu()

                break

            except socket.timeout:
                continue
            except KeyboardInterrupt:
                print("\n正在停止...")
                self.running = False
                break
            except Exception as e:
                print(f"错误: {e}")
                continue

        self.stop()

    def _get_local_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"

    def _send(self, msg):
        if self.client_socket:
            try:
                self.client_socket.sendall(msg.encode())
                print(f"[发送] {msg}")
            except Exception as e:
                print(f"[发送失败] {e}")

    def _receive_loop(self):
        buffer = ""
        while self.running and self.client_socket:
            try:
                self.client_socket.settimeout(1.0)
                data = self.client_socket.recv(1024)
                if not data:
                    print("\n机械臂断开连接")
                    self.client_socket = None
                    break

                msg = data.decode()
                buffer += msg

                # 处理消息
                while buffer:
                    if buffer == "yes_grab":
                        print(f"\n[收到] {buffer}")
                        print("机械臂请求抓取坐标！")
                        buffer = ""
                    elif buffer in ["A", "B", "C"]:
                        print(f"\n[收到] {buffer} (确认)")
                        buffer = ""
                    else:
                        print(f"\n[收到] {buffer}")
                        buffer = ""

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"\n接收错误: {e}")
                break

    def _interactive_menu(self):
        print("\n" + "=" * 60)
        print("交互菜单")
        print("=" * 60)

        while self.running and self.client_socket:
            print("\n请选择操作:")
            print("1. 发送单个坐标")
            print("2. 发送3个坐标 (自动等A/B/C确认)")
            print("3. 输入测试坐标序列")
            print("q. 退出")

            choice = input("\n请输入选项 (1/2/3/q): ").strip().lower()

            if choice == "q":
                print("正在退出...")
                self.running = False
                break
            elif choice == "1":
                self._send_single_coord()
            elif choice == "2":
                self._send_three_coords()
            elif choice == "3":
                self._send_test_sequence()
            else:
                print("无效选项，请重试")

    def _send_single_coord(self):
        print("\n--- 发送单个坐标 ---")
        print("格式: x,y (例如: 100,200)")

        coord = input("请输入坐标: ").strip()
        if "," not in coord:
            print("格式错误！")
            return

        try:
            x, y = coord.split(",")
            x = float(x.strip())
            y = float(y.strip())
            msg = f"{x},{y},0;"
            self._send(msg)
        except:
            print("坐标格式错误！")

    def _send_three_coords(self):
        print("\n--- 发送3个坐标 ---")
        coords = []

        for i in range(3):
            label = ["第一个", "第二个", "第三个"][i]
            confirm = ["A", "B", "C"][i]
            coord = input(f"请输入{label}坐标 (x,y): ").strip()
            coords.append(coord)

        print("\n开始发送...")

        for i, coord in enumerate(coords):
            try:
                x, y = coord.split(",")
                x = float(x.strip())
                y = float(y.strip())
                msg = f"{x},{y},0;"
                confirm_char = ["A", "B", "C"][i]

                self._send(msg)
                print(f"等待 {confirm_char} 确认...")

                # 这里简化处理，实际项目中需要等确认消息
                time.sleep(1)

            except:
                print(f"坐标格式错误: {coord}")
                break

    def _send_test_sequence(self):
        print("\n--- 测试坐标序列 ---")
        print("使用默认测试坐标:")
        print("  1: 100,200")
        print("  2: 300,400")
        print("  3: 500,600")

        use_default = input("使用默认坐标? (y/n): ").strip().lower()

        if use_default == "y":
            coords = ["100,200", "300,400", "500,600"]
        else:
            coords = []
            for i in range(3):
                coord = input(f"请输入第{i + 1}个坐标 (x,y): ").strip()
                coords.append(coord)

        print("\n开始发送测试序列...")

        for i, coord in enumerate(coords):
            try:
                x, y = coord.split(",")
                x = float(x.strip())
                y = float(y.strip())
                msg = f"{x},{y},0;"
                self._send(msg)
                time.sleep(1)
            except:
                print(f"坐标格式错误: {coord}")
                break

    def stop(self):
        self.running = False
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        print("\n服务器已停止")


if __name__ == "__main__":
    server = ArmTestServer()
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n被用户中断")
        server.stop()
    except Exception as e:
        print(f"\n错误: {e}")
        server.stop()
