# test_arm_simple.py
import socket
import threading
import time


def handle_client(client_sock, addr):
    print(f"\n{'=' * 60}")
    print(f"机械臂已连接！IP: {addr[0]}")
    print(f"{'=' * 60}")

    try:
        client_sock.sendall("START".encode())
        print("已发送: START")

        while True:
            try:
                client_sock.settimeout(1.0)
                data = client_sock.recv(1024)

                if not data:
                    break

                msg = data.decode()
                print(f"\n收到: [{msg}]")

                if msg == "yes_grab":
                    print("  -> 收到抓取请求，回复测试坐标...")
                    test_coords = ["100,200,0;", "300,400,0;", "500,600,0;"]
                    for coord in test_coords:
                        time.sleep(0.5)
                        try:
                            client_sock.sendall(coord.encode())
                            print(f"  已发送: {coord}")
                        except Exception as e:
                            print(f"  发送失败: {e}")
                elif msg in ["A", "B", "C"]:
                    print(f"  -> 收到确认: {msg}")

            except socket.timeout:
                continue

    except Exception as e:
        print(f"\n错误: {e}")
    finally:
        client_sock.close()
        print(f"\n连接已关闭")


def main():
    print("=" * 60)
    print("机械臂测试服务器")
    print("=" * 60)
    print("请确保机械臂配置连接到本机的 8770 端口\n")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server_socket.bind(("0.0.0.0", 8769))
        server_socket.listen(5)
        print("正在监听 0.0.0.0:8769")
        print("等待机械臂连接...\n")

        while True:
            try:
                server_socket.settimeout(1.0)
                try:
                    client_sock, addr = server_socket.accept()
                    client_thread = threading.Thread(
                        target=handle_client,
                        args=(client_sock, addr),
                        daemon=True
                    )
                    client_thread.start()
                except socket.timeout:
                    continue
            except KeyboardInterrupt:
                print("\n正在停止...")
                break

    except Exception as e:
        print(f"启动失败: {e}")
    finally:
        server_socket.close()
        print("已关闭")


if __name__ == "__main__":
    main()