#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的机械臂TCP服务器测试脚本
监听8770端口，显示机械臂发送的所有消息
"""

import socket
import threading
import time


def handle_client(client_socket, client_address):
    """处理单个机械臂连接"""
    print(f"\n{'=' * 60}")
    print(f"机械臂已连接！IP: {client_address[0]}")
    print(f"{'=' * 60}")

    # 发送START消息（和项目代码一致）
    try:
        client_socket.sendall("START".encode())
        print("已发送: START")
    except Exception as e:
        print(f"发送START失败: {e}")

    try:
        while True:
            # 接收数据
            client_socket.settimeout(1.0)
            try:
                data = client_socket.recv(1024)
                if not data:
                    print("\n机械臂断开连接")
                    break

                msg = data.decode()
                print(f"\n收到消息: [{msg}]")

                # 识别消息类型
                if msg == "yes_grab":
                    print("  -> 这是请求拍照/抓取的信号！")
                    # 回复一些测试坐标
                    test_coords = [
                        "100,200,0;",
                        "300,400,0;",
                        "500,600,0;"
                    ]
                    print("  -> 发送测试坐标...")
                    for coord in test_coords:
                        time.sleep(0.5)
                        try:
                            client_socket.sendall(coord.encode())
                            print(f"     已发送: {coord}")
                        except Exception as e:
                            print(f"     发送失败: {e}")

                elif msg in ["A", "B", "C"]:
                    print(f"  -> 这是抓取完成确认信号 {msg}！")

                else:
                    print(f"  -> 未知消息")

            except socket.timeout:
                continue

    except Exception as e:
        print(f"\n连接出错: {e}")
    finally:
        client_socket.close()
        print(f"连接已关闭")


def main():
    host = "0.0.0.0"
    port = 8769

    print("=" * 60)
    print("机械臂测试服务器")
    print("=" * 60)
    print(f"正在监听: {host}:{port}")
    print("等待机械臂连接...")
    print()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server_socket.bind((host, port))
        server_socket.listen(5)

        while True:
            try:
                server_socket.settimeout(1.0)
                try:
                    client_socket, client_address = server_socket.accept()

                    # 在新线程中处理连接
                    client_thread = threading.Thread(
                        target=handle_client,
                        args=(client_socket, client_address),
                        daemon=True
                    )
                    client_thread.start()

                except socket.timeout:
                    continue

            except KeyboardInterrupt:
                print("\n正在停止服务器...")
                break

    except Exception as e:
        print(f"服务器错误: {e}")
    finally:
        server_socket.close()
        print("服务器已关闭")


if __name__ == "__main__":
    main()
