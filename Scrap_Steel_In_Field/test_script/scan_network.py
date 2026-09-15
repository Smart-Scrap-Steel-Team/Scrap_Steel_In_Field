#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查发现的设备
"""

import requests
import webbrowser
import socket

print("=" * 60)
print("检查发现的设备")
print("=" * 60)

devices = [
    {"ip": "192.168.3.193", "note": "可能是机械臂"},
    {"ip": "192.168.3.151", "note": "另一个摄像头/NVR"},
    {"ip": "192.168.3.10", "note": "你的新摄像头"},
]

for device in devices:
    ip = device["ip"]
    print(f"\n--- 检查 {ip} ({device['note']}) ---")

    # 尝试访问网页界面
    try:
        url = f"http://{ip}"
        print(f"正在访问 {url} ...")
        response = requests.get(url, timeout=3)
        print(f"✓ 网页可访问! 状态码: {response.status_code}")
        print(f"  页面标题或内容: {response.text[:100]}...")

        # 尝试用浏览器打开
        print(f"  正在用浏览器打开...")
        webbrowser.open(url)
    except Exception as e:
        print(f"✗ 无法访问网页: {e}")

    # 尝试扫描更多端口
    print(f"\n正在扫描更多端口...")
    test_ports = [80, 8080, 8000, 8888, 22, 23, 443, 554, 8770, 9000]
    for port in test_ports:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex((ip, port))
            sock.close()
            if result == 0:
                desc = ""
                if port == 80:
                    desc = "(HTTP/网页)"
                elif port == 554:
                    desc = "(RTSP/视频)"
                elif port == 8770:
                    desc = "(机械臂)"
                elif port == 22:
                    desc = "(SSH)"
                elif port == 23:
                    desc = "(Telnet)"
                print(f"✓ 端口 {port} 开放 {desc}")
        except:
            pass

print("\n" + "=" * 60)
print("提示：")
print("- 如果 192.168.3.193 有网页界面，用浏览器打开看看是什么设备")
print("- 机械臂可能需要配置连接到你的电脑 IP 和端口 8770")
print("- 检查机械臂的说明书或标签，看是否有默认 IP")
