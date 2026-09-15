#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查网络配置，查看你电脑的IP地址
"""

import socket
import subprocess
import platform

def get_local_ip():
    """获取本机IP地址"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def check_ports():
    """检查8770端口是否被占用"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', 8770))
    sock.close()
    return result == 0

print("="*60)
print("网络配置检查")
print("="*60)

local_ip = get_local_ip()
print(f"\n你电脑的IP地址: {local_ip}")
print(f"   -> 机械臂需要连接到这个IP的8770端口")

port_used = check_ports()
print(f"\n8770端口状态: {'已被占用' if port_used else '空闲'}")

print("\n" + "="*60)
print("配置机械臂时，请设置：")
print("="*60)
print(f"  服务器IP: {local_ip}")
print(f"  端口: 8770")
print()
print("然后运行 test_arm_server.py 来测试连接！")
print()

# Windows上查看网络连接
if platform.system() == "Windows":
    try:
        print("当前网络连接:")
        result = subprocess.run(
            ['netstat', '-ano'],
            capture_output=True,
            text=True
        )
        for line in result.stdout.split('\n'):
            if ':8770' in line or '8770' in line:
                print(f"  {line.strip()}")
    except:
        pass
