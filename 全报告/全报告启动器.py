#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
废钢卸货分析报告 · Python 启动器
一键启动本地Web服务，在浏览器中打开报告页面。
"""

import os
import sys
import webbrowser
import http.server
import socketserver
import threading
import time
import socket
from urllib.parse import urlparse, parse_qs

# ---------- 配置 ----------
PORT = 8080                   # 服务端口，可修改
HTML_FILE = "index.html"      # 报告文件名（请确保与当前目录下的文件名一致）
BROWSER_AUTO_OPEN = True      # 启动后是否自动打开浏览器
# -------------------------

def get_local_ip():
    """获取本机局域网IP（用于显示）"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return '127.0.0.1'

def check_html_file():
    """检查HTML文件是否存在，若不存在则给出提示并退出"""
    if not os.path.isfile(HTML_FILE):
        print(f"❌ 错误: 找不到 '{HTML_FILE}' 文件。")
        print(f"   请确保 '{HTML_FILE}' 与启动器放在同一目录下。")
        print(f"   当前目录: {os.getcwd()}")
        sys.exit(1)

def start_server(port, html_file):
    """
    启动HTTP服务器，并返回服务器对象和线程
    """
    # 切换到文件所在目录，确保相对路径正确
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 创建自定义Handler，支持默认首页
    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            # 如果请求根路径，重定向到 index.html（或指定的HTML文件）
            if self.path == '/':
                self.path = '/' + html_file
            # 调用父类处理
            return super().do_GET()

        # 增加日志前缀，更美观
        def log_message(self, format, *args):
            sys.stdout.write(f"[{time.strftime('%H:%M:%S')}] {format % args}\n")

    # 允许端口复用
    socketserver.TCPServer.allow_reuse_address = True

    try:
        server = socketserver.TCPServer(("", port), CustomHandler)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ 端口 {port} 已被占用，请修改 PORT 变量或关闭占用进程。")
            sys.exit(1)
        else:
            raise

    # 在独立线程中运行服务器
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    return server, server_thread

def open_browser(port):
    """在浏览器中打开报告页面"""
    url = f"http://localhost:{port}/"
    webbrowser.open(url)
    print(f"🌐 已在浏览器中打开: {url}")

def main():
    print("=" * 56)
    print("  🚛 废钢卸货分析报告 · 本地启动器")
    print("=" * 56)

    # 检查HTML文件
    check_html_file()

    # 获取本机IP
    local_ip = get_local_ip()

    # 显示配置信息
    print(f"📄 报告文件: {HTML_FILE}")
    print(f"🔌 服务端口: {PORT}")
    print(f"🌍 本机IP:   {local_ip}")
    print(f"📡 访问地址: http://localhost:{PORT}/")
    if local_ip != '127.0.0.1':
        print(f"   局域网地址: http://{local_ip}:{PORT}/")
    print("-" * 56)

    # 启动服务器
    try:
        server, thread = start_server(PORT, HTML_FILE)
    except Exception as e:
        print(f"❌ 启动服务器失败: {e}")
        sys.exit(1)

    print("✅ 服务器已启动 (按 Ctrl+C 停止)")

    # 自动打开浏览器
    if BROWSER_AUTO_OPEN:
        # 稍微延迟，等待服务器完全就绪
        time.sleep(0.5)
        open_browser(PORT)

    print("\n💡 提示: 在浏览器中打开 http://localhost:{} 查看报告".format(PORT))
    print("   按 Ctrl+C 停止服务器并退出。\n")

    try:
        # 保持主线程运行，等待中断
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 正在关闭服务器...")
        server.shutdown()
        server.server_close()
        print("✅ 服务器已停止。")
        sys.exit(0)

if __name__ == "__main__":
    main()