
import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth
import urllib3
import sys
import os

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class Camera:
    def __init__(self, camera_ip, username, password):
        self.camera_ip = camera_ip
        self.username = username
        self.password = password
        self.camera_timeout = 5
        self.session = requests.Session()
        self.session.auth = HTTPDigestAuth(username, password)
        self.session.verify = False
        self.session.timeout = self.camera_timeout

    def get_snapshot(self, channel):
        """获取单张图片（HTTP方式）"""
        try:
            snapshot_url = f"http://{self.camera_ip}/ISAPI/Streaming/channels/{channel:02d}01/picture?snapShotImageType=JPEG"
            response = self.session.get(snapshot_url, timeout=self.camera_timeout)
            if response.status_code == 200:
                image_data = np.frombuffer(response.content, dtype=np.uint8)
                frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                return frame
        except Exception as e:
            print(f"HTTP方式失败: {e}")
        return None

    def get_snapshot_rtsp(self, channel, rtsp_path):
        """获取单张图片（RTSP方式）"""
        try:
            rtsp_url = f"rtsp://{self.username}:{self.password}@{self.camera_ip}:554{rtsp_path}"
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                return frame
        except Exception as e:
            print(f"RTSP方式失败: {e}")
        return None

