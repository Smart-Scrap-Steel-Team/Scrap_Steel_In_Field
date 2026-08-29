from logger_config import logger
import cv2
import numpy as np
import time
import requests
from requests.auth import HTTPDigestAuth
import urllib3
# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class Camera:
    def __init__(self, nvr_ip, nvr_username, nvr_password):
        self.channel = None
        self.nvr_ip = nvr_ip
        self.nvr_username = nvr_username
        self.nvr_password = nvr_password
        # 超时时间5000ms
        self.camera_timeout = 3  # 修改为秒
        self.cap = None  # 视频捕获对象
        # 设置请求会话
        self.session = requests.Session()
        self.session.auth = HTTPDigestAuth(nvr_username, nvr_password)
        self.session.verify = False  # 忽略SSL证书验证
        self.session.timeout = self.camera_timeout  # 设置超时时间
        # 设置重试策略
        retry_strategy = requests.adapters.Retry(
            total=3,  # 最大重试次数
            backoff_factor=1,  # 重试间隔
            status_forcelist=[500, 502, 503, 504]  # 需要重试的HTTP状态码
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get_camera_image(self, channel):
        """
        获取指定通道的摄像头图像
        channel (int): 摄像头通道号 (1-8)
        Returns:numpy.ndarray: 摄像头图像，如果获取失败则返回None
        """
        self.channel = channel
        if not 1 <= self.channel <= 8:
            logger.error(f"通道号必须在1-8之间，当前输入: {self.channel}")
            return None
        try:
            # 使用HTTP方式获取图像，添加高清参数http://192.168.3.168/ISAPI/Streaming/channels/101/picture
            snapshot_url = f"http://{self.nvr_ip}/ISAPI/Streaming/channels/{self.channel:02d}01/picture?snapShotImageType=JPEG"
            logger.debug(f"尝试获取图像，URL: {snapshot_url}")
            
            # 发送请求获取图像
            response = self.session.get(snapshot_url, timeout=self.camera_timeout)
            if response.status_code != 200:
                logger.error(f"获取通道 {self.channel} 图像失败，状态码: {response.status_code}")
                return None
            
            # 将响应内容转换为OpenCV图像
            image_data = np.frombuffer(response.content, dtype=np.uint8)
            frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            
            if frame is None or frame.size == 0:
                logger.error(f"无法解码通道 {self.channel} 的图像数据")
                return None
                
            return frame
        except requests.exceptions.RequestException as e:
            logger.error(f"网络请求错误: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"获取通道 {self.channel} 图像时发生错误: {str(e)}")
            return None

    def start_video_stream(self, channel):
        """
        开始视频流读取
        channel (int): 摄像头通道号 (1-8)
        Returns: generator: 返回一个生成器，每次yield一帧图像
        """
        self.channel = channel
        if not 1 <= self.channel <= 8:
            logger.error(f"通道号必须在1-8之间，当前输入: {self.channel}")
            return None

        try:
            while True:
                try:
                    # 使用HTTP方式获取图像，添加高清参数
                    # 
                    snapshot_url = f"http://{self.nvr_ip}/ISAPI/Streaming/channels/{self.channel:02d}01/picture?snapShotImageType=JPEG"
                    logger.debug(f"尝试获取图像，URL: {snapshot_url}")
                    
                    # 发送请求获取图像
                    response = self.session.get(snapshot_url, timeout=self.camera_timeout)
                    if response.status_code != 200:
                        logger.warning(f"获取通道 {self.channel} 图像失败，状态码: {response.status_code}")
                        # time.sleep(1)  # 等待1秒后重试
                        continue
                    
                    # 将响应内容转换为OpenCV图像
                    image_data = np.frombuffer(response.content, dtype=np.uint8)
                    frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                    
                    if frame is None or frame.size == 0:
                        logger.warning(f"无法解码通道 {self.channel} 的图像数据")
                        # time.sleep(1)  # 等待1秒后重试
                        continue

                    yield frame
                    time.sleep(0.2)
                except requests.exceptions.RequestException as e:
                    logger.error(f"网络请求错误: {str(e)}")
                    time.sleep(1)  # 发生错误时等待1秒后重试
                except Exception as e:
                    logger.error(f"读取视频流时发生错误: {str(e)}")
                    time.sleep(1)  # 发生错误时等待1秒后重试
        except Exception as e:
            logger.error(f"启动视频流时发生错误: {str(e)}")
            return None
        
    def get_camera_image_by_rtspurl(self, channel=None):
        """
        通过RTSP URL获取摄像头图像
        Args:
            channel (int): 摄像头通道号 (1-8)
        Returns:
            numpy.ndarray: 摄像头图像帧，如果获取失败则返回None
        """
        if channel is not None:
            self.channel = channel
            
        if self.channel is None or not 1 <= self.channel <= 9:
            logger.error(f"通道号必须在1-8之间，当前输入: {self.channel}")
            return None
            
        # 使用海康威视的H.265 RTSP URL格式
        # rtsp_url = f"rtsp://{self.nvr_username}:{self.nvr_password}@{self.nvr_ip}:554/Streaming/Channels/{self.channel:02d}01"
        # rtsp://192.168.3.14:554/user=admin_password=0CJbuXuI_channel=0_stream=0&onvif=5.sdp?real_stream
        rtsp_url = f"rtsp://192.168.3.14:554/user=admin_password=0CJbuXuI_channel=0_stream=0&onvif=5.sdp?real_stream"
        logger.info(f"尝试连接RTSP URL: {rtsp_url}")
        
        cap = None
        try:
            # 设置OpenCV参数
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                logger.error(f"无法打开RTSP URL: {rtsp_url}")
                return None
                
            # 设置H.265解码参数
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲区大小
            cap.set(cv2.CAP_PROP_FPS, 30)  # 设置帧率
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '5'))  # 设置H.265编码
            
            retry_count = 0
            max_retries = 3
            retry_interval = 0.2  # 秒
            
            while retry_count < max_retries:
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning(f"无法读取帧，重试次数: {retry_count + 1}")
                    retry_count += 1
                    time.sleep(retry_interval)
                    continue
                
                # 检查图像是否有效
                if frame is None or frame.size == 0:
                    logger.warning("获取到空帧，重试...")
                    retry_count += 1
                    time.sleep(retry_interval)
                    continue
                
                # 成功获取有效图像
                return frame
        except Exception as e:
            logger.error(f"获取图像过程中发生错误: {str(e)}")
        finally:
            if cap is not None:
                cap.release()
        logger.error(f"达到最大重试次数({max_retries})，获取图像失败")
        return None

    def get_camera_stream(self, channel=None):
        """
        获取摄像头数据流
        Args:
            channel (int): 摄像头通道号 (1-8)
        Yields:
            numpy.ndarray: 摄像头图像帧
        """
        if channel is not None:
            self.channel = channel
            
        if self.channel is None or not 1 <= self.channel <= 9:
            logger.error(f"通道号必须在1-8之间，当前输入: {self.channel}")
            return
            
        # 使用海康威视的H.265 RTSP URL格式
        rtsp_url = f"rtsp://{self.nvr_username}:{self.nvr_password}@{self.nvr_ip}:554/Streaming/Channels/{self.channel:02d}01"
        logger.info(f"尝试连接RTSP URL: {rtsp_url}")
        
        # 设置OpenCV参数
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.error(f"无法打开RTSP URL: {rtsp_url}")
            return
            
        # 设置H.265解码参数
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲区大小
        cap.set(cv2.CAP_PROP_FPS, 30)  # 设置帧率
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '5'))  # 设置H.265编码
        
        retry_count = 0
        max_retries = 3
        retry_interval = 1  # 秒
        last_frame_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                current_time = time.time()
                
                if not ret:
                    logger.warning(f"无法读取帧，重试次数: {retry_count + 1}")
                    retry_count += 1
                    
                    if retry_count >= max_retries:
                        logger.error("达到最大重试次数，重新建立连接")
                        # 重新建立连接
                        cap.release()
                        time.sleep(retry_interval)
                        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                        if not cap.isOpened():
                            logger.error("重新连接失败")
                            break
                        retry_count = 0
                    else:
                        time.sleep(retry_interval)
                        continue
                
                # 检查帧率
                if current_time - last_frame_time < 0.033:  # 约30fps
                    time.sleep(0.033 - (current_time - last_frame_time))
                
                retry_count = 0  # 成功读取帧后重置重试计数
                last_frame_time = current_time
                yield frame
                
        except Exception as e:
            logger.error(f"视频流处理过程中发生错误: {str(e)}")
        finally:
            if cap:
                cap.release()


# 使用示例
if __name__ == "__main__":
    # 获取通道1的图像
    channel = 1
    # 海康威视NVR连接参数 
    # 1：四号相机（出场地磅） 3：三号相机（出场闸机） 4：七号摄像头（场外堆料区） 
    # 5：二号摄像头（安全生产） 6：一号摄像头（火焰烟雾监测） 
    # 7：五号摄像头（进场地磅） 8：六号摄像头（进场车牌识别） 
    nvr_ip = "192.168.3.13"
    nvr_username = "admin"
    nvr_password = "Xd2025328"
    camera = Camera(nvr_ip, nvr_username, nvr_password)
    
    # 测试RTSP URL获取
    try:
        image = camera.get_camera_image_by_rtspurl(3)
        if image is not None and not image.size == 0:
            cv2.imshow("Video Stream", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            logger.error("获取到的图像为空或无效")
    except Exception as e:
        logger.error(f"显示图像时发生错误: {str(e)}")
    finally:
        cv2.destroyAllWindows()
