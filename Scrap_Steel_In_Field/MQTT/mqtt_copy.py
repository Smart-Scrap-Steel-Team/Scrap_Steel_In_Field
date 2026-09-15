import paho.mqtt.client as mqtt
import time

class MQTTClient:
    def __init__(self, broker="192.168.3.144", port=1883, client_id=None, topics=["Channel1"], qos=1):
        self.broker = broker
        self.port = port
        self.client_id = client_id
        self.topics = topics  # 支持多个主题
        self.qos = qos
        self.last_message = None  # 记录上一条消息，用于连续去重
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id, clean_session=True)
        self.message_callback = None  # 用户自定义消息处理回调函数
        
        # 设置回调函数
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_publish = self.on_publish
        self.client.on_subscribe = self.on_subscribe

    def set_message_callback(self, callback):
        """
        设置自定义消息处理回调函数
        
        Args:
            callback: 回调函数，接收参数 (topic, content)
        """
        self.message_callback = callback
        
    def on_message(self, client, userdata, message):
        try:
            # 直接获取纯文本消息
            content = message.payload.decode()
            
            # 检查是否是连续重复的消息
            if content == self.last_message:
                # print(f"设备 {self.client_id} 忽略连续重复消息: {content}")
                return
            
            # 打印接收消息的详细信息
            # print(f"设备 {self.client_id} 收到消息（主题: {message.topic}）: {content}")
            self.last_message = content
            
            # 调用用户自定义的消息处理函数
            if self.message_callback:
                self.message_callback(message.topic, content)
                
        except Exception as e:
            print(f"设备 {self.client_id} 处理消息失败: {e}")

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            # print(f"设备 {self.client_id} 成功连接到MQTT服务器！")
            # 订阅所有指定的主题
            for topic in self.topics:
                self.client.subscribe(topic, qos=self.qos)
        else:
            print(f"设备 {self.client_id} 连接失败，错误码: {rc}")

    def on_publish(self, client, userdata, mid):
        print(f"设备 {self.client_id} MQTT启动消息发布成功，消息ID: {mid}")

    def on_subscribe(self, client, userdata, mid, granted_qos):
        # print(f"设备 {self.client_id} 成功订阅主题，QoS: {granted_qos}")
        pass

    def connect(self):
        """连接到MQTT服务器并启动消息循环"""
        self.client.connect(self.broker, self.port)
        self.client.loop_start()
        # print(f"设备 {self.client_id} 已启动")

    def publish(self, message, topic=None, retain=False):
        """
        发布消息到指定主题，若未指定则发布到所有初始化主题
        
        Args:
            message: 消息内容
            topic: 可选，指定发布到哪个主题
            retain: 是否设置为保留消息，默认False
        """
        # 直接发送消息文本
        message_text = str(message)
        
        # 如果指定了主题，只发布到该主题
        if topic:
            self.client.publish(topic, message_text, qos=self.qos, retain=retain)
            print(f"设备 {self.client_id} 向主题 {topic} 发布消息 {'(保留)' if retain else ''}")
        else:
            # 否则发布到所有初始化时的主题
            for t in self.topics:
                self.client.publish(t, message_text, qos=self.qos, retain=retain)
                print(f"设备 {self.client_id} 向主题 {t} 发布消息 {'(保留)' if retain else ''}")
        time.sleep(0.1)

    def disconnect(self):
        """断开连接并清理资源"""
        self.client.loop_stop()
        self.client.disconnect()
        print(f"设备 {self.client_id} 已断开连接")

class MessageProcessor:
    """简化的MQTT消息处理器"""
    
    def __init__(self, broker="192.168.3.144", port=1883, topic="Channel1", client_id=None):
        """初始化消息处理器"""
        if client_id is None:
            client_id = f"processor_{int(time.time())}"
            
        self.client = MQTTClient(
            broker=broker,
            port=port,
            client_id=client_id,
            topics=[topic] if isinstance(topic, str) else topic
        )
        self.client.set_message_callback(self._on_message)
        self.handlers = {}
    
    def _on_message(self, topic, content):
        """内部消息处理函数"""
        # 查找并执行对应主题的处理函数
        if topic in self.handlers:
            # 传递消息内容
            return self.handlers[topic](content)
    
    def register_handler(self, topic, handler):
        """注册特定主题的消息处理函数
        
        处理函数格式应为: handler(content)
        处理函数可以返回值，通常用于返回处理后的消息内容
        """
        self.handlers[topic] = handler
        return self  # 支持链式调用
    
    def publish(self, message, topic=None, retain=False):
        """发布消息到指定主题
        
        Args:
            message: 要发布的消息内容
            topic: 可选，指定发布到哪个主题，如果不指定则发布到所有初始化时的主题
            retain: 是否设置为保留消息，默认False
        """
        self.client.publish(message, topic, retain=retain)
    
    def start(self):
        """启动消息处理器"""
        self.client.connect()
        # print(f"MQTT处理器已启动，正在监听...")
        
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            # print("\n处理器已停止")
            pass
        finally:
            self.client.disconnect()

# 创建全局MQTT处理器实例
mqtt_processor = MessageProcessor(
    broker="192.168.3.144",
    port=1883,
    topic=["Channel1"],
    client_id="arm1"
)

def process_message(content):
    print(f"收到MQTT消息: {content}")
    return content

# MQTT处理器线程函数
def mqtt_receive():
    mqtt_processor.register_handler("Channel1", process_message)
    # 启动处理器
    mqtt_processor.start()  # 这是阻塞调用，会一直运行

# MQTT发送消息函数
def mqtt_send(message, retain=False):
    try:
        mqtt_processor.publish(message, retain=retain)
    except Exception as e:
        print(f"发送MQTT消息时出错: {e}")
        
if __name__ == "__main__":
    # main()
    # simple_processor_example()
    # mqtt_receive()
    mqtt_send("XD0001:5")
