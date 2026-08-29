import requests

def on_field_light(light_state=True):
    """
    控制场地灯光的接口函数
    
    参数:
        light_state (bool): 灯光状态，True表示开启，False表示关闭
    
    返回:
        dict: API响应的JSON数据
    """
    # API基础URL
    base_url = "http://192.168.3.204:5000/remote/on_field_light"
    
    # 构建请求参数
    params = {
        "field_light_state": str(light_state).lower()
    }
    
    try:
        # 发送GET请求
        response = requests.get(base_url, params=params, timeout=5)
        
        # 检查响应状态
        response.raise_for_status()
        
        # 返回JSON响应
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return {"error": str(e), "success": False}

def off_field_light(light_state=True):
    """
    控制场外灯光的接口函数
    
    参数:
        light_state (bool): 灯光状态，True表示开启，False表示关闭
    
    返回:
        dict: API响应的JSON数据
    """
    # API基础URL
    base_url = "http://192.168.3.204:5000/remote/off_field_light"
    
    # 构建请求参数
    params = {
        "field_light_state": str(light_state).lower()
    }
    
    try:
        # 发送GET请求
        response = requests.get(base_url, params=params, timeout=5)
        
        # 检查响应状态
        response.raise_for_status()
        
        # 返回JSON响应
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return {"error": str(e), "success": False}

# 使用示例
if __name__ == "__main__":
    # 开启场内灯光
    result_on = on_field_light(True)
    print(f"开启场内灯光结果: {result_on}")
    
    # 开启场外灯光
    result_off_on = off_field_light(True)
    print(f"开启场外灯光结果: {result_off_on}")
    
    # 关闭场内灯光
    # result_off = on_field_light(False)
    # print(f"关闭场内灯光结果: {result_off}")
    
    # 关闭场外灯光
    # result_off_off = off_field_light(False)
    # print(f"关闭场外灯光结果: {result_off_off}")
