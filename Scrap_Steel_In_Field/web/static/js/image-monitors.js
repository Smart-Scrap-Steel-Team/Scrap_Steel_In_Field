// 检查是否有新监控图片
function checkForNewMonitorImage() {
    fetch('update_image', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            updateMonitorImage(data.path);
            console.log("监控图片已更新: " + new Date().toLocaleTimeString());
        } else {
            console.log("没有监控图片: " + data.message);
            document.getElementById('monitorImage').innerHTML = 
                `<h3>实时监控</h3><p>等待图像加载...</p>`;
        }
    })
    .catch(error => {
        console.error("检查监控更新失败:", error);
        document.getElementById('monitorImage').innerHTML = 
            `<h3>实时监控</h3><p>等待图像加载...</p>`;
    });
}

// ==================== 实时视频流功能 ====================

// 车辆图像实时视频流
let liveStreamInterval = null;
let isLiveStreamActive = false;

// 启动车辆图像实时视频流（使用MJPEG流方式）
function startLiveVehicleStream() {
    const vehicleImageDiv = document.getElementById('vehicleImage');
    if (!vehicleImageDiv) return;
    
    // 使用MJPEG流直接显示
    vehicleImageDiv.innerHTML = `
        <h3>车辆图像 - 实时监控</h3>
        <img id="vehicleLiveStream" src="/get_live_stream" 
             alt="车辆实时视频" 
             style="width:100%; height:calc(100% - 30px); object-fit:contain;"
             onerror="handleLiveStreamError()">
    `;
    isLiveStreamActive = true;
    console.log("车辆实时视频流已启动");
}

// 使用轮询方式获取实时帧（备用方案）
function startLiveVehicleFrameUpdate() {
    const vehicleImageDiv = document.getElementById('vehicleImage');
    if (!vehicleImageDiv) return;
    
    // 创建图像元素
    vehicleImageDiv.innerHTML = `
        <h3>车辆图像 - 实时监控</h3>
        <img id="vehicleLiveFrame" src="" 
             alt="车辆实时画面" 
             style="width:100%; height:calc(100% - 30px); object-fit:contain;">
    `;
    
    isLiveStreamActive = true;
    
    // 立即更新第一帧
    updateLiveVehicleFrame();
    
    // 每200ms更新一帧（约5fps）
    liveStreamInterval = setInterval(updateLiveVehicleFrame, 200);
    console.log("车辆实时帧更新已启动");
}

// 更新单帧图像
function updateLiveVehicleFrame() {
    const imgElement = document.getElementById('vehicleLiveFrame');
    if (!imgElement || !isLiveStreamActive) return;
    
    // 添加时间戳防止缓存
    imgElement.src = `/get_live_frame?t=${new Date().getTime()}`;
}

// 停止实时视频流
function stopLiveVehicleStream() {
    isLiveStreamActive = false;
    if (liveStreamInterval) {
        clearInterval(liveStreamInterval);
        liveStreamInterval = null;
    }
    console.log("车辆实时视频流已停止");
}

// 处理视频流错误
function handleLiveStreamError() {
    console.error("视频流加载失败，尝试使用备用方案");
    // 如果MJPEG流失败，切换到轮询方式
    setTimeout(() => {
        if (isLiveStreamActive) {
            startLiveVehicleFrameUpdate();
        }
    }, 1000);
}

// 检查摄像头状态
function checkCameraStatus() {
    fetch('/check_camera_status')
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log("摄像头状态:", data.camera_online ? "在线" : "离线");
        } else {
            console.error("检查摄像头状态失败:", data.error);
        }
    })
    .catch(error => {
        console.error("检查摄像头状态请求失败:", error);
    });
}

// 检查是否有新饼图
function checkForNewPieChart() {
    fetch('update_pie_chart', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            updatePieChart(data.path);
            console.log("饼图已更新: " + new Date().toLocaleTimeString());
        } else {
            console.log("没有饼图: " + data.message);
            document.getElementById('pieChart').innerHTML = 
                `<h3>危险品分布</h3><p>等待图像加载...</p>`;
        }
    })
    .catch(error => {
        console.error("检查饼图更新失败:", error);
        document.getElementById('pieChart').innerHTML = 
            `<h3>危险品分布</h3><p>等待图像加载...</p>`;
    });
}

// 检查是否有新废钢比例图
function checkForNewScrapChart() {
    fetch('update_scrap_chart', {
        method: 'POST'
    })
}

// 检查是否有新危险品检测图片
function checkForNewDangerImage() {
    fetch('update_detection', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            updateDangerImage(data.path);
            console.log("危险品检测图片已更新: " + new Date().toLocaleTimeString());
        } else {
            console.log("没有危险品检测图片: " + data.message);
            document.getElementById('dangerImage').innerHTML = 
                `<h3>危险品检测</h3><p>等待图像加载...</p>`;
        }
    })
    .catch(error => {
        console.error("检查危险品检测更新失败:", error);
        document.getElementById('dangerImage').innerHTML = 
            `<h3>危险品检测</h3><p>等待图像加载...</p>`;
    });
}

// 更新监控图片显示
function updateMonitorImage(path) {
    let imgElement = document.getElementById('monitorImg');
    if(imgElement) {
        imgElement.src = path + "?t=" + new Date().getTime();
    } else {
        document.getElementById('monitorImage').innerHTML = 
            `<img id="monitorImg" src="${path}?t=${new Date().getTime()}" 
             alt="实时监控" style="width:100%; height:100%; object-fit:contain;">`;
    }
}

    // 更新饼图显示
    function updatePieChart(path) {
        let imgElement = document.getElementById('pieChartImg');
        if(imgElement) {
            imgElement.src = path + "?t=" + new Date().getTime();
        } else {
            document.getElementById('pieChart').innerHTML = 
                `<img id="pieChartImg" src="${path}?t=${new Date().getTime()}" 
                alt="危险品分布饼图" style="width:100%; height:100%; object-fit:contain;">`;
        }
    }

// 更新危险品检测图片显示
function updateDangerImage(path) {
    let imgElement = document.getElementById('dangerImg');
    if(imgElement) {
        imgElement.src = path + "?t=" + new Date().getTime();
    } else {
        document.getElementById('dangerImage').innerHTML = 
            `<img id="dangerImg" src="${path}?t=${new Date().getTime()}" 
             alt="危险物识别" style="width:100%; height:100%; object-fit:contain;">`;
    }
}

// 直接加载JSON文件（优化版本）
function directLoadJsonFile() {
    fetch('/vehicle_records/vehicles.json')  // 使用正确的路径
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP错误! 状态: ${response.status}`);
        }
        return response.text();
    })
    .then(text => {
        console.log("成功获取JSON文件，长度:", text.length);
        if (text.trim() === '') {
            document.getElementById('jsonData').innerHTML = '<p>没有数据</p>';
            document.getElementById('jsonHistory').innerHTML = '<p>没有历史数据</p>';
            return;
        }
        
        const lines = text.trim().split('\n');
        const jsonObjects = [];
        
        for (const line of lines) {
            if (line.trim()) {
                try {
                    const obj = JSON.parse(line);
                    jsonObjects.push(obj);
                } catch (e) {
                    console.error("解析JSON失败:", e, "行内容:", line);
                }
            }
        }
        
        if (jsonObjects.length > 0) {
            // 显示最新一条记录
            const latestData = jsonObjects[jsonObjects.length - 1];
            
            // 创建最新记录的表格
            let dataHtml = '<h3></h3><table class="vehicle-table">';
            
            // 添加中文字段名映射
            const fieldNames = {
                'plate_number': '车牌号',
                'time': '时间',
                'status': '状态'
            };
            
            // 添加状态映射
            const statusMap = {
                'arrive': '到达',
                'leave': '离开'
            };
            
            for (const key in latestData) {
                let value = latestData[key];
                
                // 对状态字段进行翻译
                if (key === 'status' && statusMap[value]) {
                    value = statusMap[value];
                }
                
                dataHtml += `<tr>
                    <td><strong>${fieldNames[key] || key}</strong></td>
                    <td>${value}</td>
                </tr>`;
            }
            dataHtml += '</table>';
            document.getElementById('jsonData').innerHTML = dataHtml;
            
            // 显示历史记录
            let historyHtml = '<h3></h3><table class="vehicle-history-table">';
            
            // 表头
            historyHtml += '<tr>';
            for (const key in fieldNames) {
                historyHtml += `<th>${fieldNames[key]}</th>`;
            }
            historyHtml += '</tr>';
            
            // 最多显示10条历史记录，按时间倒序排列
            const historyRecords = jsonObjects.slice(-10).reverse();
            for (const record of historyRecords) {
                historyHtml += '<tr>';
                for (const key in fieldNames) {
                    let value = record[key] || '';
                    
                    // 对状态字段进行翻译
                    if (key === 'status' && statusMap[value]) {
                        value = statusMap[value];
                    }
                    
                    historyHtml += `<td>${value}</td>`;
                }
                historyHtml += '</tr>';
            }
            historyHtml += '</table>';
            document.getElementById('jsonHistory').innerHTML = historyHtml;
            
            console.log("成功显示JSON数据，共", jsonObjects.length, "条记录");
        } else {
            document.getElementById('jsonData').innerHTML = '<p>没有有效数据</p>';
            document.getElementById('jsonHistory').innerHTML = '<p>没有有效历史数据</p>';
        }
    })
    .catch(error => {
        console.error("加载JSON文件失败:", error);
        document.getElementById('jsonData').innerHTML = '<p>加载数据失败</p>';
        document.getElementById('jsonHistory').innerHTML = '<p>加载历史数据失败</p>';
    });
}

// 页面加载完成后初始化所有监控
window.onload = function() {
    // 初始加载所有图片和数据
    checkForNewMonitorImage();
    checkForNewPieChart();
    checkForNewDangerImage();
    directLoadJsonFile();  // 直接加载JSON
    
    // 启动车辆图像实时视频流（默认使用MJPEG流方式）
    startLiveVehicleStream();
    
    // 检查摄像头状态
    checkCameraStatus();
    
    // 每秒更新一次所有图片和数据
    setInterval(checkForNewMonitorImage, 1000);
    setInterval(checkForNewPieChart, 1000);
    setInterval(checkForNewDangerImage, 1000);
    setInterval(directLoadJsonFile, 5000);  // 每5秒更新一次JSON数据
    
    // 每30秒检查一次摄像头状态
    setInterval(checkCameraStatus, 30000);
}; 