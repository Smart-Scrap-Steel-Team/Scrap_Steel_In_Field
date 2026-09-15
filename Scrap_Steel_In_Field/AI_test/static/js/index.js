// 获取当前时间并格式化为 YYYY-MM-DD HH:mm:ss 格式
function formatDate(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    const seconds = String(date.getSeconds()).padStart(2, '0');
    return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
}

// 更新页面上的时间
function updateTime() {
    const now = new Date();
    document.getElementById('headtime').innerText = formatDate(now);
}

// 初始化时立即调用一次
updateTime();
// 每秒更新一次时间
setInterval(updateTime, 1000);

// 更新危险物数量显示
function updateDangerCounts() {
    fetch('/get_latest_danger_counts')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const counts = data.counts;
                document.getElementById('gasCyl1Count').textContent = counts.GasCyl1;
                document.getElementById('gasCyl2Count').textContent = counts.GasCyl2;
                document.getElementById('fireExt1Count').textContent = counts.FireExt1;
                document.getElementById('fireExt2Count').textContent = counts.FireExt2;
            }
        })
        .catch(error => console.error('获取危险物数量失败:', error));
}

// 更新危险物总数显示
function updateDangerTotal() {
    fetch('/get_danger_total')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                document.getElementById('totalDangerCount').textContent = data.total_count;
            }
        })
        .catch(error => console.error('获取危险物总数失败:', error));
}

// 新增变量跟踪机械臂工作状态
let arm1WorkStart = null;  // 记录工作开始时间戳
let progressTimer = null;  // 记录进度条定时器ID

// 更新机械臂状态显示（修改后的函数）
function updateArmStatus() {
    fetch('/arm_status')
        .then(response => response.json())
        .then(data => {
            const statusElement = document.getElementById('arm1Status');
            const progressElement = document.getElementById('arm1Progress');
            const itemElement = statusElement.closest('.arm-status-item'); // 获取父元素 .arm-status-item

            if (data.arm1_status === true) {
                // 进入工作中状态
                statusElement.textContent = '工作';
                if (itemElement) { // 添加 status-on 类，移除 status-off 类
                    itemElement.classList.add('status-on');
                    itemElement.classList.remove('status-off');
                }
                if (!arm1WorkStart) {
                    arm1WorkStart = Date.now();  // 记录开始时间
                    // 启动进度条（总时长改为200秒）
                    const totalDuration = 200000;  // 200秒（200*1000毫秒）
                    progressTimer = setInterval(() => {
                        const elapsed = Date.now() - arm1WorkStart;
                        const progress = Math.min((elapsed / totalDuration) * 100, 100);
                        progressElement.style.width = `${progress}%`;
                        // 进度完成后自动重置（可选）
                        if (progress >= 100) {
                            clearInterval(progressTimer);
                            arm1WorkStart = null;
                            progressElement.style.width = '0%';
                        }
                    }, 100);  // 每100ms更新一次进度
                }
            } else if (data.arm1_status === false) {
                // 进入未启动状态
                statusElement.textContent = '暂停';
                if (itemElement) { // 添加 status-off 类，移除 status-on 类
                    itemElement.classList.add('status-off');
                    itemElement.classList.remove('status-on');
                }
                clearInterval(progressTimer);  // 停止进度更新
                arm1WorkStart = null;          // 重置开始时间
                progressElement.style.width = '0%';  // 进度条归零
            } else {
                // 未知状态
                statusElement.textContent = '未知';
                if (itemElement) { // 移除 status-on 和 status-off 类
                    itemElement.classList.remove('status-on');
                    itemElement.classList.remove('status-off');
                }
                clearInterval(progressTimer);
                arm1WorkStart = null;
                document.getElementById('arm1Progress').style.width = '0%';
            }
        })
        .catch(error => {
            console.error('获取机械臂状态失败:', error);
            document.getElementById('arm1Status').textContent = '未知';
            const itemElement = document.getElementById('arm1Status').closest('.arm-status-item'); // 获取父元素 .arm-status-item
             if (itemElement) { // 移除 status-on 和 status-off 类
                itemElement.classList.remove('status-on');
                itemElement.classList.remove('status-off');
            }
            clearInterval(progressTimer);
            arm1WorkStart = null;
            document.getElementById('arm1Progress').style.width = '0%';
        });
}

// 每2秒更新一次机械臂状态
setInterval(updateArmStatus, 2000);

// 检查是否有新监控图片
function checkForNewMonitorImage() {
  fetch('update_image', {
      method: 'POST'
  })
  .then(response => response.json())
  .then(data => {
      console.log("检查监控图片更新结果: ", data);
      // 无论后端是否报告有新图片，都尝试从 /get_image 加载最新图片
      updateMonitorImage('/get_image');
  })
  .catch(error => {
      console.error("检查监控更新失败:", error);
      document.getElementById('monitorImage').innerHTML = 
          `<h3>实时监控</h3><p>等待图像加载...</p>`;
  });
}  

// 检查是否有新危险品检测图片
function checkForNewDangerImage() {
  fetch('update_detection', {
      method: 'POST'
  })
  .then(response => response.json())
  .then(data => {
      console.log("检查危险品检测图片更新结果: ", data);
      // 无论后端是否报告有新图片，都尝试从 /get_detection 加载最新图片
      updateDangerImage('/get_detection');
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

// 检查是否有新分割图片
function checkForNewSegmentationImage() {
  fetch('update_segmentation', {
      method: 'POST'
  })
  .then(response => response.json())
  .then(data => {
      console.log("检查分割图片更新结果: ", data);
      // 无论后端是否报告有新图片，都尝试从 /get_segmentation 加载最新图片
      updateSegmentationImage('/get_segmentation');
  })
  .catch(error => {
      console.error("检查分割图片更新失败:", error);
      document.getElementById('segmentationImage').innerHTML = 
          `<h3>分割结果</h3><p>等待图像加载...</p>`;
  });
}

// 更新分割图片显示
function updateSegmentationImage(path) {
  let imgElement = document.getElementById('segmentationImg');
  if(imgElement) {
      imgElement.src = path + "?t=" + new Date().getTime();
  } else {
      document.getElementById('segmentationImage').innerHTML = 
          `<img id="segmentationImg" src="${path}?t=${new Date().getTime()}" 
           alt="分割结果" style="width:100%; height:100%; object-fit:contain;">`;
  }
}

// 更新分割统计数据
function updateSegmentationStats() {
    fetch('/get_segmentation_stats')
        .then(response => response.json())
        .then(data => {
            console.log("Debug - Received data:", data);  // 添加调试信息
            if (data.success) {
                const stats = data.stats;
                console.log("Debug - Stats:", stats);  // 添加调试信息
                console.log("Debug - Total count:", data.total_count);  // 添加调试信息
                console.log("Debug - Total area:", data.total_area);  // 添加调试信息

                // 更新方管类
                if (stats['方管类']) { // 添加检查
                    document.getElementById('squareTubeCount').textContent = stats['方管类'].count;
                } else {
                    document.getElementById('squareTubeCount').textContent = 'N/A'; // 或者设置为 0
                }

                // 更新圆管类
                if (stats['圆管类']) { // 添加检查
                    document.getElementById('roundTubeCount').textContent = stats['圆管类'].count;
                } else {
                    document.getElementById('roundTubeCount').textContent = 'N/A';
                }

                // 更新钢板类
                if (stats['钢板类']) { // 添加检查
                    document.getElementById('steelPlateCount').textContent = stats['钢板类'].count;
                } else {
                    document.getElementById('steelPlateCount').textContent = 'N/A';
                }

                // 更新镀锌件
                if (stats['镀锌件']) { // 添加检查
                    document.getElementById('galvanizedCount').textContent = stats['镀锌件'].count;
                } else {
                    document.getElementById('galvanizedCount').textContent = 'N/A';
                }

                // 更新钢筋
                if (stats['钢筋']) { // 添加检查
                    document.getElementById('rebarCount').textContent = stats['钢筋'].count;
                } else {
                    document.getElementById('rebarCount').textContent = 'N/A';
                }

                // 更新管道连接件
                if (stats['管道件']) { // 添加检查
                    document.getElementById('pipeFittingCount').textContent = stats['管道件'].count;
                } else {
                    document.getElementById('pipeFittingCount').textContent = 'N/A';
                }

                // 更新金属连接件
                if (stats['金属件']) { // 添加检查
                    document.getElementById('metalFittingCount').textContent = stats['金属件'].count;
                } else {
                    document.getElementById('metalFittingCount').textContent = 'N/A';
                }

                // 更新打孔废料
                if (stats['打孔废料']) { // 添加检查
                    document.getElementById('scrapCount').textContent = stats['打孔废料'].count;
                } else {
                    document.getElementById('scrapCount').textContent = 'N/A';
                }

                // 直接使用JSON中的总数和总面积
                document.getElementById('totalCount').textContent = data.total_count;
                document.getElementById('totalAreaCount').textContent = parseFloat(data.total_area).toFixed(2);
            } else {
                console.error("Debug - Error in data:", data.error);  // 添加调试信息
            }
        })
        .catch(error => {
            console.error('获取分割统计数据失败:', error);
        });
}

// 添加全局变量跟踪文件数据计数
let lastSegmentationFileTime = 0;
let lastDangerFileTime = 0;
let lastDangerDataCount = 0;
let lastSegmentationDataCount = 0;

// 全局变量用于存储 ECharts 实例
let scrapPieChartInstance = null;
let scrapCountPieChartInstance = null;
let dangerPieChartInstance = null;

// 检查segmentation_result.json是否更新
function checkSegmentationFileUpdate() {
    fetch('/check_segmentation_file_new_data')
        .then(response => response.json())
        .then(data => {
            console.log("检查分割结果文件更新 - 当前数据:", data);
            console.log("上次计数:", lastSegmentationDataCount, "当前计数:", data.data_count);
            
            // 数据行数增加表示有新数据
            if (data.success && data.data_count > lastSegmentationDataCount) {
                console.log('检测到segmentation_result.json文件新增数据，将更新饼图', new Date().toLocaleTimeString());
                console.log('新数据计数:', data.data_count, '旧数据计数:', lastSegmentationDataCount);
                
                // 更新最后数据计数和修改时间
                lastSegmentationDataCount = data.data_count;
                lastSegmentationFileTime = data.last_modified;
                
                // 延迟2秒后执行废钢饼图更新，确保数据已加载
                setTimeout(() => {
                    // 更新废钢面积饼图
                    console.log('开始更新废钢面积饼图', new Date().toLocaleTimeString());
                    updateScrapPieChart();
                    // 更新废钢数量饼图
                    console.log('开始更新废钢数量饼图', new Date().toLocaleTimeString());
                    updateScrapCountPieChart();
                }, 2000);
            }
        })
        .catch(error => {
            console.error('检查文件更新失败:', error);
        });
}

// 检查detection_results.json是否更新
function checkDangerFileUpdate() {
    fetch('/check_danger_file_new_data')
        .then(response => response.json())
        .then(data => {
            console.log("检查危险物文件更新 - 当前数据:", data);
            console.log("上次计数:", lastDangerDataCount, "当前计数:", data.data_count);
            
            // 数据行数增加表示有新数据
            if (data.success && data.data_count > lastDangerDataCount) {
                console.log('检测到detection_results.json文件新增数据，将更新饼图', new Date().toLocaleTimeString());
                console.log('新数据计数:', data.data_count, '旧数据计数:', lastDangerDataCount);
                
                // 更新最后数据计数和修改时间
                lastDangerDataCount = data.data_count;
                lastDangerFileTime = data.last_modified;
                
                // 立即更新数量显示
                updateDangerCounts();
                updateDangerTotal();
                
                // 延迟2秒后执行危险物饼图更新，确保数据已加载
                console.log('开始计时2秒后更新危险物饼图', new Date().toLocaleTimeString());
                setTimeout(() => {
                    // 更新危险物饼图
                    console.log('开始更新危险物饼图', new Date().toLocaleTimeString());
                    updateDangerPieChart();
                }, 1000);
            }
        })
        .catch(error => {
            console.error('检查文件更新失败:', error);
        });
}

// 生成废钢比例饼图
function updateScrapPieChart(){
    fetch('/get_segmentation_stats')
        .then(response=>response)
        //function updateScrapPieChart() {
    //fetch('/get_segmentation_stats')
        //.then(response => response.json())
        .then(data => {
            if (data.success) {
                const stats = data.stats;
                // 准备饼图数据
                const chartData = [];
                
                // 从统计数据中提取面积信息
                for (const category in stats) {
                    // 只添加有面积的类别
                    if (stats[category].area > 0) {
                        chartData.push({
                            name: category,
                            value: parseFloat(stats[category].area).toFixed(2)
                        });
                    }
                }
                
                // 如果没有数据，添加一个占位数据
                if (chartData.length === 0) {
                    chartData.push({
                        name: '无数据',
                        value: 100
                    });
                }
                
                // 初始化饼图
                const chartDom = document.getElementById('scrapPieChart');
                if (!chartDom) { // 添加元素存在检查
                    console.error("找不到 scrapPieChart 元素");
                    return;
                }
                // 确保每次都初始化新的echarts实例或获取已有实例
                scrapPieChartInstance = echarts.getInstanceByDom(chartDom);
                if (!scrapPieChartInstance) {
                    scrapPieChartInstance = echarts.init(chartDom);
                }
                
                // 饼图配置
                const option = {
                    title: {
                        text: '废钢面积比例',
                        left: 'center',
                        top: 10,
                        textStyle: {
                            color: '#00F4FD'
                        }
                    },
                    tooltip: {
                        trigger: 'item',
                        formatter: '{a} <br/>{b}: {c} ({d}%)'
                    },
                    legend: {
                        orient: 'vertical',
                        right: 20,
                        top: 'center',
                        textStyle: {
                            color: '#fff',
                            fontSize: 10
                        },
                        itemWidth: 10,
                        itemHeight: 10
                    },
                    color: ['#3AA0FF', '#36CBCB', '4ECB73', '#FBD437', 'F2637B', '#975FE4', '#36C6A0', '#F66214'],
                    series: [
                        {
                            name: '废钢面积',
                            type: 'pie',
                            radius: ['30%', '60%'],
                            center: ['50%', '55%'],
                            avoidLabelOverlap: true,
                            itemStyle: {
                                borderRadius: 4,
                                borderColor: '#fff',
                                borderWidth: 1
                            },
                            label: {
                                show: false
                            },
                            emphasis: {
                                label: {
                                    show: true,
                                    fontSize: 10,
                                    fontWeight: 'bold',
                                    color: '#fff'
                                }
                            },
                            labelLine: {
                                show: false
                            },
                            data: chartData
                        }
                    ]
                };
                
                // 设置饼图并渲染
                scrapPieChartInstance.setOption(option);
                console.log("废钢比例饼图已更新");
            } else {
                console.error("获取废钢统计数据失败");
                document.getElementById('scrapPieChart').innerHTML = 
                    `<h3>废钢比例</h3><p>等待数据加载...</p>`;
            }
        })
        .catch(error => {
            console.error('生成废钢比例饼图失败:', error);
            document.getElementById('scrapPieChart').innerHTML = 
                `<h3>废钢比例</h3><p>等待数据加载...</p>`;
        });
}

// 生成废钢数量比例饼图
function updateScrapCountPieChart() {
    fetch('/get_segmentation_stats')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const stats = data.stats;
                // 准备饼图数据
                const chartData = [];
                
                // 从统计数据中提取数量信息
                for (const category in stats) {
                    // 只添加有数量的类别
                    if (stats[category].count > 0) {
                        chartData.push({
                            name: category,
                            value: stats[category].count
                        });
                    }
                }
                
                // 如果没有数据，添加一个占位数据
                if (chartData.length === 0) {
                    chartData.push({
                        name: '无数据',
                        value: 100
                    });
                }
                
                // 初始化饼图
                const chartDom = document.getElementById('scrapCountPieChart');
                 if (!chartDom) { // 添加元素存在检查
                    console.error("找不到 scrapCountPieChart 元素");
                    return;
                }
                 // 确保每次都初始化新的echarts实例或获取已有实例
                scrapCountPieChartInstance = echarts.getInstanceByDom(chartDom);
                if (!scrapCountPieChartInstance) {
                     scrapCountPieChartInstance = echarts.init(chartDom);
                }
                
                // 饼图配置
                const option = {
                    title: {
                        text: '废钢数量比例',
                        left: 'center',
                        top: 10,
                        textStyle: {
                            color: '#00F4FD'
                        }
                    },
                    tooltip: {
                        trigger: 'item',
                        formatter: '{a} <br/>{b}: {c} ({d}%)'
                    },
                    legend: {
                        orient: 'vertical',
                        right: 20,
                        top: 'center',
                        textStyle: {
                            color: '#fff'
                        },
                        itemWidth: 10,
                        itemHeight: 10
                    },
                    color: ['#F66214', '#36C6A0', '#975FE4', '#F2637B', '#FBD437', '4ECB73', '#36CBCB', '#3AA0FF'],
                    series: [
                        {
                            name: '废钢数量',
                            type: 'pie',
                            radius: ['30%', '60%'],
                            center: ['50%', '55%'],
                            avoidLabelOverlap: true,
                            itemStyle: {
                                borderRadius: 4,
                                borderColor: '#fff',
                                borderWidth: 1
                            },
                            label: {
                                show: false
                            },
                            emphasis: {
                                label: {
                                    show: true,
                                    fontSize: 10,
                                    fontWeight: 'bold',
                                    color: '#fff'
                                }
                            },
                            labelLine: {
                                show: false
                            },
                            data: chartData
                        }
                    ]
                };
                
                // 设置饼图并渲染
                scrapCountPieChartInstance.setOption(option);
                console.log("废钢数量比例饼图已更新");
            } else {
                console.error("获取废钢统计数据失败");
                document.getElementById('scrapCountPieChart').innerHTML = 
                    `<h3>废钢数量比例</h3><p>等待数据加载...</p>`;
            }
        })
        .catch(error => {
            console.error('生成废钢数量比例饼图失败:', error);
            document.getElementById('scrapCountPieChart').innerHTML = 
                `<h3>废钢数量比例</h3><p>等待数据加载...</p>`;
        });
}

// 生成危险物比例饼图
function updateDangerPieChart() {
    fetch('/get_latest_danger_counts')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const counts = data.counts;
                // 准备饼图数据
                const chartData = [];
                
                // 从计数数据中提取各类危险物
                const dangerTypes = {
                    'GasCyl1': '气瓶1型',
                    'GasCyl2': '气瓶2型',
                    'FireExt1': '灭火器1型',
                    'FireExt2': '灭火器2型'
                };
                
                // 为每种危险物添加数据点
                for (const [key, label] of Object.entries(dangerTypes)) {
                    if (counts[key] > 0) {
                        chartData.push({
                            name: label,
                            value: counts[key]
                        });
                    }
                }
                
                // 如果没有数据，添加一个占位数据
                if (chartData.length === 0) {
                    chartData.push({
                        name: '无危险物',
                        value: 100
                    });
                }
                
                // 初始化饼图
                const chartDom = document.getElementById('dangerPieChart');
                // 确保DOM元素存在
                if (!chartDom) {
                    console.error("找不到dangerPieChart元素");
                    return;
                }

                // 确保每次都初始化新的echarts实例或获取已有实例
                dangerPieChartInstance = echarts.getInstanceByDom(chartDom);
                if (!dangerPieChartInstance) {
                    dangerPieChartInstance = echarts.init(chartDom);
                }
                
                // 饼图配置
                const option = {
                    title: {
                        text: '危险物比例',
                        left: 'center',
                        top: 10,
                        textStyle: {
                            color: '#00F4FD'
                        }
                    },
                    tooltip: {
                        trigger: 'item',
                        formatter: '{a} <br/>{b}: {c} ({d}%)'
                    },
                    legend: {
                        orient: 'vertical',
                        right: 20, // 图例在右侧
                        top: 'center',
                        textStyle: {
                            color: '#fff',
                            fontSize: 10 // 设置图例文字大小为 10 像素
                        },
                        itemWidth: 10, // 缩小图例标记宽度
                        itemHeight: 10 // 缩小图例标记高度
                    },
                    color: ['#FF5252', '#FF9800', '#4CAF50', '#2196F3'],
                    series: [
                        {
                            name: '危险物数量',
                            type: 'pie',
                            radius: ['30%', '60%'],
                            center: ['45%', '55%'], // 将饼图中心移到容器水平居中 45% 的位置，为图例留出空间
                            avoidLabelOverlap: true,
                            itemStyle: {
                                borderRadius: 4,
                                borderColor: '#fff',
                                borderWidth: 1
                            },
                            label: {
                                show: false
                            },
                            emphasis: {
                                label: {
                                    show: true,
                                    fontSize: 10,
                                    fontWeight: 'bold',
                                    color: '#fff'
                                }
                            },
                            labelLine: {
                                show: false
                            },
                            data: chartData
                        }
                    ]
                };
                
                // 清空旧图表
                dangerPieChartInstance.clear();
                // 设置饼图并渲染
                dangerPieChartInstance.setOption(option);
                console.log("危险物比例饼图已更新", new Date().toLocaleTimeString());
            } else {
                console.error("获取危险物数据失败");
                document.getElementById('dangerPieChart').innerHTML = 
                    `<h3>危险物比例</h3><p>等待数据加载...</p>`;
            }
        })
        .catch(error => {
            console.error('生成危险物比例饼图失败:', error);
            document.getElementById('dangerPieChart').innerHTML = 
                `<h3>危险物比例</h3><p>等待数据加载...</p>`;
        });
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
  const initialLoads = [
    checkForNewMonitorImage(),
    checkForNewDangerImage(),
    checkForNewSegmentationImage(),
    updateSegmentationStats(),  // 添加分割统计数据更新
    updateScrapPieChart(),
    updateScrapCountPieChart(),
    updateDangerCounts(),      // 初始化显示危险物数量
    updateDangerTotal(),       // 添加危险物总数更新
    updateDangerPieChart(),    // 添加危险物比例饼图更新
    directLoadJsonFile(),
    updateArmStatus() // 添加机械臂状态更新
  ];

  Promise.all(initialLoads)
    .then(() => {
      console.log("所有初始数据加载完成，显示容器");
      // 移除隐藏类，显示所有数据容器
      document.getElementById('monitorImage').classList.remove('hidden-until-loaded');
      document.getElementById('dangerImage').classList.remove('hidden-until-loaded');
      document.getElementById('segmentationImage').classList.remove('hidden-until-loaded');
      document.getElementById('scrapPieChart').classList.remove('hidden-until-loaded');
      document.getElementById('scrapCountPieChart').classList.remove('hidden-until-loaded');
      document.getElementById('dangerPieChart').classList.remove('hidden-until-loaded');
      document.getElementById('totalStats').classList.remove('hidden-until-loaded');
      document.getElementById('totalArea').classList.remove('hidden-until-loaded');
      document.getElementById('totalDanger').classList.remove('hidden-until-loaded');
      document.getElementById('segmentationStats').classList.remove('hidden-until-loaded');
      document.getElementById('jsonDataContainer').classList.remove('hidden-until-loaded'); // jsonDataContainer 而不是 jsonData
      document.getElementById('jsonHistoryContainer').classList.remove('hidden-until-loaded'); // jsonHistoryContainer 而不是 jsonHistory
      document.getElementById('dangerCountsContainer').classList.remove('hidden-until-loaded');
      document.getElementById('armStatusContainer').classList.remove('hidden-until-loaded');

      // 获取文件初始修改时间和数据计数 (这些不需要等待所有加载完成)
      fetch('/check_segmentation_file_new_data')
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            lastSegmentationFileTime = data.last_modified;
            lastSegmentationDataCount = data.data_count;
            console.log("废钢文件初始时间戳:", lastSegmentationFileTime);
            console.log("废钢文件初始数据计数:", lastSegmentationDataCount);
          }
        });
        
      fetch('/check_danger_file_new_data')
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            lastDangerFileTime = data.last_modified;
            lastDangerDataCount = data.data_count;
            console.log("危险物文件初始时间戳:", lastDangerFileTime);
            console.log("危险物文件初始数据计数:", lastDangerDataCount);
          }
        })
        .catch(error => {
          console.error('获取危险物文件初始数据失败:', error);
        });

      // 每秒更新一次所有图片和数据
      setInterval(checkForNewMonitorImage, 1000);
      setInterval(checkForNewDangerImage, 1000);
      setInterval(checkForNewSegmentationImage, 1000);
      setInterval(updateSegmentationStats, 1000);  // 每秒更新一次分割统计数据
      setInterval(directLoadJsonFile, 1000);
      setInterval(updateArmStatus, 2000);
      
      // 每2秒检查一次文件是否更新 (这些会触发饼图和数量更新)
      setInterval(checkSegmentationFileUpdate, 2000);
      setInterval(checkDangerFileUpdate, 2000);
    })
    .catch(error => {
      console.error("初始数据加载失败:", error);
      // 可以选择在这里显示一个错误消息
    });

  // 添加窗口resize事件监听器
  window.addEventListener('resize', function() {
    if (scrapPieChartInstance) {
      scrapPieChartInstance.resize();
    }
    if (scrapCountPieChartInstance) {
      scrapCountPieChartInstance.resize();
    }
    if (dangerPieChartInstance) {
      dangerPieChartInstance.resize();
    }
  });
};