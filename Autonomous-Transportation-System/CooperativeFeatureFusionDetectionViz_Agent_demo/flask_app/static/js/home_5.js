// const longitudeDisplay = document.getElementById('longitude');
// const latitudeDisplay = document.getElementById('latitude');
// const headingDisplay = document.getElementById('heading');
// const speedDisplay = document.getElementById('speed');

// const IdDisplay = document.getElementById('id');
const communicationRateDisplay = document.getElementById('communication-rate');
const logContainer = document.getElementById('logContainer')

// const commandInput = document.getElementById('commandInput');


// commandInput.addEventListener('keypress', function(event) {
//     // 检查按下的键是否是回车键 (keyCode 13 或 key 'Enter')
//     if (event.key === 'Enter') {
//         // 阻止回车键的默认行为（例如提交表单，如果输入框在一个form中）
//         event.preventDefault();

//         // 调用要触发的函数
//         sendCommand();
//     }
// });

// async function sendCommand(){
//     const inputValue = commandInput.value.trim();
//     if (inputValue === "") {
//         return;
//     }

// //    console.info(inputValue)

//     // 使用 fetch API 发送 POST 请求
//     fetch('/send_command', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json', // 告诉服务器我们发送的是 JSON
//         },
//         body: JSON.stringify({ command: inputValue }), // 将数据转换为 JSON 字符串
//     })
//     .then(response => response.json()) // 解析 JSON 响应
//     .then(data => {
//         console.log('后端响应:', data);
//         if (data.status === 'success') {
//             console.success(`后端成功处理: ${data.message}`);
//         } else {
//             console.error(`后端错误: ${data.message}`);
//         }
//     })
//     .catch(error => {
//         console.error('发送请求失败:', error);
//     });

//     commandInput.value = ''; // 清空输入框
// }


// 定义一个函数来获取并更新车辆状态
// function updateState() {
//     fetch('/get_state') // 向后端API发起请求
//         .then(response => response.json()) // 将响应解析为JSON
//         .then(data => {
//             longitudeDisplay.textContent = data.longitude;
//             latitudeDisplay.textContent = data.latitude;
//             headingDisplay.textContent = data.heading;
//             speedDisplay.textContent = data.speed;
//             communicationRateDisplay.textContent = data.communication_rate
//         })
//         .catch(error => {
//             console.error('获取车辆状态失败:', error);
//             longitudeDisplay.textContent = 'N/A';
//             latitudeDisplay.textContent = 'N/A';
//             headingDisplay.textContent = 'N/A';
//             speedDisplay.textContent = 'N/A';
//             communicationRateDisplay.textContent = 'N/A'
//         });
// }

// function updateId() {
//     fetch('/get_id') // 向后端API发起请求
//         .then(response => response.json()) // 将响应解析为JSON
//         .then(data => {
//             IdDisplay.textContent = data.id;
//         })
//         .catch(error => {
//             console.error('获取车辆ID失败:', error);
//             IdDisplay.textContent = 'N/A';
//         });
// }

async function fetchNewLog() {
    fetch('/get_log') // 向后端API发起请求
        .then(response => response.json()) // 将响应解析为JSON
        .then(data => {
            if (data.log){
                console.log('Received log:', data.log);
                var logEntry = document.createElement('div');
                logEntry.className = 'log-entry';

                // 尝试根据日志级别添加样式
                if (data.log.includes(" - INFO - ")) {
                    logEntry.classList.add('INFO');
                } else if (data.log.includes(" - WARNING - ")) {
                    logEntry.classList.add('WARNING');
                } else if (data.log.includes(" - ERROR - ")) {
                    logEntry.classList.add('ERROR');
                } else if (data.log.includes(" - CRITICAL - ")) {
                    logEntry.classList.add('CRITICAL');
                } else if (data.log.includes(" - DEBUG - ")) {
                    logEntry.classList.add('DEBUG');
                }

                logEntry.textContent = data.log;
                logContainer.appendChild(logEntry);
                
                // 如果当前在底部，则自动滚动到底部
                if (logContainer.scrollTop + logContainer.clientHeight >= logContainer.scrollHeight) {
                    logContainer.scrollTop = logContainer.scrollHeight;
                }
            }
            fetchNewLog()
        })
        .catch(error => {
            console.error('获取日志失败:', error);
            fetchNewLog()
        });
}

// 用于存储每个图片元素对应的定时器 ID
const imgTimeouts = {};

// 封装一个通用的更新图片并重置倒计时的函数
function updateImageWithTimeout(elementId, base64Data) {
    const imgElement = document.getElementById(elementId);
    if (!imgElement) return;

    // 1. 设置图片内容
    imgElement.src = 'data:image/jpeg;base64,' + base64Data;
    // (可选) 确保图片可见，防止之前被隐藏
    imgElement.style.display = 'block';

    // 2. 如果该元素已有正在运行的定时器，先清除它（“重置闹钟”）
    if (imgTimeouts[elementId]) {
        clearTimeout(imgTimeouts[elementId]);
    }

    // 3. 开启一个新的 10 秒定时器
    imgTimeouts[elementId] = setTimeout(function() {
        // 10秒后执行：清除图片
        console.log(`Image ${elementId} expired (no update for 10s). Clearing...`);

        // 方法 B: 移除 src 属性 (推荐)
        imgElement.removeAttribute('src');
        imgElement.style.display = 'none';

         // 方法 C: 直接隐藏元素 (视觉效果最好，不会留白框)


    }, 10000); // 10000 毫秒 = 10 秒
}

var socket = io();

// 监听 'update_frames' 事件
socket.on('update_frames', function(data) {
    // data 就是后端发送的 data_payload 字典

    if (data.pcd_img) {
        updateImageWithTimeout('pcd-img', data.pcd_img);
    }

    // if (data.request_map_img) {
    //     updateImageWithTimeout('request-map-img', data.request_map_img);
    // }

    // if (data.others_comm_mask_img) {
    //     updateImageWithTimeout('others-comm-mask-img', data.others_comm_mask_img);
    // }

    if (data.ego_feature_img) {
        updateImageWithTimeout('ego-feature-img', data.ego_feature_img);
    }

    if (data.fused_feature_img) {
        updateImageWithTimeout('fused-feature-img', data.fused_feature_img);
    }

    // if (data.pred_img_0) {
    //     updateImageWithTimeout('pred-img-0', data.pred_img_0);
    // }

    // if (data.pred_img_3) {
    //     updateImageWithTimeout('pred-img-3', data.pred_img_3);
    // }
});

socket.on('connect', function() {
    console.log("Connected to server!");
});

// 每秒更新一次
// setInterval(updateState, 1000); // 1000 毫秒 = 1 秒
// setInterval(updateId, 10 * 1000); // 10000 毫秒 = 10 秒
//setInterval(updateCommunicationRate, 1000); // 1000 毫秒 = 1 秒

// 第一次加载时，希望立即更新（覆盖初始值），调用一次
// updateState();
// updateId();
// fetchNewLog();
//updateCommunicationRate();