#!/usr/bin/env python3
"""Send Feishu progress report via Feishu Open API"""
import json, urllib.request, urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """SuperModel项目进度更新 (v2.56.0 - 2026-04-10 16:30)：

✅ 本次完成（学习进度）：
  - 触觉/力觉/IMU传感器模块已完整实现：
    * tactile.py: 电子皮肤触觉阵列 (TactileArray, PressureProcessor, VirtualTactileSensor)
    * force.py: 六维力矩传感器 (ForceTorqueSensor, WrenchProcessor, VirtualForceSensor)
    * imu.py: 惯性测量单元 (IMUSensor, PoseEstimator, VirtualIMUSensor)
  - 控制模块(control/)已完善：
    * 32个控制器模块，覆盖运动/轨迹/阻抗/MPC/安全/遥操作等
    * 触觉/力觉/IMU专用控制器
    * 蜂群控制swarm_control.py
  - AGV五级规格表已完整：
    * 触觉: S/M/L/XL/XXL (阵列8×8→48×48, 分辨率12→16bit)
    * 力觉: S/M/L/XL/XXL (3轴→6轴, 力范围100→5000N)
    * IMU: S/M/L/XL/XXL (MPU6050→ADIS16470, 噪声密度400→10μg/√Hz)
  - 测试用例已完成：
    * sensor_tests.py: 341项测试全通过
    * fusion_tests.py: 73项测试全通过
    * 全套测试: 2671项测试全通过, 38项跳过

📊 项目统计：
  - 代码文件: ~50个Python模块
  - 测试文件: ~60个测试模块
  - 设计文档: ~15份SPEC/接口文档
  - 累计提交: 256次

🔄 当前阶段：v1.71 全面测试+接口文档 (进行中)
📅 下一步：v2.0 真实AGV部署+视觉-语言-动作

仓库: https://github.com/DIT4FUN/SuperModel"""

def send():
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    data = json.dumps({"app_id": APP_ID, "app_secret": APP_SECRET}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        token = json.loads(r.read())["tenant_access_token"]

    msg_url = "https://open.feishu.cn/open-apis/im/v1/messages"
    body = json.dumps({
        "receive_id": CHAT_ID,
        "msg_type": "text",
        "content": json.dumps({"text": MESSAGE})
    }).encode()
    req = urllib.request.Request(
        msg_url + f"?receive_id_type=chat_id",
        data=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        print("Sent:", r.read().decode())

if __name__ == "__main__":
    send()
