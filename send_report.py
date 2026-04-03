#!/usr/bin/env python3
"""Send Feishu progress report via Feishu Open API"""
import json, urllib.request, urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """SuperModel项目进度更新 (v1.50.0 - 2026-04-03 13:16)：
✅ 本次完成：
  - 触觉传感器模块 (tactile.py) 完整实现
    * TactileArray: 电子皮肤触觉阵列，支持压力/温度/接近觉/滑移检测
    * PressureProcessor: 压力信号处理
    * VirtualTactileSensor: 仿真环境虚拟触觉传感器
    * AGV五级规格: S(8×8)/M(16×16)/L(24×24)/XL(32×32)/XXL(48×48)
  - 力觉传感器模块 (force.py) 完整实现
    * ForceTorqueSensor: 六维力矩传感器 (ATI风格)
    * Wrench: 力旋量数据结构
    * WrenchProcessor: 力信号处理
    * VirtualForceSensor: 仿真环境虚拟力传感器
    * AGV五级规格: S(3轴±100N)/M(6轴±200N)/L(±500N)/XL(±1000N)/XXL(±5000N)
  - IMU传感器模块 (imu.py) 完整实现
    * IMUSensor: BMI088/MPU6050/MPU9250/ADIS16470 接口
    * Pose/PoseEstimator: 姿态估计 (Madgwick/互补滤波/Kalman)
    * VirtualIMUSensor: 仿真环境虚拟IMU
    * AGV五级规格: S(MPU6050 100Hz)/M(BMI088 200Hz)/L(500Hz)/XL(1kHz)/XXL(2kHz)
  - 控制模块完善: 21个子模块全部就绪
  - 仿真环境完善: 7个仿真模块 (MuJoCo/PyBullet/Gazebo/Gymnasium等)
  - 测试用例完善: sensor_tests.py(130项)/fusion_tests.py(104项)
  - 文档同步: CHANGELOG/PROGRESS/README/MODULE_INDEX/SPEC/AGV_SPEC_QUICKREF → v1.50.0
  - GitHub提交: d93426a 'docs: 更新版本至v1.50.0 触觉/力觉/IMU模块完成 文档同步 1277项测试通过'

📊 模块状态总览 (v1.50.0 - 1277项测试通过):
  传感器(5类): vision/audio/tactile/force/imu ✅ + encoders/manager
  控制(21子模块): motor/motion/trajectory/mpc/impedance/force/imu/tactile/agv/安全监控/避障/规划/ROS2/多AGV/teleop/supervisor/autotune
  融合: 跨模态Transformer / 互补滤波 / EKF / 多传感器融合
  学习: Dreamer / 世界模型 / 自监督 / 自主学习框架
  仿真: MuJoCo / PyBullet / Gazebo / Gymnasium / 仓储物流场景
  文档: SPEC.md / MODULE_INDEX.md / DESIGN.md / AGV_SPEC_QUICKREF.md / 完整接口文档
  AGV五级规格: 触觉/力觉/IMU/感知/融合/认知/控制(Supervisor五级)/学习/通信/安全 + 快速选型指南

🔜 下一步: 真实机器人集成测试、端到端具身智能演示、五级规格优化"""


def get_token():
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    payload = json.dumps({"app_id": APP_ID, "app_secret": APP_SECRET}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode()).get("tenant_access_token", "")


def send_message(token: str, chat_id: str, text: str) -> dict:
    url = "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id"
    payload = json.dumps({
        "receive_id": chat_id,
        "msg_type": "text",
        "content": json.dumps({"text": text})
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    })
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


if __name__ == "__main__":
    try:
        token = get_token()
        result = send_message(token, CHAT_ID, MESSAGE)
        if result.get("code") == 0:
            print("✅ Feishu message sent successfully")
        else:
            print(f"❌ Feishu API error: {result}")
    except urllib.error.URLError as e:
        print(f"❌ Network error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
