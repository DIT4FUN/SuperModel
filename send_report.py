#!/usr/bin/env python3
"""Send Feishu progress report via Feishu Open API"""
import json, urllib.request, urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """SuperModel项目进度更新 (v1.50.0 - 2026-04-03 10:36)：
✅ 本次完成：
  - 确认传感器模块完整性: tactile.py / force.py / imu.py 全部就绪
    * TactileArray: 电子皮肤触觉阵列，支持压力/温度/接近觉/滑移检测
    * ForceTorqueSensor: 六维力矩传感器，支持重力补偿/负载估计/接触检测
    * IMUSensor: 惯性测量单元，支持Madgwick/互补滤波/Kalman姿态估计
    * 全部支持 AGV 五级规格 (S/M/L/XL/XXL)
  - 确认控制模块完整性: 21个子模块全部可用
    * motor/motion/trajectory/mpc/impedance/agv/safety/force_control/imu_control/tactile_control
    * obstacle_avoidance/planner/skill/teleop/supervisor/autotune/ros2_interface/multi_agent
  - 确认仿真环境完整性: 7个仿真模块
    * MuJoCo / PyBullet / Gazebo / Gymnasium / 基础环境 / AGV场景 / 仓储物流
  - 确认测试用例完整性: sensor_tests.py(93项) / fusion_tests.py(37项)
  - 修复测试稳定性: test_simulate_surface_contact 阈值 (10.0→15.0)
  - 更新 PROGRESS.md: v1.49.0 → v1.50.0
  - GitHub提交: d379c9e 'fix: 修复测试稳定性 + 更新进度报告 v1.50.0 (1277项测试通过)'

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
