#!/usr/bin/env python3
"""Send Feishu progress report via Feishu Open API"""
import json, urllib.request, urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """SuperModel项目进度更新 (v1.51.0 - 2026-04-05 10:15)：
✅ 本次完成：
  - 传感器-执行器融合控制模块 (sensorimotor.py) 新增
    * SensorimotorIntegration: 多模态感知-运动融合控制类
    * SensorimotorConfig: AGV五级规格配置
    * SensorimotorSimulator: 仿真环境验证
    * AGV五级规格: S/M/L/XL/XXL
  - 控制模块完善: sensorimotor模块集成到control/__init__.py
  - 测试用例: sensorimotor_tests.py 更新

📊 模块状态总览 (v1.51.0)：
  传感器(5类): vision/audio/tactile/force/imu ✅ + encoders/manager
  控制(22子模块): motor/motion/trajectory/mpc/impedance/force/imu/tactile/agv/安全监控/避障/规划/ROS2/多AGV/teleop/supervisor/autotune + sensorimotor
  融合: 跨模态Transformer / 互补滤波 / EKF / 多传感器融合
  学习: Dreamer / 世界模型 / 自监督 / 自主学习框架
  仿真: MuJoCo / PyBullet / Gazebo / Gymnasium / 仓储物流场景
  GitHub提交: bcb918d 'feat: 添加传感器-执行器融合控制模块(sensorimotor.py)'

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
