#!/usr/bin/env python3
"""Send Feishu progress report via Feishu Open API"""
import json, urllib.request, urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """SuperModel项目进度更新 (v1.51.0 - 2026-04-05 11:32)：
✅ 本次完成：
  - 文档完善: MODULE_INTERFACE.md 补充附录A (AGV五级控制子系统完整规格表)
    * 控制子系统五级规格总表 (S/M/L/XL/XXL)
    * 感知-控制闭环延迟规格 (5ms~110ms总延迟)
    * 触觉/力觉/IMU五级配置对照
  - 全量测试验证: sensor_tests.py + fusion_tests.py 共130项全部通过
  - 项目清理: 删除临时未追踪文件，Git工作区clean

📊 模块状态总览 (v1.51.0)：
  传感器(6类): vision/audio/tactile/force/imu ✅ + encoders/manager
  控制(22子模块): motor/motion/trajectory/mpc/impedance/force/imu/tactile/agv/安全监控/避障/规划/ROS2/多AGV/teleop/supervisor/autotune + sensorimotor
  融合: 跨模态Transformer / 互补滤波 / EKF / 多传感器融合
  学习: Dreamer / 世界模型 / 自监督 / 自主学习框架
  仿真: MuJoCo / PyBullet / Gazebo / Gymnasium / 仓储物流场景
  测试: 30+测试文件，130项核心测试通过
  文档: MODULE_INTERFACE.md(165K+) / AGV五级规格完整对照表(948行)
  GitHub提交: 1cc988d 'docs: v1.51.0 补充AGV五级控制子系统规格表'

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
