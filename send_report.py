#!/usr/bin/env python3
"""Send Feishu progress report via Feishu Open API"""
import json, urllib.request, urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """SuperModel项目进度更新 (v1.44.0 - 2026-04-02 晚间)：
✅ 本次完成：
  - 新增18项传感器边界测试 (触觉/力觉/IMU空接触、多点接触、标定、饱和、NaN/Inf)
  - 新增23项融合鲁棒性测试 (加速度/陀螺仪饱和、自由落体、多传感器集成)
  - 修复2项融合集成测试 (ForceSensorType.VIRTUAL→SIX_AXIS)
  - 测试总数: 1094 → 1135 (新增41项)
  - 版本号: v1.43.0 → v1.44.0
  - CHANGELOG/MODULE_INDEX/SPEC 文档同步更新
  - GitHub已推送: f217edc → 9612c44

📊 当前模块状态 (v1.44.0 - 1135项测试通过)：
  传感器(5类)：vision ✅ / audio ✅ / tactile ✅ / force ✅ / imu ✅ + encoders/manager
  控制(18子模块)：motion/trajectory/mpc/impedance/force/imu/tactile控制/agv/安全监控/避障/规划/ROS2/多AGV/teleop/supervisor
  融合：跨模态Transformer / 互补滤波 / EKF / 多传感器融合
  学习：Dreamer / 世界模型 / 自监督 / 自主学习框架
  仿真：MuJoCo / Gazebo / Gymnasium / 仓储物流场景
  AGV五级规格表：触觉(S→XXL) / 力觉(S→XXL) / IMU(S→XXL) / 综合规格 / 通信接口

🔜 下一步：超模态大模型推理接口优化、AGV实物对接示例、持续学习框架完善"""


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
