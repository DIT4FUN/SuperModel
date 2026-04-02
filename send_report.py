#!/usr/bin/env python3
"""Send Feishu progress report via Feishu Open API"""
import json, urllib.request, urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """SuperModel项目进度更新 (2026-04-02 上午)：
✅ 本次完成：
  - 传感器模块完整实现 (tactile.py / force.py / imu.py)
  - 控制模块全部就位 (agv.py / mpc.py / planner.py / safety_controller.py 等 16 个文件)
  - docs/SPEC.md 更新：详细模块接口设计 + AGV五级规格表
  - tests/sensor_tests.py (触觉/力觉/IMU/编码器 传感器测试)
  - tests/fusion_tests.py (互补滤波/EKF/多传感器融合 测试)
✅ 测试结果：89 tests passed in 0.54s
✅ GitHub已推送：01cb6e9

📊 AGV五级规格表：
  L1: ≤500kg, ±10mm (磁条/二维码)
  L2: 500-1500kg, ±5mm (激光导航)
  L3: 1500-3000kg, ±3mm (SLAM视觉)
  L4: 3000-5000kg, ±1mm (多传感器融合)
  L5: >5000kg, <±0.5mm (超模态具身智能)

🔜 下一步：完善仿真环境 (Gazebo/MuJoCo)、端到端集成测试"""


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
