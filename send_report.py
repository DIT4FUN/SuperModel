#!/usr/bin/env python3
"""Send Feishu progress report via Feishu Open API"""
import json, urllib.request, urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """SuperModel项目进度更新 (2026-04-02 中午)：
✅ 本次完成：
  - docs/SPEC.md 接口文档完善 + 通信接口规格表
  - docs/DESIGN.md 架构设计文档更新
  - 测试用例通过：1094 passed / 22 skipped / 28 warnings (30.98s)
  - GitHub已推送：12a3c3f

📊 AGV五级传感器规格（完整）：
  触觉：S(8×8@50Hz/500kPa) → M(16×16@100Hz/1MPa) → L(24×24@200Hz/2MPa) → XL(32×32@500Hz/5MPa) → XXL(48×48@1kHz/10MPa)
  力觉：S(3轴±100N@100Hz) → M(6轴±200N@500Hz) → L(6轴±500N@1kHz) → XL(6轴±1000N@2kHz) → XXL(6轴±5000N@5kHz)
  IMU：S(MPU6050@100Hz) → M(BMI088@200Hz) → L(BMI088@500Hz) → XL(ADIS16470@1kHz) → XXL(ADIS16470@2kHz)

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
