#!/usr/bin/env python3
"""Send Feishu progress report via Feishu Open API"""
import json, urllib.request, urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """SuperModel项目进度更新 (2026-04-02 上午第二轮)：
✅ 本次完成：
  - docs/SPEC.md 完善：详细模块接口设计 + AGV五级规格表
  - 新增通信接口规格表 (I2C/SPI/USB/CAN/Ethernet/WiFi/5G)
  - 新增传感器管理器接口 (SensorManager)
  - 新增控制模块核心接口 (AGVMotionController/SafetyMonitor/TrajectoryPlanner)
  - 新增融合网络接口 (CrossModalFusion/ComplementaryFilter/EKF)
  - 更新测试规范，统计测试用例数量
✅ 测试结果：65 sensor_tests + 24 fusion_tests = 89 tests passed
✅ GitHub已推送：9f1922f

📊 AGV五级传感器规格：
  触觉：S(8×8@50Hz) → M(16×16@100Hz) → L(24×24@200Hz) → XL(32×32@500Hz) → XXL(48×48@1kHz)
  力觉：S(3轴@100Hz) → M(6轴@500Hz) → L(6轴@1kHz) → XL(6轴@2kHz) → XXL(6轴@5kHz)
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
