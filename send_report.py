#!/usr/bin/env python3
"""Send Feishu progress report via Feishu Open API"""
import json, urllib.request, urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """SuperModel项目进度更新 (2026-04-02 晚间)：
✅ 本次完成：
  - 触觉传感器模块 tactile.py (TactileArray/VirtualTactileSensor/AGV_TACTILE_GRADES)
  - 力觉传感器模块 force.py (ForceTorqueSensor/Wrench/VirtualForceSensor/AGV_FORCE_GRADES)
  - IMU传感器模块 imu.py (IMUSensor/Pose/PoseEstimator/VirtualIMUSensor/AGV_IMU_GRADES)
  - 控制模块完善 (AGVMotionController/TrajectoryTracker/SkidSteer/Ackermann)
  - SafetyController五级安全监控 (S/M/L/XL/XXL)
  - 测试用例：sensor_tests.py (197项) / fusion_tests.py (104项) 全部通过
  - SPEC.md: AGV五级完整规格总表(七大子系统)、模块接口详细设计
  - MODULE_INDEX.md: 完整模块索引 (v1.31.0)
  - GitHub已推送：71056f1

📊 当前模块状态 (v1.44.0)：
  传感器：tactile.py ✅ / force.py ✅ / imu.py ✅ / vision.py ✅ / audio.py ✅ / encoders.py ✅
  控制(18子模块)：motion/trajectory/mpc/impedance/force/imu/tactile控制/agv/安全监控/避障/规划/ROS2/多AGV ✅
  融合：跨模态Transformer / 互补滤波 / EKF / 多传感器融合 ✅
  仿真：MuJoCo / Gazebo / Gymnasium / 仓储物流场景 ✅
  测试：1340项通过 (sensor_tests 197 / fusion_tests 104)

🔜 下一步：超模态大模型推理接口、AGV实物对接示例"""


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
