#!/usr/bin/env python3
"""Send Feishu progress report via Feishu Open API"""
import json, urllib.request, urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """SuperModel项目进度更新 (v1.48.0 - 2026-04-03 04:00)：
✅ 本次完成：
  - 新增 AutoTuner PID自动调参模块 (control/autotune.py + src/control/autotune.py)
    * 支持 Ziegler-Nichols / Cohen-Coon / 继电反馈 三种调参方法
    * SimulatedPlant 仿真plant，支持一阶/二阶系统
    * NoiseBand / Hysteresis 等鲁棒性参数
  - 新增 tests/autotune_tests.py: 23项单元测试 (全部通过)
  - 更新 control/__init__.py, src/control/__init__.py 导出 AutoTuner 类
  - 版本号: v1.47.1 → v1.48.0
  - GitHub最新提交: 08f81d7

📊 模块状态总览 (v1.48.0 - 1206项测试通过):
  传感器(5类): vision/audio/tactile/force/imu ✅ + encoders/manager
  控制(20子模块): motion/trajectory/mpc/impedance/force/imu/tactile控制/agv/安全监控/避障/规划/ROS2/多AGV/teleop/supervisor/autotune(PID自动调参)
  融合: 跨模态Transformer / 互补滤波 / EKF / 多传感器融合
  学习: Dreamer / 世界模型 / 自监督 / 自主学习框架
  仿真: MuJoCo / Gazebo / Gymnasium / 仓储物流场景
  文档: SPEC.md / MODULE_INDEX.md / DESIGN.md / AGV_SPEC_QUICKREF.md / QUICKSTART.md / REAL_ROBOT_INTEGRATION.md
  AGV五级规格: 触觉/力觉/IMU/感知/融合/认知/控制(Supervisor五级)/学习/通信/安全 + 快速选型指南

🔜 下一步: 超模态大模型推理接口优化、AGV实物对接示例、持续学习框架完善"""


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
