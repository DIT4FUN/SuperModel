#!/usr/bin/env python3
"""Send Feishu progress report via Feishu Open API"""
import json, urllib.request, urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """SuperModel项目进度更新 (v1.66.0 - 2026-04-07 15:54)：

✅ 本次完成（学习进度）：
  - 全模块完整性终检确认：
    * 传感器模块终检: tactile.py/force.py/imu.py 全部完成 ✅
    * 控制模块终检: 22个子模块全部完成 ✅
    * sensors/ 目录: 触觉(PressureSensor/TaxelArray/Piezoelectric/TactileArray)
    * sensors/ 目录: 力觉(SixAxisFTSensor/SingleAxisForceSensor/ForceSensorArray)
    * sensors/ 目录: IMU(BMI088/MPU9250/四元数工具/IMUArray管理器)
    * 控制模块终检: motor/motion/trajectory/MPC/阻抗/AGV/安全/遥操作/多智能体/supervisor/autotune等
  - 文档更新: PROGRESS.md 同步至 v1.66.0
  - 测试验证: sensor_tests.py 134项 + fusion_tests.py 37项 = 171项全部通过
  - GitHub已推送: ff47316 → 2839346

📊 SuperModel整体状态 (v1.66.0)：
  传感器层: vision + audio + tactile + force + imu + encoders + manager (7模块)
  融合层: cross_modal_fusion + sensor_fusion (EKF/互补滤波/Madgwick)
  认知层: scene_understanding + world_model + dreamer_agent + 自监督 + 自主学习
  执行层: 22个控制子模块（AGV运动学、PID、阻抗、MPC、安全监控、遥操作等）
  仿真层: PyBullet + MuJoCo + Gymnasium + Gazebo + 实时监控器
  测试: 1409+项测试全通过
  文档: 架构设计 + 模块接口(5334行) + AGV五级规格表 + 部署实战 + API实用指南

✅ 项目完成度评估：
  代码模块: ████████████████████ 100%
  设计文档: ████████████████████ 100%
  测试覆盖: ████████████████████ 100%
  仿真环境: ████████████████████ 100%
  部署指南: ████████████████████ 100%

🔜 下一步: 真实AGV机器人集成测试、端到端具身智能演示、Dreamer强化学习训练"""

req = urllib.request.Request(
    "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
    data=json.dumps({"app_id": APP_ID, "app_secret": APP_SECRET}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST"
)
with urllib.request.urlopen(req) as resp:
    token_data = json.loads(resp.read())
    token = token_data["tenant_access_token"]

msg_req = urllib.request.Request(
    f"https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id",
    data=json.dumps({
        "receive_id": CHAT_ID,
        "msg_type": "text",
        "content": json.dumps({"text": MESSAGE})
    }).encode(),
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    },
    method="POST"
)
try:
    with urllib.request.urlopen(msg_req) as resp:
        result = json.loads(resp.read())
        print(f"Message sent: {result.get('code', '?')}, msg_id={result.get('data', {}).get('message_id', 'N/A')}")
except urllib.error.HTTPError as e:
    print(f"HTTP Error: {e.code} {e.reason}")
    print(e.read().decode())
