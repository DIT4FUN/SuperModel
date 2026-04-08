#!/usr/bin/env python3
"""Send Feishu progress report v1.85.0"""
import json, urllib.request, urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """SuperModel项目进度更新 (v1.85.0 - 2026-04-09 01:41)：

✅ 本次完成（学习进度）：
  - 修复 sensor_tests.py  flaky测试:
    * VirtualForceSensor::test_virtual_force_surface_contact 断言稳定性问题
    * assertNotEqual → assertGreater(abs(...), 0.5) 避免随机噪声干扰
  - 全量测试通过验证: 1585项测试 ✓ | 38项跳过 | 0项失败

📊 SuperModel整体状态 (v1.85.0)：

  ✅ 代码模块 100% 完成:
    传感器: vision + audio + tactile + force + imu + encoders + manager (7种)
    融合层: cross_modal_fusion + sensor_fusion (EKF/互补滤波/Madgwick)
    控制层: motor + motion + pid + planner + safety + autotune + mpc + impedance + obstacle_avoidance + teleop + multi_agent (11类)
    仿真层: PyBullet + MuJoCo + Gazebo + Gymnasium + 基础仿真
    测试: 1585项测试全通过

  ✅ 设计文档 100% 完成:
    docs/SPEC.md: 模块接口设计 (触觉/力觉/IMU完整接口表)
    docs/AGV_SPEC.md: AGV五级规格对照表 (12章节完整)
    docs/MODULE_INDEX.md: 全部模块索引
    docs/SENSOR_API_GUIDE.md: 传感器API指南

  ✅ AGV五级规格体系完整:
    S级: 教学/实验室, ¥5-15K, 50Hz, RPi 4B
    M级: 物流/制造业, ¥15-50K, 100Hz, RK3588
    L级: 重载精密, ¥50-150K, 200Hz, Orin NX
    XL级: 特种/协作, ¥150-500K, 500Hz, Orin AGX
    XXL级: 航空航天/船舶, >¥500K, 1000Hz, 具身智能大脑

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
        if result.get("code") == 0:
            print("✅ 飞书消息发送成功!")
        else:
            print(f"⚠️ 发送失败: {result}")
except urllib.error.URLError as e:
    print(f"⚠️ 网络错误: {e}")
