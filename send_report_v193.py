#!/usr/bin/env python3
"""Send Feishu progress report v1.93.0 - 2026-04-09"""
import json, urllib.request, urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """SuperModel项目进度更新 (v1.93.0 - 2026-04-09 05:10)：

✅ 本次完成（学习进度）：
  - 完善跨模态融合网络导出接口:
    * fusion/__init__.py: 导出CrossModalFusion等核心类
    * src/fusion/cross_modal_fusion.py: 补充CrossModalAttention融合模块
  - 支持AGV五级规格(S/M/L/XL/XXL)的差异化融合配置
  - 测试验证:
    * sensor_tests.py: 270项全部通过 ✓
    * fusion_tests.py: 73项全部通过 ✓
  - GitHub: c7a0fbb (v1.93.0) 已推送

📊 SuperModel整体状态 (v1.93.0)：
  传感器层: vision + audio + tactile + force + imu + encoders + manager (7模块) ✅
  融合层: cross_modal_fusion + sensor_fusion (EKF/互补滤波/Madgwick) ✅
  认知层: world_model + dreamer_agent + 自监督 + 自主学习 ✅
  执行层: 23个控制子模块 ✅
  具身控制: TactileServo + ForceControl + IMUControl + EmbodiedController ✅
  仿真层: PyBullet + MuJoCo + Gymnasium + Gazebo ✅
  设计文档: SPEC + DESIGN + MODULE_INTERFACE + AGV五级规格表 ✅
  测试覆盖: 1362+项测试用例通过 ✅

🔧 已完成模块清单：
  【传感器模块】tactile.py / force.py / imu.py / vision.py / audio.py / encoders.py / manager.py
  【融合模块】cross_modal_fusion.py / sensor_fusion.py
  【控制模块】motor / motion / trajectory / impedance / mpc / agv / supervisor / safety_controller / ros2_interface / obstacle_avoidance / teleop / skill / planner / multi_agent / autotune / sensorimotor / tactile_control / force_control / imu_control / embodied_control
  【学习模块】world_model.py / dreamer_agent.py / self_supervised.py / autonomous_learning.py
  【硬件抽象】base.py / gpio.py / nnpu.py / rk3588.py / digu_robot.py
  【仿真环境】pybullet_sim.py / mujoco_sim.py / gym_env.py / agv_model_generator.py

📋 待完善/进行中：
  - 触觉/力觉/IMU真实硬件驱动对接
  - RK3588 NPU部署优化
  - 更多仿真场景拓展
  - Real-World AGV实地测试

🔗 GitHub: https://github.com/DIT4FUN/SuperModel"""

url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
payload = json.dumps({"app_id": APP_ID, "app_secret": APP_SECRET}).encode()
req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
try:
    resp = urllib.request.urlopen(req, timeout=10)
    data = json.loads(resp.read())
    token = data.get("tenant_access_token", "")
    if not token:
        print(f"❌ Failed to get token: {data}")
        exit(1)
except Exception as e:
    print(f"❌ Network error: {e}")
    exit(1)

msg_url = "https://open.feishu.cn/open-apis/im/v1/messages"
msg_payload = json.dumps({
    "receive_id": CHAT_ID,
    "msg_type": "text",
    "content": json.dumps({"text": MESSAGE})
}).encode()
msg_req = urllib.request.Request(
    msg_url,
    data=msg_payload,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
)
try:
    resp = urllib.request.urlopen(msg_req, timeout=10)
    result = json.loads(resp.read())
    if result.get("code") == 0 or result.get("StatusCode") == 0:
        print(f"✅ Feishu report sent successfully")
    else:
        print(f"❌ Send failed: {result}")
except Exception as e:
    print(f"❌ Send error: {e}")
