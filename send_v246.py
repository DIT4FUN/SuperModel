#!/usr/bin/env python3
"""Send Feishu progress report via Feishu Open API"""
import json, urllib.request, urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """SuperModel项目进度更新 (v2.46.0 - 2026-04-10 12:06)：

✅ 本次完成（学习进度）：
  - 新增AGV五级传感器-控制集成测试 (tests/five_grade_sensor_control_tests.py, 454行, 24项)
    * 触觉传感器五级规格测试 (TactileArray/VirtualTactileSensor, 3项)
      - 验证所有五级触觉规格完整性和递进性
      - 测试S(8×8@50Hz)→M(16×16@100Hz)→L(24×24@200Hz)→XL(32×32@500Hz)→XXL(48×48@1000Hz)
    * 力觉传感器五级规格测试 (ForceTorqueSensor/VirtualForceSensor, 3项)
      - 验证6轴力觉采样率/量程/分辨率随等级提升
      - 测试S(3轴@100Hz)→XXL(6轴@5000Hz)全系列
    * IMU传感器五级规格测试 (IMUSensor/VirtualIMUSensor, 3项)
      - 验证IMU采样率递增、噪声密度递减
      - 测试MPU6050@100Hz → ADIS16470×2@2000Hz
    * 速度控制五级规格测试 (VelocityPIDController/SVelocityProfilePlanner, 4项)
      - 验证AGV五级速度控制参数一致性
      - 50Hz→1000Hz控制频率全覆盖
    * 仿真环境五级规格测试 (SimulationInterface/EmbodiedSimulator, 5项)
      - 验证AGV仿真参数五级完整性
      - 验证具身仿真等级递进性(dt/采样率/速度)
    * 传感器-融合-控制pipeline测试 (3项)
      - 触觉→质量评估→控制指令生成完整链路
      - 力觉+IMU→互补滤波→融合决策
      - 完整传感器→姿态估计→速度控制闭环(100周期)
    * 五级规格一致性测试 (3项)
      - 控制频率与规格严格一致验证
      - 采样率一致性验证
      - 等级能力递进验证(S→XXL)

  - 测试验证: 429项传感器/融合/五级集成测试全部通过 ✅
    * sensor_tests.py: 332项 ✅
    * fusion_tests.py: 73项 ✅
    * five_grade_sensor_control_tests.py: 24项 ✅

  - GitHub已推送: dc34d4d → 34720ff (v2.46.0)

📊 SuperModel整体状态 (v2.46.0)：
  代码模块: sensors(7) + fusion(2) + control(34) + learning + core + simulation
  测试: 五级集成测试(24) + sensor(332) + fusion(73) = 429项全通过 ✅
  文档: SPEC.md(22章节) + MODULE_INDEX + AGV五级规格总表 + 部署指南
  GitHub: https://github.com/DIT4FUN/SuperModel

🎯 已完成模块清单 (v2.46.0):
  ✅ 传感器层: vision + audio + tactile + force + imu + encoders + manager
  ✅ 融合层: cross_modal_fusion + sensor_fusion (EKF/互补滤波/Madgwick AHRS)
  ✅ 认知层: core_brain + context_understanding + world_model + dreamer + 自监督
  ✅ 执行层: 34个控制子模块 (motor/motion/trajectory/skill/planner/impedance/mpc/safety/agv/multi_agent/teleop/supervisor/sensor_fusion_control/skill_dispatcher/behavior_tree/bias_compensation/simulation/embodied_sim/velocity_control/...)
  ✅ 仿真层: embodied_sim + Gymnasium + PyBullet + MuJoCo
  ✅ 核心目标系统: P0-P5六层目标优先级体系
  ✅ 五级集成测试: 24项新增，全覆盖触觉/力觉/IMU/速度控制/仿真/一致性

📅 下一步建议:
  - MuJoCo强化学习环境集成
  - RK3588 NPU部署优化
  - 真实机器人验证
"""

url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
data = json.dumps({"app_id": APP_ID, "app_secret": APP_SECRET}).encode()
req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
try:
    resp = urllib.request.urlopen(req, timeout=10)
    token_data = json.loads(resp.read())
    token = token_data.get("tenant_access_token", "")
except Exception as e:
    print(f"Token error: {e}")
    exit(1)

msg_url = "https://open.feishu.cn/open-apis/im/v1/messages"
msg_data = {
    "receive_id": CHAT_ID,
    "msg_type": "text",
    "content": json.dumps({"text": MESSAGE})
}
req2 = urllib.request.Request(
    msg_url,
    data=json.dumps(msg_data).encode(),
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    },
    method="POST"
)
try:
    resp2 = urllib.request.urlopen(req2, timeout=10)
    result = json.loads(resp2.read())
    if result.get("code") == 0:
        print(f"Feishu report sent successfully!")
    else:
        print(f"Failed: {result}")
except Exception as e:
    print(f"Message error: {e}")
    exit(1)
