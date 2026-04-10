#!/usr/bin/env python3
"""Send Feishu progress report via Feishu Open API"""
import json, urllib.request, urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """SuperModel项目进度更新 (v2.45.0 - 2026-04-10 11:42)：

✅ 本次完成（学习进度）：
  - 新增速度控制模块 (src/control/velocity_control.py, 988行)
    * AGVVelocityController: AGV完整速度控制器 (运动学/PID闭环/规划/打滑检测)
    * SVelocityProfilePlanner: S曲线速度规划器 (梯形+S曲线双模式)
    * FrictionCompensator: 库伦+粘滞摩擦补偿器
    * WheelVelocitySynchronizer: 轮速同步与自适应打滑校正
    * VelocityPIDController: 自适应PID控制器 (积分抗饱和/微分滤波/前馈)
    * AGV五级规格 (S/M/L/XL/XXL): 50Hz→1000Hz连续覆盖
      - S级: 50Hz, 梯形规划, 无摩擦补偿
      - M级: 100Hz, 梯形规划, 摩擦补偿+前馈+打滑检测
      - L级: 200Hz, S曲线规划, 自适应增益, Xenomai实时
      - XL级: 500Hz, S曲线规划, 全功能
      - XXL级: 1000Hz, S曲线规划, Xenomai+FPGA

  - 测试用例 (tests/velocity_control_tests.py, 78项)
    * AGV五级规格验证 (10项)
    * S曲线速度规划 (15项)
    * 摩擦补偿 (7项)
    * 轮速同步 (7项)
    * PID控制器 (10项)
    * AGV速度控制器 (11项)
    * 集成测试 (5项)
    * 边界条件 (13项)
    * 78项全部通过 ✅

  - 文档更新 (docs/SPEC.md, 新增第21章)
    * 速度控制模块规格: 概述/核心组件/AGV五级规格/接口方法/使用示例
    * 版本历史更新至v2.45.0

  - GitHub已推送: 8732f5b → dc34d4d

📊 SuperModel整体状态 (v2.45.0)：
  代码模块: sensors(7) + fusion(2) + control(30+) + learning + core + simulation
    * 新增velocity_control.py: 5个核心类, 988行代码
  测试: velocity_control(78) + sensor(332) + fusion(73) = 483项全通过 ✅
  文档: SPEC.md(22章节) + MODULE_INDEX + AGV五级规格总表 + 部署指南
  GitHub: https://github.com/DIT4FUN/SuperModel

🎯 已完成模块清单 (v2.45.0):
  ✅ 传感器层: vision + audio + tactile + force + imu + encoders + manager
  ✅ 融合层: cross_modal_fusion + sensor_fusion (EKF/互补滤波/Madgwick AHRS)
  ✅ 认知层: core_brain + context_understanding + world_model + dreamer + 自监督
  ✅ 执行层: 30+控制子模块 (含新增velocity_control)
  ✅ 仿真层: embodied_sim + Gymnasium + PyBullet + MuJoCo
  ✅ 核心目标系统: P0-P5六层目标优先级体系
  ✅ 测试: 483项全通过

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
