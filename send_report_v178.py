#!/usr/bin/env python3
"""Send Feishu progress report v1.78"""
import json, urllib.request, urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """SuperModel项目进度更新 (v1.78.0 - 2026-04-08 21:55)：

✅ 本次完成（学习进度）：
  - 轨迹规划与跟踪模块全面完善:
    * 新增 VelocityProfiler 类 (梯形/S曲线速度规划)
    * 新增 MinimumSnapTrajectory 类 (最小Snap轨迹生成)
    * 新增 PurePursuitTracker 类 (几何跟踪控制器)
    * 新增 StanleyTracker 类 (横向误差跟踪控制器)
    * 新增 PIDTrajectoryTracker 类 (PID轨迹跟踪)
    * 新增 RRTStarPlanner 路径规划器
    * 新增 TrajectoryPlanner / TrajectoryPoint / Trajectory 等基础数据结构
    * 新增 trajectory_planning.py 专用测试接口模块
  - 控制模块增强:
    * AGV模块新增 PurePursuitTracker/StanleyTracker/PIDTrajectoryTracker
  - 修复 planner_tests.py 收集错误 (缺少 VelocityProfiler 等导入)
  - 测试: 1491项全通过 ✅ (232→1491项)

📊 SuperModel整体状态 (v1.78.0)：
  传感器层: vision + audio + tactile + force + imu + encoders + manager (7模块) ✅
  触觉模块: TactileArray + PressureProcessor + VirtualTactileSensor ✅
  力觉模块: ForceTorqueSensor + WrenchProcessor + VirtualForceSensor ✅
  IMU模块: IMUSensor + PoseEstimator + VirtualIMUSensor (Madgwick/EKF/互补滤波) ✅
  融合层: cross_modal_fusion + sensor_fusion ✅
  认知层: world_model + dreamer + 自监督 + 自主学习 ✅
  执行层: 22个控制子模块 (motion/planner/traj/mpc/impedance/force等) ✅
  轨迹规划: VelocityProfiler + MinimumSnap + PurePursuit + Stanley + PID + RRT* ✅
  测试: 1491项全通过 ✅
  文档: MODULE_INTERFACE + AGV五级规格表 + 控制子系统规格 ✅

🔧 后续计划：
  - v2.0: 真实AGV部署 + 视觉-语言-动作多模态对齐
  - v3.0: 完整超模态LLM + 具身强化学习

📁 GitHub: https://github.com/DIT4FUN/SuperModel"""

token_url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
token_data = json.dumps({"app_id": APP_ID, "app_secret": APP_SECRET}).encode()
token_req = urllib.request.Request(token_url, data=token_data, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(token_req) as resp:
    token_result = json.loads(resp.read())
access_token = token_result.get("tenant_access_token", "")

msg_req = urllib.request.Request(
    "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id",
    data=json.dumps({
        "receive_id": CHAT_ID,
        "msg_type": "text",
        "content": json.dumps({"text": MESSAGE})
    }).encode(),
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
)
try:
    with urllib.request.urlopen(msg_req) as resp:
        result = json.loads(resp.read())
        if result.get("code") == 0:
            print("Report sent successfully!")
        else:
            print(f"Failed: {result}")
except urllib.error.URLError as e:
    print(f"URL error: {e}")
