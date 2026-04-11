#!/usr/bin/env python3
"""Send Feishu progress report via Feishu Open API"""
import json, urllib.request, urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """SuperModel项目进度更新 (v2.67.0 - 2026-04-11 07:58)：

✅ 本次完成（Bug修复 + 测试完善）：
  - 修复28项传感器测试全部通过：
    * force.py: estimate_payload() abs()修复重力方向判断
    * force.py: WrenchProcessor.compute_equivalent_wrench_at() 新增方法
    * force.py: VirtualForceSensor 上下文管理器支持
    * imu.py: VirtualIMUSensor.simulate_motion() 新增方法
    * imu.py: VirtualIMUSensor 上下文管理器支持
    * imu.py: PoseEstimator.get_rotation_matrix() 新增方法
    * tactile.py: PressureProcessor.compute_pressure_histogram bins=10修复
    * tactile.py: TactileArray _frame_buffer属性(防溢出，限100帧)
    * tactile.py: VirtualTactileSensor 上下文管理器+压力单位归一化
    * tactile.py: estimate_grip_quality()补全contact_area/uniformity/stability
  - 测试结果: 2625 passed, 2 failed (pre-existing), 38 skipped
  - GitHub已推送: 89e337f

📊 SuperModel整体状态 (v2.67.0)：
  整体进度: ~95% (基础模块全完成，具身智能+场景化应用+长期记忆已集成)
  传感器: vision + audio + tactile + force + imu + encoders (7模块)
  融合: cross_modal_fusion + sensor_fusion
  认知: scene_understanding + world_model + dreamer_agent
  执行: 40+控制子模块
  具身智能: 行为树 + 仿真增强 + 真实AGV接口 + 场景感知协调
  记忆系统: 长期 + 情景 + 语义 + 程序 + 工作 + 检索 + 整合
  测试: 2625+项测试通过

🔜 下一步: 边缘部署优化(RK3588 NPU)、真实机器人验证"""

req = urllib.request.Request(
    "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
    data=json.dumps({"app_id": APP_ID, "app_secret": APP_SECRET}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST"
)
with urllib.request.urlopen(req) as resp:
    token_data = json.loads(resp.read())
    token = token_data["tenant_access_token"]

# 发送消息
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
        print(f"消息发送成功: {result.get('code', '')} {result.get('msg', '')}")
except urllib.error.HTTPError as e:
    print(f"发送失败: {e.code} {e.read()}")
