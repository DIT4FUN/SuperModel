#!/usr/bin/env python3
"""Send Feishu progress report via Feishu Open API"""
import json, urllib.request, urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """SuperModel项目进度更新 (v2.54.0 - 2026-04-10 15:41)：

✅ 本次完成（学习进度）：
  - sensors/ 模块新增AGV五级规格表：
    * tactile.py: AGV_TACTILE_GRADES (S/M/L/XL/XXL五级完整规格)
    * force.py: AGV_FORCE_GRADES (S/M/L/XL/XXL五级完整规格)
    * imu.py: AGV_IMU_GRADES (S/M/L/XL/XXL五级完整规格)
    * 新增工厂函数: create_tactile/force/imu_sensor_for_grade()
    * 新增能力表函数: list_tactile/force/imu_capabilities()
    * sensors/__init__.py: 导出所有新增五级规格符号
  - 触觉五级规格: 8x8@50Hz(S) → 48x48@1000Hz(XXL)
  - 力觉五级规格: 3轴±100N@100Hz(S) → 6轴±5000N@5000Hz(XXL)
  - IMU五级规格: MPU6050@100Hz(S) → Dual ADIS16470@2000Hz(XXL)
  - GitHub已推送: b7e3e45 → 0321d97

📊 SuperModel整体状态 (v2.54.0)：
  传感器层: vision + audio + tactile + force + imu + encoders + manager (7模块) ✅
  触觉/力觉/IMU: 完整五级AGV规格表 + 工厂函数 + 能力展示 ✅
  融合层: cross_modal_fusion + sensor_fusion (EKF/互补滤波/Madgwick) ✅
  核心层: core_goals/safety_shield/value_judgment/self_preservation/self_evolution ✅
  执行层: 30+控制子模块（AGV运动学、PID、阻抗、MPC、安全监控等）✅
  仿真层: PyBullet + MuJoCo + Gymnasium + Gazebo + 实时监控器 ✅
  测试: 2693项测试全通过 ✅
  文档: SPEC.md + MODULE_INTERFACE + AGV五级规格总表 + 部署实战 ✅

🔜 下一步建议：
  - 真实AGV机器人集成测试
  - RK3588 NPU边缘部署优化
  - 端到端具身智能长期运行测试
"""

def send_feishu_message(app_id: str, app_secret: str, chat_id: str, message: str):
    """发送飞书消息"""
    # 获取tenant_access_token
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    data = json.dumps({"app_id": app_id, "app_secret": app_secret}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        token_data = json.loads(resp.read())
    token = token_data.get("tenant_access_token", "")
    if not token:
        print("Failed to get token")
        return

    # 发送消息
    msg_url = "https://open.feishu.cn/open-apis/im/v1/messages"
    params = {"receive_id_type": "chat_id"}
    msg_data = json.dumps({
        "receive_id": chat_id,
        "msg_type": "text",
        "content": json.dumps({"text": message})
    }).encode()
    msg_req = urllib.request.Request(
        msg_url + "?receive_id_type=chat_id",
        data=msg_data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
    )
    try:
        with urllib.request.urlopen(msg_req) as mr:
            result = json.loads(mr.read())
            print(f"Message sent: {result.get('code', -1)}")
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} {e.reason}")
        print(e.read())

if __name__ == "__main__":
    send_feishu_message(APP_ID, APP_SECRET, CHAT_ID, MESSAGE)
