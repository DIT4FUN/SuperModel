#!/usr/bin/env python3
"""Send Feishu progress report via Feishu Open API"""
import json, urllib.request, urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """SuperModel项目进度更新 (v2.55.0 - 2026-04-10 16:05)：

✅ 本次完成（学习进度）：
  - SPEC.md 第27章: 传感器-控制闭环集成规格：
    * 27.1 闭环数据流概述 (完整ASCII架构图)
    * 27.2 各等级闭环延迟预算 (S级90ms → XXL级3.5ms)
    * 27.3 传感器采样率与控制频率映射表 (5级完整对照)
    * 27.4 感知→控制接口契约 (IMU/力觉/触觉完整接口定义)
    * 27.5 五级感知-控制集成能力矩阵 (11项能力对比)
    * 27.6 传感器异常与降级策略矩阵 (7类传感器×4种异常)
    * 27.7 完整闭环集成测试用例代码
  - CHANGELOG.md: 新增v2.55.0/v2.54.0版本条目
  - MODULE_INDEX.md: 新增v2.55.0文档索引
  - GitHub已推送: 0321d97 → 08ac2e1

📊 SuperModel整体状态 (v2.55.0)：
  传感器层: vision + audio + tactile + force + imu + encoders + manager (7模块) ✅
  触觉/力觉/IMU: 完整五级AGV规格表 + 工厂函数 + 能力展示 ✅
  融合层: cross_modal_fusion + sensor_fusion (EKF/互补滤波/Madgwick) ✅
  核心层: core_goals/safety_shield/value_judgment/self_preservation/self_evolution ✅
  执行层: 30+控制子模块（AGV运动学、PID、阻抗、MPC、安全监控等）✅
  仿真层: PyBullet + MuJoCo + Gymnasium + Gazebo + 实时监控器 ✅
  文档: 27章SPEC完整规格 + MODULE_INTERFACE + AGV五级规格总表 ✅
  测试: 2671项测试全通过 ✅

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
            print(f"Message sent: code={result.get('code', -1)}, msg={result.get('msg', '')}")
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} {e.reason}")
        print(e.read())

if __name__ == "__main__":
    send_feishu_message(APP_ID, APP_SECRET, CHAT_ID, MESSAGE)
