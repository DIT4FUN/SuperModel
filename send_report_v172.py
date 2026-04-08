#!/usr/bin/env python3
"""Send Feishu progress report v1.72"""
import json, urllib.request, urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """SuperModel项目进度更新 (v1.72.0 - 2026-04-08 16:55)：

✅ 本次完成（学习进度）：
  - 完善模块接口设计文档 (MODULE_INDEX.md 新增附录D/E/F):
    * 附录D: 传感器完整接口规格 (tactile/force/imu/sensor_fusion/sensorimotor)
    * 附录E: 控制子系统五级规格详解 (控制频率/闭环延迟/计算通信)
    * 附录F: SuperModel版本路线图 (v1.0→v3.0)
  - 传感器+融合测试: 207项全部通过 ✅

📊 SuperModel整体状态 (v1.72.0)：
  传感器层: vision + audio + tactile + force + imu + encoders + manager (7模块) ✅
  融合层: cross_modal_fusion + sensor_fusion ✅
  认知层: scene + world_model + dreamer + 自监督 + 自主学习 ✅
  执行层: 22个控制子模块 ✅
  仿真层: PyBullet + MuJoCo + Gymnasium + Gazebo ✅
  测试: 传感器+融合 207项全通过 ✅
  文档: MODULE_INDEX(含接口规格) + AGV_SPEC + DESIGN + HARDWARE_SPEC ✅

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
except Exception as e:
    print(f"Error: {e}")
