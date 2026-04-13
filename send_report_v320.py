#!/usr/bin/env python3
"""Send Feishu progress report via Feishu Open API"""
import json, urllib.request, urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """📊 SuperModel v3.2.0 开发进度汇报 (2026-04-13 17:05)

✅ 本次任务完成工作：

1. 协同SLAM模块 (collaborative_slam.py, ~700行):
   • MapFragment: 地图碎片管理，支持特征点合并与坐标变换
   • CollaborativeSlamAgent: 单AGV代理，管理本地地图与位姿估计
   • MapFusionEngine: ICP地图融合引擎，支持多碎片对齐与融合
   • CollaborativeSlamCoordinator: 多AGV协同SLAM协调器
   • 支持基于特征点的地图注册、地图片段融合、协同定位查询

2. HIL硬件在环测试框架 (hil_testing.py, ~650行):
   • SensorReplay: 传感器数据回放器，支持速度控制/seek/回调
   • CANBusHILSimulator: CAN Bus HIL模拟器，支持帧注入与监控统计
   • ControlCommandValidator: 控制指令验证，支持范围检查与序列验证
   • SensorActuatorHILLoop: 传感器-执行器闭环测试循环
   • HILTestRunner: 测试运行器与自动JSON报告生成

3. 新增56项测试，全部通过:
   • test_collaborative_slam.py: 24项 (FeaturePoint/MapFragment/Agent/FusionEngine/Coordinator)
   • test_hil_testing.py: 32项 (CAN/Validator/Replay/Loop/Runner)
   • 测试总数: 249项全部通过 (193原有 + 56新增)

✅ GitHub: 已提交 c8395ac → 977caf6

📊 整体研发进度: ~96%
已完成: 基础架构/视觉/听觉/触觉/力觉/IMU/跨模态融合/核心目标/自主学习/控制模块/五级AGV规格/仿真环境/测试用例/协同SLAM/HIL测试框架
待推进: 具身智能场景化应用深化（医疗/工业现场）/ 多机协同算法（已基本完成协同SLAM）/ 真实AGV硬件在环验证（已搭建HIL框架）"""

def send_message(content):
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    payload = {"app_id": APP_ID, "app_secret": APP_SECRET}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req)
        token_data = json.loads(resp.read())
        access_token = token_data.get("tenant_access_token", "")
        if not access_token:
            print("Failed to get access token")
            return
        msg_url = "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id"
        msg_payload = {
            "receive_id": CHAT_ID,
            "msg_type": "text",
            "content": json.dumps({"text": content})
        }
        msg_data = json.dumps(msg_payload).encode("utf-8")
        msg_req = urllib.request.Request(
            msg_url, data=msg_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}"
            }
        )
        msg_resp = urllib.request.urlopen(msg_req)
        print("Message sent successfully:", msg_resp.read())
    except urllib.error.HTTPError as e:
        print(f"HTTP error: {e.code} - {e.read()}")

if __name__ == "__main__":
    send_message(MESSAGE)
