#!/usr/bin/env python3
"""Send Feishu progress report via Feishu Open API"""
import json, urllib.request, urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """📊 SuperModel v3.6.1 开发进度汇报 (2026-04-13 20:57)

✅ 本次任务完成工作：

1. 具身智能控制流程文档 (docs/EMBODIED_CONTROL_FLOW.md, ~560行):
   • Pipeline 启动流程 (初始化序列/状态机)
   • 任务执行主流程 (技能匹配→行为树规划→Executor执行)
   • 行为树规划流程 (节点类型/场景自适应)
   • 记忆系统交互流程 (情景/语义/程序/工作记忆)
   • 场景自适应流程 (SceneIntelligence)
   • 硬件在环(HIL)流程架构
   • 联邦学习协同流程 (FedAvg/差分隐私/拜占庭容错)
   • 状态机与错误处理策略
   • 五级AGV规格适配对照表
   • API速查与配置参数表

✅ 测试状态: 517项测试全部通过

✅ GitHub: 已提交 1a32575 (v3.6.1)

📊 整体研发进度: ~95%
已完成: 基础架构/视觉/听觉/触觉/力觉/IMU/跨模态融合/核心目标/自主学习/控制模块/五级AGV规格/仿真环境/测试用例/协同SLAM/HIL测试/具身Pipeline/技能注册表/记忆系统/场景智能/联邦学习/具身控制流程文档
待推进: 长期记忆系统增强/多机协同优化/边缘场景覆盖"""

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
