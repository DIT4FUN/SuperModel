#!/usr/bin/env python3
"""v3.8.3 飞书进度汇报"""
import json
import urllib.request
import urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """📊 SuperModel v3.8.3 开发进度汇报 (2026-04-14 01:30)

✅ 本次任务完成工作：

1. 部署模块测试 (tests/embodiment/test_deployment.py, ~1050行, 96项):
   • DeploymentConfig 测试 (所有AGV等级, 传感器开关, 健康检查参数)
   • HealthCheckResult 测试 (状态判定, 延迟记录, 时间戳, 详情字典)
   • DeploymentValidator 测试 (无效等级, 紧急停车禁用警告, 健康间隔, 五级规格)
   • HealthMonitor 测试 (并发报告, 回调系统, 错误计数, 历史大小限制)
   • EmergencyProcedure 测试 (IMU倾角+力觉碰撞+触觉压力安全检查)
   • DeploymentManager 测试 (完整生命周期, 降级模式, 紧急停车集成)
   • 五级AGV集成测试 (S/M/L/XL/XXL参数化, 传感器数量/控制频率/最大速度)
   • 并发部署测试 (多管理器并发, 并发健康报告, 并发状态变化)
   • 边缘情况测试 (零间隔, 极端配置, 超长消息)
   • 96项全部通过 ✅

2. 模块覆盖补全:
   • src/embodied/deployment.py 此前无任何测试, 现100%覆盖
   • DeploymentValidator / HealthMonitor / EmergencyProcedure 全方法覆盖
   • 所有AGV五级等级参数化验证

3. 文档更新:
   • CHANGELOG.md → v3.8.3
   • PROGRESS.md → v3.8.3
   • MODULE_INDEX.md → v3.8.3

✅ 测试状态: 971项pytest测试100%通过 (新增96项)

✅ GitHub: 已提交 e51eb47 (v3.8.3)

📊 整体研发进度: ~96%
已完成: 基础架构/视觉/听觉/触觉/力觉/IMU/跨模态融合/核心目标/自主学习/控制模块/五级AGV规格/仿真环境/测试用例/协同SLAM/HIL测试/具身Pipeline/技能注册表/记忆系统/场景智能/联邦学习/具身控制流程文档/部署管理
待推进: 视觉-语言-动作端到端/超模态大模型集成"""

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
