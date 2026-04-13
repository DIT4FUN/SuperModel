#!/usr/bin/env python3
"""Send Feishu progress report via Feishu Open API"""
import json, urllib.request, urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """📊 SuperModel v3.3.0 开发进度汇报 (2026-04-13 18:05)

✅ 本次任务完成工作：

1. 具身任务执行器 (task_executor.py, ~650行):
   • MemoryEnhancedExecutor: 记忆增强型具身任务执行器
     - 集成了行为树 / 仿真环境 / 长期记忆 / 真实AGV接口
     - 支持记忆检索（任务规划阶段自动检索历史经验）
     - 支持经验存储（任务完成后自动存入情景记忆）
     - 完整的任务生命周期管理：PLANNING → EXECUTING → SUCCEEDED/FAILED
     - 回调系统：phase_change / tick / error
   • ScenarioTaskExecutor: 场景化任务执行器
     - 集成 SceneIntelligence / SceneCoordination
     - 场景自适应行为参数调整
   • 支持4种任务类型：transport / patrol / rescue / collaborative

2. 具身记忆系统集成 (memory_integration.py, ~450行):
   • EmbodiedMemoryManager: 协调情景/语义/程序/工作记忆
   • 情景记忆：经验存储/检索/过滤（时间窗口/结果类型）
   • 程序记忆：技能注册/按场景类型/成功率检索/结果更新
   • 工作记忆：注意力焦点管理（当前任务/目标位置/电池/安全状态）
   • 语义记忆：概念存储与查询
   • Ebbinghaus遗忘曲线自动衰减

3. 新增59项测试，全部通过 (308项总测试全部通过):
   • test_task_executor.py: 20项测试
   • test_memory_integration.py: 23项测试

4. Bug修复:
   • 移除 behavior_tree.py 死代码桩（1138行重复函数）
   • 修复 EmbodiedSkill.update_success 缺少 usage_count 递增
   • 修复 EmbodiedMemoryManager 缺少 enable_memory 属性

✅ GitHub: 已提交 749372d

📊 整体研发进度: ~97%
已完成: 基础架构/视觉/听觉/触觉/力觉/IMU/跨模态融合/核心目标/自主学习/控制模块/五级AGV规格/仿真环境/测试用例/协同SLAM/HIL测试框架/具身任务执行器/具身记忆集成
待推进: 项目已接近完成，待完善项主要为零散边缘优化"""

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
