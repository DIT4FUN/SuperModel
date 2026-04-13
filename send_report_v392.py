#!/usr/bin/env python3
"""v3.9.2 飞书进度汇报"""
import json
import urllib.request
import urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """📊 SuperModel v3.9.2 开发进度汇报 (2026-04-14 05:00)

✅ 本次任务完成工作：

1. 具身长期记忆系统 (src/memory/embodied_long_term_memory.py, 975行):
   • EmbodiedExperience: 具身经验条目(场景上下文/动作序列/结果评估/AGV等级)
   • SceneMemoryIndex: 场景-记忆关联索引(按场景/标签/经验类型快速检索)
   • SkillMemoryRecord: 技能记忆记录(执行统计/学习曲线/掌握度评估/遗忘检测)
   • AGVGradeAwareMemory: AGV等级感知记忆(跨等级经验迁移效益计算)
   • ExperienceCompressor: 经验压缩器(相似经验合并/存储优化/知识蒸馏)
   • MemoryBasedTaskPredictor: 基于记忆的任务结果预测器(成功率预测/置信度/参考经验)
   • EmbodiedLongTermMemory: 完整具身长期记忆系统(经验存储/检索/技能记忆/知识导出)
   • src/memory/__init__.py 完整导出新增模块全部符号

2. 具身长期记忆测试套件 (tests/test_embodied_long_term_memory.py, 39项测试):
   • TestEmbodiedExperience: 经验数据类序列化/反序列化/奖励计算
   • TestSceneMemoryIndex: 场景检索/标签过滤/经验类型过滤/统计
   • TestSkillMemoryRecord: 技能记录更新/掌握度判定/遗忘检测/学习曲线
   • TestAGVGradeAwareMemory: 等级感知检索/迁移效益计算
   • TestExperienceCompressor: 相似度计算/经验压缩/单经验处理
   • TestMemoryBasedTaskPredictor: 成功率预测/无历史默认/失败原因统计
   • TestEmbodiedLongTermMemory: 完整集成测试(存储/检索/技能/预测/导出)
   • TestGlobalInstance: 全局工厂函数单例测试
   • 全部39项通过 ✅

✅ 测试状态: 39项新测试全部通过，embodied测试套件1155项100%通过

✅ GitHub: 已提交 4284aa5 (v3.9.2)

📊 整体研发进度: ~96%
已完成: 基础架构/视觉/听觉/触觉/力觉/IMU/跨模态融合/核心目标/自主学习/控制模块/五级AGV规格/仿真环境/测试用例/协同SLAM/HIL测试/具身Pipeline(含状态持久化)/技能注册表/记忆系统/场景智能/联邦学习/具身控制流程文档/部署管理/元认知模块/具身长期记忆系统
待推进: 视觉-语言-动作端到端/超模态大模型集成"""

def send_message(content):
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    payload = {"app_id": APP_ID, "app_secret": APP_SECRET}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as resp:
        token_data = json.loads(resp.read())
    token = token_data.get("tenant_access_token", "")
    
    msg_url = "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id"
    msg_data = {
        "receive_id": CHAT_ID,
        "msg_type": "text",
        "content": json.dumps({"text": content}),
    }
    msg_req = urllib.request.Request(
        msg_url,
        data=json.dumps(msg_data).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(msg_req) as resp:
            result = json.loads(resp.read())
            print(f"Message sent: code={result.get('code')}, msg_id={result.get('data', {}).get('message_id', 'N/A')}")
            return result
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"Error: {e.code} - {body}")
        return None

if __name__ == "__main__":
    send_message(MESSAGE)
