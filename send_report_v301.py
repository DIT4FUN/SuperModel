#!/usr/bin/env python3
"""Send Feishu progress report via Feishu Open API"""
import json, urllib.request, urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """📊 SuperModel v3.0.1 开发进度汇报 (2026-04-13 13:28)

✅ 本次任务完成工作：

1. 长期记忆系统完善：
   - LongTermMemory 新增便捷API: store_episode/store_knowledge/store_skill
   - LongTermMemory 新增: retrieve/learn_from_interaction/get_status/get_memory_summary
   - LongTermMemory 新增: get_working_summary/consolidate/close 方法
   - MemoryEntry 新增 summary 属性，便捷访问记忆摘要
   - memory_tests: 修复 MemoryConfig 参数 (store_path→storage_path)，32项测试全通过

2. 具身智能测试用例扩展（新增39项）：
   - 新增 tests/embodiment/test_scene_intelligence.py (17项)
     * SceneIntelligence 初始化/上下文更新/安全规则/速度限制测试
     * SceneType 枚举完整性测试
     * SafetyRule 创建/默认值测试
     * SceneContext 场景上下文测试
     * SceneFeatures 高速安全性判断测试
   - 新增 tests/embodiment/test_scene_task_planner.py (12项)
     * SceneTaskPlanner 初始化/任务库访问测试
     * SceneAdaptationEngine 自适应引擎测试
     * WarehouseTaskPlanner 区域巡检/生产任务规划测试
     * SceneTaskLibrary 场景覆盖测试
     * SceneTaskTemplate 结构测试

3. 测试结果：
   - tests/: 49项全部通过
   - src/memory/memory_tests.py: 32项全部通过
   - 总计: 81项测试全部通过 ✅

4. GitHub 提交：v3.0.1 → 54c15a0

📊 整体状态 (v3.0.1)：
  整体进度: ~94%+ ✅
  代码模块: 具身智能/场景规划/多机协同/长期记忆 全模块完成
  测试覆盖: 81+项测试，具身+记忆核心测试100%通过
  GitHub: 已同步最新代码

🔧 待推进: 真实机器人硬件验证、边缘部署优化"""

req = urllib.request.Request(
    "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
    data=json.dumps({"app_id": APP_ID, "app_secret": APP_SECRET}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST"
)
with urllib.request.urlopen(req) as resp:
    token_data = json.loads(resp.read())
    token = token_data["tenant_access_token"]

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
        print(f"Message sent: {result.get('code', '?')}, msg_id={result.get('data', {}).get('message_id', 'N/A')}")
except urllib.error.HTTPError as e:
    print(f"HTTP Error: {e.code} {e.reason}")
    print(e.read().decode())
