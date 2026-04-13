#!/usr/bin/env python3
"""v3.9.1 飞书进度汇报"""
import json
import urllib.request
import urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """📊 SuperModel v3.9.1 开发进度汇报 (2026-04-14 04:10)

✅ 本次任务完成工作：

1. 元认知模块 (src/core/meta_cognition.py, 1105行):
   • CognitiveLoadTracker: 感知/推理/动作三维认知负荷追踪
   • AttentionManager: 注意力资源管理 (FOCUSED/DIVIDED/SUSTAINED/VIGILANT/FATIGUED/DEPLETED)
   • UncertaintyTracker: 贝叶斯推理不确定性量化
   • BiasDetector: 认知偏差检测 (确认/锚定/可得性/过度自信/近因/首因/赌徒谬误)
   • ConfidenceEvaluator: 多维度决策信心评估
   • SelfEfficacyMonitor: 基于成功率的自我效能动态监控
   • MetaCognitionEngine: 完整元认知引擎 (evaluate_situation → 干预建议)
   • AGV五级规格适配 (S:基础监控 → XXL:完整元认知 + 跨任务迁移)
   • src/core/__init__.py 完整导出 MetaCognitionEngine 等14个公共API

2. 元认知测试套件 (tests/test_meta_cognition.py, 881行, 84项测试):
   • CognitiveLoadTracker / AttentionManager / UncertaintyTracker 基础功能
   • BiasDetector / ConfidenceEvaluator / SelfEfficacyMonitor 功能测试
   • MetaCognitionEngine 集成测试 (AGV五级参数化 × 全子模块)
   • 全部84项通过 ✅

3. E2E具身场景集成测试 (tests/embodiment/test_embodied_e2e_scenarios.py, 649行, 71项测试):
   • EmbodiedPipeline × SceneIntelligence × BehaviorTree 完整集成
   • 场景化任务执行 (仓库/医院/工厂/餐厅/户外) × AGV五级规格矩阵
   • 状态持久化与恢复 × 行为树具身任务规划
   • 全部71项通过 ✅

4. 医疗场景测试 (tests/embodiment/test_healthcare_scene.py, 569行, 52项测试):
   • HealthcareSceneController / PatientCallHandler / MedicationDeliveryPlanner
   • InfectionControlMonitor / SpecimenTransportManager
   • AGV五级医疗场景适配测试
   • 全部52项通过 ✅

5. 工业场景测试 (tests/embodiment/test_industrial_scene.py, 567行, 57项测试):
   • ProductionLineController / QualityInspectionStation / PredictiveMaintenanceMonitor
   • ToolManagementSystem / SafetyMonitoringSystem / MaterialFlowCoordinator
   • AGV五级工业场景适配测试
   • 全部57项通过 ✅

✅ 测试状态: 264项新测试全部通过，总测试数约1235项pytest 100%通过

✅ GitHub: 已提交 84a3c11 (v3.9.1)

📊 整体研发进度: ~95%
已完成: 基础架构/视觉/听觉/触觉/力觉/IMU/跨模态融合/核心目标/自主学习/控制模块/五级AGV规格/仿真环境/测试用例/协同SLAM/HIL测试/具身Pipeline(含状态持久化)/技能注册表/记忆系统/场景智能/联邦学习/具身控制流程文档/部署管理/元认知模块
待推进: 视觉-语言-动作端到端/超模态大模型集成"""

def send_message(content):
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    payload = {"app_id": APP_ID, "app_secret": APP_SECRET}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            token_data = json.loads(resp.read().decode("utf-8"))
            token = token_data.get("tenant_access_token", "")
    except Exception as e:
        print(f"获取token失败: {e}")
        return False
    if not token:
        print("未获取到有效token")
        return False

    msg_url = "https://open.feishu.cn/open-apis/im/v1/messages"
    msg_payload = {
        "receive_id": CHAT_ID,
        "msg_type": "text",
        "content": json.dumps({"text": content})
    }
    msg_data = json.dumps(msg_payload).encode("utf-8")
    msg_req = urllib.request.Request(
        msg_url,
        data=msg_data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
    )
    try:
        with urllib.request.urlopen(msg_req, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            print(f"发送结果: {result.get('code', -1)} - {result.get('msg', 'unknown')}")
            return result.get('code') == 0
    except Exception as e:
        print(f"发送消息失败: {e}")
        return False

if __name__ == "__main__":
    success = send_message(MESSAGE)
    exit(0 if success else 1)
