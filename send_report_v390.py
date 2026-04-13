#!/usr/bin/env python3
"""v3.9.0 飞书进度汇报"""
import json
import urllib.request
import urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """📊 SuperModel v3.9.0 开发进度汇报 (2026-04-14 02:15)

✅ 本次任务完成工作：

1. 具身Pipeline状态持久化与恢复 (src/embodied/embodied_pipeline.py, +180行):
   • save_state(): 保存完整Pipeline状态 (配置/任务队列/已完成任务)
   • restore_state(): 从保存状态恢复任务队列和已完成记录
   • export_checkpoint(): 导出检查点到JSON文件
   • import_checkpoint(): 从文件重建并恢复Pipeline (classmethod)
   • reset_health(): 重置错误状态恢复到READY
   • get_health_report(): 完整健康报告 (模块状态/任务统计/成功率/平均耗时)
   • 修复 scene_type 大小写不一致 bug (__init__ 现自动uppercase)

2. 完整集成测试套件 (tests/embodiment/test_embodied_pipeline_full.py, ~420行, 50项):
   • Pipeline基础初始化: 全AGV等级(S/M/L/XL/XXL)、全场景类型(warehouse/hospital/factory/restaurant/outdoor)
   • 生命周期: start/stop/pause/resume 状态转换
   • 完整组合矩阵: 5等级×5场景=25种组合的启动/停止/任务提交
   • 任务队列: 单/多任务提交、优先级、自动ID生成
   • 状态持久化: 保存/恢复/导出/导入检查点、五级等级往返
   • 健康报告: 结构/模块可用性/任务统计/平均耗时/成功率
   • 行为树集成: 全等级规划器初始化/能力检查/任务注册
   • 技能注册表: 单例/统计/状态转换/场景分类/类别分类
   • Pipeline配置: 默认值/转字典/模块开关/传感器开关
   • 并发安全: 多线程任务提交/并发状态保存
   • 50项全部通过 ✅

✅ 测试状态: 975项具身测试 + 46项顶层测试 = 1021项pytest 100%通过

✅ GitHub: 已提交 984f1e1 (v3.9.0)

📊 整体研发进度: ~96%
已完成: 基础架构/视觉/听觉/触觉/力觉/IMU/跨模态融合/核心目标/自主学习/控制模块/五级AGV规格/仿真环境/测试用例/协同SLAM/HIL测试/具身Pipeline(含状态持久化)/技能注册表/记忆系统/场景智能/联邦学习/具身控制流程文档/部署管理
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
