#!/usr/bin/env python3
"""SuperModel v2.14.0 进度汇报"""

import sys
sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel')

FEISHU_APP_ID = "cli_9f0f64c4c8d8d00d"
FEISHU_APP_SECRET = "GewMHVpc2vkowEMH_BxGLcGxxnWzCysRYkpPLLBg0fY"
FEISHU_BOT_NAME = "SuperModel超模态大模型"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

def get_tenant_access_token():
    import requests
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    headers = {"Content-Type": "application/json"}
    data = {"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET}
    resp = requests.post(url, json=data, timeout=10)
    resp.raise_for_status()
    return resp.json()["tenant_access_token"]

def send_message(token, content):
    import requests
    url = "https://open.feishu.cn/open-apis/im/v1/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "receive_id": CHAT_ID,
        "msg_type": "text",
        "content": content
    }
    params = {"receive_id_type": "open_id"}
    resp = requests.post(url, headers=headers, json=payload, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()

def main():
    token = get_tenant_access_token()
    
    message = """📈 **SuperModel v2.14.0 研发进度汇报**
🕐 2026-04-09 21:37 (UTC+8)

**✅ 本次新增内容**

**1. 具身控制新模块 (embodied_control.py)**
• `SurfaceFollowingController`: 触觉引导表面跟踪控制器
  - 4种控制模式: 恒力/导纳/阻抗/自适应
  - 压力梯度 → 表面法向估计
  - 支持: 表面擦拭/打磨/抛光/扫描
• `AssemblyController`: 精密装配控制器
  - 6阶段状态机: IDLE→APPROACH→SEARCH→INSERT→SEAT→VERIFY
  - 支持: peg-in-hole/螺纹连接/卡扣装配
  - 螺旋/光栅/随机三种搜索模式
  - 卡阻检测+偏斜检测+插入失败检测

**2. SPEC.md 文档更新**
• 新增第16章: 表面跟踪+装配控制器接口详细设计
• 新增第17章: AGV五级具身控制完整规格表
  - 具身感知规格 (触觉/力觉/IMU五级对比)
  - 具身控制规格 (力控模式/响应时间/任务类型)
  - 具身任务执行规格 (抓取/插入/打磨/MPC)
  - 健康监控与降级策略五级对比

**3. 测试覆盖扩展**
• 新增 SurfaceFollowingController 测试 14项
• 新增 AssemblyController 测试 20项
• embodied_control_tests.py: 76项测试全通过

**📊 测试统计**
```
pytest tests/ — 1993 passed, 16 skipped
```

**🔗 GitHub**
```
commit: 33035c5
branch: main
```

**📋 待完成任务**
• 仿真环境完善 (PyBullet/Gazebo深度集成)
• 硬件驱动层完善
• 端到端具身任务演示

---
🦞 SuperModel 具身智能大脑 | v2.14.0"""

    result = send_message(token, message)
    print(f"消息发送成功: {result}")

if __name__ == "__main__":
    main()
