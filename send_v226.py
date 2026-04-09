#!/usr/bin/env python3
"""SuperModel v2.26.0 进度汇报 - GymAGVSpec类型安全升级"""

import sys, requests
sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel')

FEISHU_APP_ID = "cli_9f0f64c4c8d8d00d"
FEISHU_APP_SECRET = "GewMHVpc2vkowEMH_BxGLcGxxnWzCysRYkpPLLBg0fY"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"


def get_token():
    r = requests.post("https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
                      json={"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET}, timeout=10)
    r.raise_for_status()
    return r.json()["tenant_access_token"]


def send(token, content):
    r = requests.post(
        "https://open.feishu.cn/open-apis/im/v1/messages",
        headers={"Authorization": f"Bearer {token}"},
        json={"receive_id": CHAT_ID, "msg_type": "text",
              "content": content},
        params={"receive_id_type": "chat_id"}, timeout=10)
    r.raise_for_status()
    return r.json()


msg = """**SuperModel v2.26.0 进度汇报** 🤖

📅 2026-04-10 02:00 (UTC+8)

---

✅ **本次完成**

1. **GymAGVSpec 类型安全规格类**
   - 新增 `@dataclass GymAGVSpec` (类型安全的AGV五级规格结构)
   - `from_grade('M')` → 返回带类型的规格对象
   - 属性访问: `spec.control_freq_hz`, `spec.payload_kg`, `spec.processor`
   - `get_control_params()` / `get_sensor_params()` 子集提取

2. **新辅助函数**
   - `get_gym_agv_spec(grade)` → GymAGVSpec对象
   - `get_agv_control_params(grade)` → 控制参数字典
   - `compute_agv_reward(grade, error, action, ...)` → RL奖励值
   - `list_gym_agv_specs()` → 所有五级规格

3. **向后兼容**
   - AGV_GYM_GRADE_SPEC 旧字典格式保持不变
   - 所有现有 create_agv_env() 调用无需修改

4. **新增测试** (9项)
   - GymAGVSpec dataclass 测试
   - get_agv_control_params 测试
   - compute_agv_reward 碰撞惩罚测试
   - get_sensor_params / get_control_params 测试
   - 全部 332项 sensor_tests + 73项 fusion_tests 通过

5. **文档更新**
   - 新增 AGV_FIVE_LEVEL_SPEC_GRID.md (v2.0 总表, 200行)
   - 更新 MODULE_INTERFACE.md 附录K (Gymnasium环境接口)

---

📊 **测试规模**
- sensor_tests: **332项** (新增9项)
- fusion_tests: **73项**
- 合计: **405项** 全部通过 ✅

---

🔗 **GitHub**: https://github.com/DIT4FUN/SuperModel
📦 **提交**: `6711ba7` feat: GymAGVSpec类型安全AGV五级规格类

---

🏗️ **项目现状**
- ✅ 基础架构 (sensors/fusion/perception/learning/control/hardware/simulation)
- ✅ 触觉/力觉/IMU传感器模块
- ✅ Gymnasium仿真环境 (AGV五级配置)
- ✅ 控制模块 (16类控制算法)
- ✅ 文档 (15个设计文档)
- 🔄 下一步: 具身智能仿真环境强化 / 边缘部署优化"""

token = get_token()
print(send(token, f'{{"text": "{msg}"}}'))
