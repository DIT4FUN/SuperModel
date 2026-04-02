#!/usr/bin/env python3
"""Send Feishu progress report"""
import json, urllib.request, urllib.error, os

CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"
MESSAGE = """SuperModel项目进度更新 (2026-04-02)：
✅ 已完成：
  - sensors/tactile.py (触觉传感器：压阻/触感阵列/压电)
  - sensors/force.py (六维力传感器：ATI mini40)
  - sensors/imu.py (IMU传感器：BMI088/MPU9250 + AHRS融合)
  - control/motor.py (电机控制：DC/BLDC/伺服/步进)
  - control/motion.py (运动控制：差速/麦轮/轨迹规划)
  - control/pid.py (PID控制器：通用/二维/自动整定)
  - control/safety.py (安全监控：限速/边界/碰撞/紧急停止)
  - tests/sensor_tests.py (传感器单元测试)
  - tests/fusion_tests.py (融合算法测试)
✅ README.md 已更新（完整模块文档）
✅ docs/DESIGN.md 已更新（AGV五级规格表详细版）
✅ 代码已提交并推送至GitHub (3e408fc)

📊 AGV五级规格表：
  L1: ≤500kg, ±10mm (磁条/二维码)
  L2: 500-1500kg, ±5mm (激光导航)
  L3: 1500-3000kg, ±3mm (SLAM视觉)
  L4: 3000-5000kg, ±1mm (多传感器融合)
  L5: >5000kg, <±0.5mm (超模态具身智能)

🔜 下一步：完善仿真环境与集成测试"""

webhook = os.environ.get('FEISHU_WEBHOOK_URL', '')
if not webhook:
    for path in [
        '/home/treeman/.openclaw/credentials/feishu-pairing.json',
        '/home/treeman/.openclaw/credentials/feishu-default-allowFrom.json',
    ]:
        try:
            with open(path) as f:
                d = json.load(f)
                for k, v in d.items():
                    if isinstance(v, str) and 'webhook' in k.lower():
                        webhook = v
                        break
                    if isinstance(v, dict):
                        for k2, v2 in v.items():
                            if isinstance(v2, str) and 'webhook' in k2.lower():
                                webhook = v2
                                break
                if webhook:
                    break
        except:
            pass

if not webhook:
    print("ERROR: No Feishu webhook URL found")
    exit(1)

payload = {"msg_type": "text", "content": {"text": MESSAGE}}
data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(
    webhook,
    data=data,
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        if result.get("code") == 0:
            print("✅ Feishu message sent successfully")
        else:
            print(f"❌ Feishu API error: {result}")
except urllib.error.URLError as e:
    print(f"❌ Network error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
