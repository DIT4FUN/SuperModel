#!/usr/bin/env python3
"""Send Feishu progress report"""
import json, urllib.request, urllib.error, os

CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"
MESSAGE = """SuperModel项目进度更新：
✅ 已完成：tactile/force/IMU传感器模块、control控制模块、测试用例
✅ AGV五级规格表已添加到docs/SPEC.md
✅ 代码已提交并推送至GitHub (v1.44.0 - 0a0de8b)
📁 新增文件：
  - src/sensors/tactile.py (触觉传感器)
  - src/sensors/force.py (六轴力传感器)
  - src/sensors/imu.py (IMU传感器)
  - tests/sensor_tests.py
  - tests/fusion_tests.py
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
