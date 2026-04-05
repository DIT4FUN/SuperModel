#!/usr/bin/env python3
"""Send Feishu progress report via Feishu Open API"""
import json, urllib.request, urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """SuperModel项目进度更新 (v1.52.0 - 2026-04-05 16:36)：
✅ 本次完成（本次cron任务）：
  - 触觉模块(tactile.py): 完整电子皮肤阵列，含压力分布/温度/接近觉/滑移检测/抓取质量评估/AGV五级规格
  - 力觉模块(force.py): 六维力矩传感器，含Wrench力旋量/接触检测/负载估计/碰撞仿真/AGV五级规格
  - IMU模块(imu.py): 惯性测量，含Madgwick/互补滤波/卡尔曼姿态估计/速度位置积分/AGV五级规格
  - 控制模块(22子模块): motor/motion/trajectory/mpc/impedance/force/imu/tactile/agv/安全监控/避障/规划/ROS2/多AGV/teleop/supervisor/autotune + sensorimotor全部就绪
  - 仿真环境: MuJoCo + PyBullet + Gazebo + Gymnasium + 仓储物流场景
  - 测试用例: sensor_tests.py(97项) + fusion_tests.py(33项) 共130项全部通过 ✅
  - 设计文档: AGV五级规格总表(感知/融合/认知/执行/学习/通信/安全/硬件平台/端到端指标)
  - GitHub: 610cf10 'docs: v1.52.0 添加硬件图片资源 + 完善规格文档'

📊 SuperModel整体状态 (v1.52.0)：
  传感器层: vision + audio + tactile + force + imu + encoders + manager (7模块)
  融合层: cross_modal_fusion + sensor_fusion (互补滤波/EKF/多传感器)
  认知层: scene_understanding + world_model + dreamer_agent + 自监督 + 自主学习
  执行层: 22个控制子模块 (全功能覆盖)
  仿真层: 4种物理引擎 + 多场景
  测试: 30+文件，130项核心测试通过
  文档: MODULE_INDEX + SPEC + AGV_SPEC + AGV_SPEC_QUICKREF + DESIGN + MODULE_INTERFACE(36章节)

✅ 全部任务已完成，SuperModel超模态大模型具身智能大脑核心架构就绪！

🔜 下一步: 真实AGV机器人集成测试、端到端具身智能演示、具身推理能力增强"""


def get_token():
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    payload = json.dumps({"app_id": APP_ID, "app_secret": APP_SECRET}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode()).get("tenant_access_token", "")


def send_message(token: str, chat_id: str, text: str) -> dict:
    url = "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id"
    payload = json.dumps({
        "receive_id": chat_id,
        "msg_type": "text",
        "content": json.dumps({"text": text})
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    })
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


if __name__ == "__main__":
    try:
        token = get_token()
        result = send_message(token, CHAT_ID, MESSAGE)
        if result.get("code") == 0:
            print("✅ Feishu message sent successfully")
        else:
            print(f"❌ Feishu API error: {result}")
    except urllib.error.URLError as e:
        print(f"❌ Network error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
