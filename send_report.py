#!/usr/bin/env python3
"""Send Feishu progress report via Feishu Open API"""
import json, urllib.request, urllib.error

APP_ID = "cli_a94ec91dc1f8dcd5"
APP_SECRET = "Htb0eWcTokzIMdpiLaK6Aht0XnNetp7S"
CHAT_ID = "oc_930bbab59ae0857f8f4781724990fe23"

MESSAGE = """SuperModel项目进度更新 (v1.50.0 - 2026-04-03 09:08)：
✅ 本次完成：
  - 新增 PyBullet 物理引擎仿真模块 (src/simulation/pybullet_sim.py, 1107行)
    * PyBulletSimulator 仿真器，支持差速驱动AGV仿真
    * 动态 URDF 生成 (generate_agv_urdf)，支持 S/M/L/XL/XXL 五级规格
    * AGV 五级物理参数: 质量/轮径/轮距/惯量等自动匹配
    * 传感器模拟: IMU、里程计、接触力、相机
    * 障碍物加载: Box/Cylinder/Sphere
    * 修复 str.format() 对 {body_height/2+wheel_radius-0.01} 的解析错误
    * PyBulletGUI 枚举，支持 DIRECT/GUI/EGL 等模式
  - 新增 PyBullet 测试套件 (tests/pybullet_sim_tests.py, 699行)
    * PyBulletConfig 单元测试 (7项)
    * URDF 生成测试 (4项): 全等级/车轮/传感器/输出路径
    * PyBulletSimulator 基础测试 (13项): 初始化/加载/步进/控制/里程计
    * AGV 五级规格测试 (3项)
    * 鲁棒性测试 (3项): 多实例/dt精度/传感器噪声)
  - 更新 src/simulation/__init__.py 导出 PyBullet 模块
  - 修复 tests/pybullet_sim_tests.py 的 UnboundLocalError (HAS_PYBULLET 作用域问题)
  - 修复 mock_pybullet fixture: patch HAS_PYBULLET/p/pybullet_data 三处引用
  - 版本号: v1.49.0 → v1.50.0
  - GitHub最新提交: 待推送

📊 模块状态总览 (v1.50.0 - 1277项测试通过):
  传感器(5类): vision/audio/tactile/force/imu ✅ + encoders/manager
  控制(21子模块): motor / motion / trajectory / mpc / impedance / force / imu / tactile / agv / 安全监控 / 避障 / 规划 / ROS2 / 多AGV / teleop / supervisor / autotune
  融合: 跨模态Transformer / 互补滤波 / EKF / 多传感器融合
  学习: Dreamer / 世界模型 / 自监督 / 自主学习框架
  仿真: MuJoCo / Gazebo / Gymnasium / PyBullet / 仓储物流场景
  文档: SPEC.md / MODULE_INDEX.md / DESIGN.md / AGV_SPEC_QUICKREF.md / MODULE_INTERFACE.md(5288行) / REAL_ROBOT_INTEGRATION.md
  AGV五级规格: 触觉/力觉/IMU/感知/融合/认知/控制(Supervisor五级)/学习/通信/安全 + 快速选型指南

🔜 下一步: PyBullet 真实AGV运动仿真、Gazebo-ROS2集成完善、AGV五级运动控制优化"""


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
