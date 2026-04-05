#!/usr/bin/env python3
"""
SuperModel MuJoCo AGV仿真演示
============================
使用MuJoCo进行AGV仿真

Usage:
    python3 run_mujoco_demo.py
"""

import sys
sys.path.insert(0, 'src')

from simulation.mujoco_sim import HAS_MUJOCO
import numpy as np

def main():
    print("=" * 60)
    print("SuperModel MuJoCo AGV仿真演示")
    print("=" * 60)
    
    if not HAS_MUJOCO:
        print("❌ MuJoCo 未安装!")
        print("请运行: ./venv/bin/pip install mujoco")
        return
    
    print(f"✅ MuJoCo 已就绪")
    
    # 测试MuJoCo基本功能
    print("\n📦 测试MuJoCo基本功能...")
    import mujoco
    
    # 创建简单的AGV XML
    xml = """
    <mujoco model="agv_simple">
        <compiler angle="radian"/>
        <option timestep="0.002" gravity="0 0 -9.81"/>
        
        <worldbody>
            <!-- 地面 -->
            <geom type="plane" name="ground" size="10 10 0.1" pos="0 0 -0.001" 
                  friction="1 0.005 0.0001" rgba="0.5 0.5 0.5 1"/>
            
            <!-- AGV车体 -->
            <body name="chassis" pos="0 0 0.1">
                <freejoint/>
                <inertial pos="0 0 0" mass="10" diaginertia="0.1 0.1 0.1"/>
                
                <!-- 底盘几何 -->
                <geom type="box" size="0.25 0.25 0.02" rgba="0.2 0.6 0.8 1"/>
                
                <!-- 左轮 -->
                <body name="left_wheel" pos="-0.05 0.22 0" axisangle="1 0 0 90">
                    <joint name="left_wheel_joint" type="hinge" axis="0 0 1" damping="0.5"/>
                    <geom type="cylinder" size="0.08 0.025" rgba="0.1 0.1 0.1 1" friction="1 0.005 0.0001"/>
                </body>
                
                <!-- 右轮 -->
                <body name="right_wheel" pos="-0.05 -0.22 0" axisangle="1 0 0 90">
                    <joint name="right_wheel_joint" type="hinge" axis="0 0 1" damping="0.5"/>
                    <geom type="cylinder" size="0.08 0.025" rgba="0.1 0.1 0.1 1" friction="1 0.005 0.0001"/>
                </body>
            </body>
        </worldbody>
        
        <actuator>
            <motor joint="left_wheel_joint" ctrllimited="true" ctrlrange="-10 10" gear="1"/>
            <motor joint="right_wheel_joint" ctrllimited="true" ctrlrange="-10 10" gear="1"/>
        </actuator>
    </mujoco>
    """
    
    print("   创建模型...")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    
    print(f"   位置自由度(nq): {model.nq}")
    print(f"   速度自由度(nv): {model.nv}")
    print(f"   控制维度(nu): {model.nu}")
    
    # 仿真循环
    print("\n🚀 开始仿真...")
    print("-" * 60)
    
    # 差速驱动仿真
    v_l = 2.0  # 左轮速度
    v_r = 2.0   # 右轮速度
    
    for step in range(500):
        # 设置控制 (左轮, 右轮)
        data.ctrl[0] = v_l
        data.ctrl[1] = v_r
        
        # 仿真一步
        mujoco.mj_step(model, data)
        
        if step % 100 == 0:
            print(f"   Step {step:4d}: pos=({data.qpos[0]:.2f}, {data.qpos[1]:.2f}, {data.qpos[2]:.2f})")
    
    print("-" * 60)
    print("✅ 仿真完成!")
    
    # 最终状态
    print(f"\n📊 最终状态:")
    print(f"   位置: x={data.qpos[0]:.3f}, y={data.qpos[1]:.3f}, z={data.qpos[2]:.3f}")
    print(f"   速度: vx={data.qvel[0]:.3f}, vy={data.qvel[1]:.3f}, vz={data.qvel[2]:.3f}")
    
    # 原地转弯测试
    print("\n🔄 转弯测试 (v_l=1.0, v_r=-1.0)...")
    mujoco.mj_resetData(model, data)
    
    for step in range(200):
        data.ctrl[0] = 1.0
        data.ctrl[1] = -1.0
        mujoco.mj_step(model, data)
    
    # 获取朝向角
    quat = data.qpos[3:7]
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    yaw = np.arctan2(mat[3], mat[0])
    print(f"   转弯后朝向: {np.degrees(yaw):.1f}°")
    
    print("\n✅ MuJoCo AGV仿真测试完成!")


if __name__ == '__main__':
    main()
