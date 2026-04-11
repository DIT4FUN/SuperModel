"""
Test cases for AGV Hardware Interface
"""

import pytest
import time
import math
import sys
sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel')
from hardware.agv_interface import AGVInterfaceFactory, AGVType, AGVStatus


def test_simulated_agv_basic():
    """测试仿真AGV基本功能"""
    agv = AGVInterfaceFactory.create("sim", agv_type=AGVType.DIFFERENTIAL, noise_level=0.0)
    
    # 连接测试
    assert agv.connect() == True
    assert agv.is_connected() == True
    assert agv.get_status() == AGVStatus.IDLE
    
    # 速度设置测试
    assert agv.set_velocity(1.0, 0.5) == True
    time.sleep(0.1)
    agv.update_status()
    
    v, omega = agv.get_velocity()
    assert abs(v - 1.0) < 0.1
    assert abs(omega - 0.5) < 0.1
    
    # 运动测试
    initial_x, initial_y, initial_theta = agv.get_pose()
    time.sleep(0.5)
    agv.update_status()
    new_x, new_y, new_theta = agv.get_pose()
    
    assert new_x > initial_x  # 向前移动
    assert new_theta > initial_theta  # 左转
    
    # 急停测试
    assert agv.emergency_stop() == True
    assert agv.get_status() == AGVStatus.EMERGENCY_STOP
    v, omega = agv.get_velocity()
    assert abs(v) < 0.01
    assert abs(omega) < 0.01
    
    # 错误重置测试
    assert agv.reset_error() == True
    assert agv.get_status() == AGVStatus.IDLE
    
    # 断开连接
    agv.disconnect()
    assert agv.is_connected() == False


def test_simulated_agv_movement():
    """测试仿真AGV运动精度"""
    agv = AGVInterfaceFactory.create("sim", agv_type=AGVType.DIFFERENTIAL, noise_level=0.0, friction_coeff=0.0)
    agv.connect()
    
    # 直线移动1米
    agv.set_velocity(1.0, 0.0)
    start_time = time.time()
    while time.time() - start_time < 1.0:
        agv.update_status()
        time.sleep(0.01)
    
    agv.set_velocity(0.0, 0.0)
    time.sleep(0.2)
    agv.update_status()
    
    x, y, theta = agv.get_pose()
    assert abs(x - 1.0) < 0.05  # 误差小于5cm
    assert abs(y) < 0.02
    assert abs(theta) < 0.02
    
    # 原地旋转90度
    agv.set_velocity(0.0, math.pi/2)  # 90度/秒
    start_time = time.time()
    while time.time() - start_time < 1.0:
        agv.update_status()
        time.sleep(0.01)
    
    agv.set_velocity(0.0, 0.0)
    time.sleep(0.2)
    agv.update_status()
    
    x, y, theta = agv.get_pose()
    assert abs(theta - math.pi/2) < 0.05  # 误差小于3度
    
    agv.disconnect()


def test_agv_factory():
    """测试AGV工厂类"""
    # 仿真接口
    sim_agv = AGVInterfaceFactory.create("sim")
    assert sim_agv.agv_type == AGVType.DIFFERENTIAL
    
    # CAN接口
    can_agv = AGVInterfaceFactory.create("can", channel="can1", bitrate=250000)
    assert can_agv.channel == "can1"
    assert can_agv.bitrate == 250000
    
    # Modbus接口
    modbus_agv = AGVInterfaceFactory.create("modbus", host="192.168.0.10", port=503)
    assert modbus_agv.host == "192.168.0.10"
    assert modbus_agv.port == 503


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
