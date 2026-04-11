#!/usr/bin/env python3
"""
real_agv_deploy.py - 真实AGV机器人完整部署示例
SuperModel 超模态大模型具身智能系统

运行方式:
    python examples/real_agv_deploy.py --grade M --can can0 --lidar-port /dev/ttyUSB0 --imu-port /dev/ttyUSB1
"""

import argparse
import time
import logging
from typing import Dict, Any

from src.embodied.real_agv_interface import RealAGVController, AGVHardwareConfig
from src.embodied.deployment import (
    DeploymentConfig,
    DeploymentValidator,
    DeploymentManager,
    create_deployment_manager,
)
from src.embodied.behavior_tree import AGVTaskPlanner, EmbodiedTask
from src.core.core_brain import CoreBrain

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='SuperModel Real AGV Deployment')
    parser.add_argument('--grade', type=str, default='M', choices=['S', 'M', 'L', 'XL', 'XXL'],
                      help='AGV grade (default: M)')
    parser.add_argument('--can', type=str, default='can0',
                      help='CAN interface (default: can0)')
    parser.add_argument('--can-baudrate', type=int, default=500000,
                      help='CAN baudrate (default: 500000)')
    parser.add_argument('--lidar-port', type=str, default='/dev/ttyUSB0',
                      help='Lidar serial port')
    parser.add_argument('--lidar-baudrate', type=int, default=921600,
                      help='Lidar baudrate')
    parser.add_argument('--imu-port', type=str, default='/dev/ttyUSB1',
                      help='IMU serial port')
    parser.add_argument('--imu-baudrate', type=int, default=115200,
                      help='IMU baudrate')
    parser.add_argument('--control-frequency', type=float, default=50.0,
                      help='Control loop frequency (Hz)')
    parser.add_argument('--enable-tactile', action='store_true', default=True,
                      help='Enable tactile sensor')
    parser.add_argument('--enable-force', action='store_true', default=True,
                      help='Enable force sensor')
    parser.add_argument('--dry-run', action='store_true',
                      help='Dry run without actual hardware connection')
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Starting SuperModel deployment for {args.grade} AGV")

    # 1. 创建硬件配置
    hw_config = AGVHardwareConfig.from_grade(args.grade)
    hw_config.can_interface = args.can
    hw_config.can_baudrate = args.can_baudrate
    hw_config.lidar_port = args.lidar_port
    hw_config.lidar_baudrate = args.lidar_baudrate
    hw_config.imu_port = args.imu_port
    hw_config.imu_baudrate = args.imu_baudrate
    hw_config.control_frequency = args.control_frequency

    # 2. 创建部署配置
    deploy_config = DeploymentConfig(
        grade=args.grade,
        can_channel=args.can,
        can_baudrate=args.can_baudrate,
        lidar_port=args.lidar_port,
        lidar_baudrate=args.lidar_baudrate,
        imu_port=args.imu_port,
        imu_baudrate=args.imu_baudrate,
        enable_tactile=args.enable_tactile,
        enable_force=args.enable_force,
        control_frequency=args.control_frequency,
        enable_health_monitoring=True,
        enable_emergency_stop=True,
    )

    # 3. 验证配置
    validator = DeploymentValidator()
    validation = validator.validate_config(deploy_config)

    if not validation.is_healthy():
        logger.error(f"Deployment validation failed: {validation.message}")
        return 1

    logger.info("Deployment validation passed")

    if args.dry_run:
        logger.info("Dry run complete, exiting")
        return 0

    # 4. 创建部署管理器并部署
    try:
        manager = create_deployment_manager(deploy_config)
        success = manager.deploy()

        if not success:
            logger.error("Deployment failed")
            return 1

        logger.info("Deployment successful, entering main control loop")

        # 5. 主循环
        interval = 1.0 / args.control_frequency
        iteration = 0

        while manager.state.is_running():
            start_time = time.time()

            # 单步执行
            manager.step()

            # 获取健康状态
            if iteration % 50 == 0:  # 每一秒报告一次
                health = manager.get_health_summary()
                logger.debug(f"Health: {health}")

                if health.state == 'warning':
                    logger.warning(f"Health warning: {health}")
                elif health.state == 'critical':
                    logger.critical(f"Health critical: {health}")

            iteration += 1

            # 保持频率
            elapsed = time.time() - start_time
            if elapsed < interval:
                time.sleep(interval - elapsed)

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down")
    except Exception as e:
        logger.exception(f"Error during execution: {e}")
    finally:
        if 'manager' in locals():
            manager.shutdown()
            logger.info("Shutdown complete")

    return 0


if __name__ == '__main__':
    exit(main())
