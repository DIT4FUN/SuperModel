#!/usr/bin/env python3
"""
SuperModel 五级AGV规格演示脚本
=============================
演示: 如何加载不同等级AGV配置并运行基本测试
作者: SuperModel Team
版本: v2.64.0
日期: 2026-04-10
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入SuperModel模块
from src.sensors.tactile import get_tactile_spec, AGV_TACTILE_GRADES
from src.sensors.force import get_force_spec, AGV_FORCE_GRADES
from src.sensors.imu import get_imu_spec, AGV_IMU_GRADES

def load_demo_config():
    """加载演示配置"""
    config_path = Path(__file__).parent.parent / 'data' / 'demo' / 'all_grades_config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def demo_grade(grade_name, grade_config):
    """演示单个等级配置"""
    print(f"\n{'='*60}")
    print(f"等级: {grade_name} - {grade_config['name']}")
    print(f"负载: {grade_config['payload_kg']} kg")
    print(f"最大速度: {grade_config['max_speed_mps']} m/s")
    print(f"尺寸: {grade_config['dimensions']['length_mm']}×{grade_config['dimensions']['width_mm']}×{grade_config['dimensions']['height_mm']} mm")
    print('-'*60)
    
    # 获取规格验证
    try:
        tactile_spec = get_tactile_spec(grade_name)
        force_spec = get_force_spec(grade_name)
        imu_spec = get_imu_spec(grade_name)
        
        rows_t, cols_t = tactile_spec['array']
        print(f"✅ 触觉规格验证: {rows_t}×{cols_t} @ {tactile_spec['freq_hz']}Hz, {tactile_spec['res']}bit")
        print(f"✅ 力觉规格验证: {force_spec['axes']}轴 @ {force_spec['sampling_hz']}Hz")
        print(f"✅ IMU规格验证: {imu_spec['type']} @ {imu_spec['sample_hz']}Hz")
        
        # 验证规格是否与配置一致
        expected_freq = grade_config['sensors']['tactile']['rate_hz']
        if tactile_spec['freq_hz'] == expected_freq:
            print(f"📋 规格匹配: ✓ 触觉频率 {tactile_spec['freq_hz']}Hz")
        else:
            print(f"⚠️  规格不匹配: 预期 {expected_freq}Hz, 实际 {tactile_spec['freq_hz']}Hz")
        
        return True
    except Exception as e:
        print(f"❌ 规格验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("="*60)
    print("SuperModel AGV五级规格演示")
    print("="*60)
    
    config = load_demo_config()
    print(f"\n加载配置成功: {config['description']}")
    print(f"生成时间: {config['generated_at']}")
    print(f"包含等级: {list(config['grades'].keys())}")
    
    results = {}
    for grade_name, grade_config in config['grades'].items():
        results[grade_name] = demo_grade(grade_name, grade_config)
    
    print(f"\n{'='*60}")
    print("测试结果汇总:")
    print(f"{'='*60}")
    for grade_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{grade_name}: {status}")
    
    total_success = sum(results.values())
    total = len(results)
    print(f"\n总计: {total_success}/{total} 等级测试通过")
    
    if total_success == total:
        print("\n🎉 所有等级演示成功！SuperModel工作正常。")
    else:
        print("\n⚠️  部分等级测试失败，请检查配置。")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
