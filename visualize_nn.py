#!/usr/bin/env python3
"""
SuperModel 神经网络结构可视化
============================
显示每层神经网络的具体结构

Usage:
    python3 visualize_nn.py              # 文本模式
    python3 visualize_nn.py --matplotlib  # Matplotlib图形模式
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def print_layer(name, input_shape, output_shape, layers_desc):
    """打印神经网络层结构"""
    print(f"\n{'='*70}")
    print(f"📦 {name}")
    print(f"{'='*70}")
    print(f"输入: {input_shape}  →  输出: {output_shape}")
    print("-" * 70)
    for i, layer in enumerate(layers_desc, 1):
        print(f"  {i:2}. {layer}")
    print()


def visualize_text():
    """文本模式神经网络结构可视化"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║         SuperModel 超模态大模型 - 神经网络结构可视化                   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # 1. Vision Encoder
    print_layer(
        "Vision Encoder (视觉编码器)",
        "[B, H, W, 3] (RGB图像)",
        "[B, 512] (特征向量)",
        [
            "Conv2D(3, 64, kernel=7×7, stride=2, padding=3) + BatchNorm + ReLU",
            "MaxPool2D(kernel=3×3, stride=2)",
            "Conv2D(64, 128, kernel=3×3, stride=1, padding=1) + BatchNorm + ReLU",
            "Conv2D(128, 256, kernel=3×3, stride=2, padding=1) + BatchNorm + ReLU",
            "Conv2D(256, 512, kernel=3×3, stride=1, padding=1) + BatchNorm + ReLU",
            "AdaptiveAvgPool2D(output_size=1)",
            "Flatten() + Linear(512, 512) + ReLU + Dropout(0.5)",
        ]
    )
    
    # 2. Audio Encoder
    print_layer(
        "Audio Encoder (音频编码器)",
        "[B, T, 1] (音频波形)",
        "[B, 128] (特征向量)",
        [
            "Conv1D(1, 64, kernel=25, stride=4) + ReLU",
            "Conv1D(64, 128, kernel=25, stride=4) + ReLU",
            "Conv1D(128, 128, kernel=25, stride=4) + ReLU",
            "AdaptiveAvgPool1D(output_size=1)",
            "Flatten() + Linear(128, 128) + ReLU + Dropout(0.3)",
        ]
    )
    
    # 3. Tactile Encoder
    print_layer(
        "Tactile Encoder (触觉编码器)",
        "[B, N_taxels] (压力阵列)",
        "[B, 64] (特征向量)",
        [
            "Linear(N_taxels, 64) + ReLU",
            "Linear(64, 64) + ReLU",
            "LayerNorm(64)",
        ]
    )
    
    # 4. Force Encoder
    print_layer(
        "Force Encoder (力觉编码器)",
        "[B, 6] (Fx, Fy, Fz, Mx, My, Mz)",
        "[B, 32] (特征向量)",
        [
            "Linear(6, 32) + ReLU",
            "Linear(32, 32) + ReLU",
            "LayerNorm(32)",
        ]
    )
    
    # 5. IMU Encoder
    print_layer(
        "IMU Encoder (IMU编码器)",
        "[B, 6] (ax, ay, az, gx, gy, gz)",
        "[B, 64] (特征向量)",
        [
            "Linear(6, 64) + ReLU",
            "Linear(64, 64) + ReLU",
            "LayerNorm(64)",
        ]
    )
    
    # 6. Cross-Modal Fusion
    print_layer(
        "Cross-Modal Fusion (跨模态融合)",
        "[B, 832] (拼接特征)",
        "[B, 256] (统一表示)",
        [
            "Linear(832, 512) + LayerNorm + ReLU + Dropout(0.1)",
            "Multi-Head Self-Attention(num_heads=4, d_model=512)",
            "  ├─ Head 1: Vision ↔ Audio (Q, K, V cross-attention)",
            "  ├─ Head 2: Vision ↔ Tactile",
            "  ├─ Head 3: Force ↔ IMU",
            "  └─ Head 4: Global fusion",
            "Add & LayerNorm",
            "Feed-Forward(512 → 2048 → 512) + Dropout(0.1)",
            "Add & LayerNorm",
            "Linear(512, 256) + LayerNorm + ReLU",
        ]
    )
    
    # 7. RSSM (World Model)
    print_layer(
        "RSSM - Recurrent State Space Model (世界模型)",
        "obs_embed[B,512] + action[B,7]",
        "h[B,512] (det), z[B,32] (stoch)",
        [
            "┌─ Deterministic Path ─────────────────────────────────────────┐",
            "│  GRU(obs_embed[512] + action[7] + prev_h[512], hidden=512) │",
            "│  → h_t = GRU(h_{t-1}, x_t)                               │",
            "└─────────────────────────────────────────────────────────────┘",
            "┌─ Stochastic Path ───────────────────────────────────────────┐",
            "│  Prior:    Linear(h_t, 256) → ReLU → Linear(256, 64)       │",
            "│             → mu, sigma → Sample z ~ N(mu, sigma)        │",
            "│  Posterior: Linear(obs_embed + h_t + action, 256) → ...    │",
            "│             → mu, sigma → Sample z ~ N(mu, sigma)          │",
            "│  KL Loss:   KL(q(z|x) || p(z|h))                          │",
            "└─────────────────────────────────────────────────────────────┘",
            "┌─ Representation ───────────────────────────────────────────┐",
            "│  z_t = Stoch(z_prior, z_posterior) (combination)          │",
            "│  state_t = concat(h_t, z_t) → [512 + 32]                   │",
            "└─────────────────────────────────────────────────────────────┘",
        ]
    )
    
    # 8. Observation Decoder
    print_layer(
        "Observation Decoder (观测解码器)",
        "state[B, 544] (h:512 + z:32)",
        "[B, 512] (重构观测)",
        [
            "Linear(544, 512) + ReLU",
            "Linear(512, 1024) + ReLU",
            "Linear(1024, 2048) + Sigmoid (图像重构)",
            "Reshape → Deconv2D layers → 重建原始图像",
        ]
    )
    
    # 9. Reward Predictor
    print_layer(
        "Reward Predictor (奖励预测器)",
        "state[B, 544]",
        "[B, 1] (奖励值)",
        [
            "Linear(544, 128) + ReLU",
            "Linear(128, 64) + ReLU",
            "Linear(64, 1)  (无激活, 回归奖励)",
        ]
    )
    
    # 10. Dreamer Actor
    print_layer(
        "Dreamer Actor (策略网络)",
        "h[B, 512] (RSSM hidden state)",
        "[B, 7] (动作)",
        [
            "Linear(512, 256) + ReLU",
            "Linear(256, 128) + ReLU",
            "Linear(128, 14) → Split → [mu, sigma]",
            "Tanh() → squashed_normal → action",
            "Log_std: learnable parameter [-2, 1]",
        ]
    )
    
    # 11. Dreamer Critic
    print_layer(
        "Dreamer Critic (价值网络)",
        "h[B, 512]",
        "[B, 1] (状态价值)",
        [
            "Linear(512, 256) + ReLU",
            "Linear(256, 128) + ReLU",
            "Linear(128, 1)  (无激活, 回归价值)",
        ]
    )
    
    # 12. MPC Controller
    print_layer(
        "MPC Controller (模型预测控制)",
        "state + target + horizon[N]",
        "optimal_control[N, 7]",
        [
            "for k in range(horizon):",
            "  ┌─ Dynamics Model ──────────────────────────────────┐",
            "  │  state_pred = RSSM(state, action)                 │",
            "  │  cost = L(state_pred, target) + λ*||action||²   │",
            "  └──────────────────────────────────────────────────┘",
            "endfor",
            "optimizer.minimize(cost) → action_sequence",
            "return action_sequence[0]  (第一个动作为输出)",
        ]
    )
    
    # 13. AGV Kinematics
    print_layer(
        "AGV Kinematics (运动学)",
        "twist[vx, vy, omega] or wheel_vel[L, R]",
        "wheel_vel[L, R] or twist[vx, vy, omega]",
        [
            "┌─ 差速驱动逆运动学 ─────────────────────────────────────┐",
            "│  ω_L = (v - omega * track_width/2) / wheel_radius    │",
            "│  ω_R = (v + omega * track_width/2) / wheel_radius    │",
            "└──────────────────────────────────────────────────────┘",
            "┌─ 差速驱动正运动学 ─────────────────────────────────────┐",
            "│  v = (ω_R + ω_L) * wheel_radius / 2                  │",
            "│  omega = (ω_R - ω_L) * wheel_radius / track_width    │",
            "└──────────────────────────────────────────────────────┘",
        ]
    )
    
    # 14. Safety Monitor
    print_layer(
        "Safety Monitor (安全监控)",
        "velocity + force + position",
        "safe_status + emergency_stop",
        [
            "┌─ Velocity Limit Check ────────────────────────────────┐",
            "│  if v > v_max or |omega| > omega_max:                 │",
            "│    → EMERGENCY_STOP                                  │",
            "│  end                                                 │",
            "└─────────────────────────────────────────────────────┘",
            "┌─ Force Limit Check ─────────────────────────────────┐",
            "│  if |F| > F_max or |tau| > tau_max:                 │",
            "│    → EMERGENCY_STOP                                  │",
            "│  end                                                 │",
            "└─────────────────────────────────────────────────────┘",
            "┌─ Boundary Check ─────────────────────────────────────┐",
            "│  if position outside workspace:                       │",
            "│    → STOP + ALERT                                   │",
            "│  end                                                 │",
            "└─────────────────────────────────────────────────────┘",
        ]
    )
    
    # 数据维度汇总
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                        数据维度汇总表                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  模态         │  原始数据   │  编码后   │  融合后   │  总计       ║
║  ─────────────┼────────────┼──────────┼──────────┼──────────       ║
║  Vision       │  H×W×3    │  512     │  -       │  512         ║
║  Audio        │  T×1      │  128     │  -       │  128         ║
║  Tactile      │  N        │  64      │  -       │  64          ║
║  Force        │  6        │  32      │  -       │  32          ║
║  IMU          │  6        │  64      │  -       │  64          ║
║  ─────────────┼────────────┼──────────┼──────────┼──────────       ║
║  拼接维度     │  -        │  -       │  800     │  800         ║
║  融合后       │  -        │  -       │  256     │  256         ║
║  RSSM状态     │  -        │  -       │  544     │  544         ║
║  动作空间     │  -        │  -       │  -       │  7 (2轮差速) ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


def visualize_matplotlib():
    """Matplotlib图形模式"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
        import numpy as np
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SuperModel Neural Network Architecture', fontsize=16, fontweight='bold')
        
        def draw_network_box(ax, x, y, width, height, color, text):
            box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02",
                                facecolor=color, edgecolor='#333333', linewidth=1.5)
            ax.add_patch(box)
            ax.text(x + width/2, y + height/2, text, ha='center', va='center',
                   fontsize=8, fontweight='bold', wrap=True)
        
        def draw_layer(ax, x, y, width, height, color, text, subtext=''):
            box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.01",
                                facecolor=color, edgecolor='#666666', linewidth=1)
            ax.add_patch(box)
            ax.text(x + width/2, y + height/2, text, ha='center', va='center',
                   fontsize=7, fontweight='bold')
            if subtext:
                ax.text(x + width/2, y + height/2 - 0.15, subtext, ha='center', va='center',
                       fontsize=6, color='#888888')
        
        def draw_arrow(ax, x1, y1, x2, y2):
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))
        
        # 1. Vision Encoder (左上)
        ax = axes[0, 0]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title('Vision Encoder', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        draw_layer(ax, 1, 8, 8, 1.5, '#E3F2FD', 'Conv2D(7×7, 64)\n+ BN + ReLU')
        draw_layer(ax, 1, 6, 8, 1.5, '#E3F2FD', 'MaxPool(3×3) + Conv2D(3×3, 128)')
        draw_layer(ax, 1, 4, 8, 1.5, '#E3F2FD', 'Conv2D(3×3, 256) + BN + ReLU')
        draw_layer(ax, 1, 2, 8, 1.5, '#E3F2FD', 'Conv2D(3×3, 512) + AdaptivePool')
        draw_layer(ax, 3, 0, 4, 1.5, '#BBDEFB', 'Linear(512) → [B, 512]')
        
        for y in [7.5, 5.5, 3.5]:
            draw_arrow(ax, 5, y, 5, y - 0.8)
        draw_arrow(ax, 5, 2, 5, 1.5)
        
        # 2. Cross-Modal Fusion (中上)
        ax = axes[0, 1]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title('Cross-Modal Fusion', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        modalities = ['Vision\n512', 'Audio\n128', 'Tactile\n64', 'Force\n32', 'IMU\n64']
        x_positions = [1, 2.5, 4, 5.5, 7]
        for i, (mod, x) in enumerate(zip(modalities, x_positions)):
            draw_layer(ax, x, 8, 1.5, 1.5, '#FFF3E0', mod)
            draw_arrow(ax, x + 0.75, 8, x + 0.75, 6.5)
        
        draw_layer(ax, 1, 5.5, 8, 1.5, '#FFE0B2', 'Input Projection → 512')
        draw_arrow(ax, 5, 5.5, 5, 4.5)
        
        draw_layer(ax, 1, 3.5, 8, 1.5, '#FFCC80', 'Multi-Head Attention (4 heads)')
        draw_arrow(ax, 5, 3.5, 5, 2.5)
        
        draw_layer(ax, 3, 1.5, 4, 1.5, '#FFB74D', 'Output: [B, 256]')
        draw_arrow(ax, 5, 2.5, 5, 3)
        
        # 3. RSSM (右上)
        ax = axes[0, 2]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title('RSSM (World Model)', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        draw_layer(ax, 0.5, 7.5, 4, 2, '#F3E5F5', 'obs_embed\n[B, 512]')
        draw_layer(ax, 5.5, 7.5, 4, 2, '#F3E5F5', 'action\n[B, 7]')
        
        draw_arrow(ax, 2.5, 7.5, 2.5, 6)
        draw_arrow(ax, 7.5, 7.5, 7.5, 6)
        
        draw_layer(ax, 1, 4.5, 8, 2, '#E1BEE7', 'GRU(519 → 512)\nDeterministic Path')
        draw_arrow(ax, 5, 4.5, 5, 3.5)
        
        draw_layer(ax, 1, 2, 4, 1.5, '#CE93D8', 'Prior Network\n[B, 32]')
        draw_layer(ax, 5, 2, 4, 1.5, '#CE93D8', 'Posterior Network\n[B, 32]')
        draw_arrow(ax, 5, 3.5, 3, 3.5)
        draw_arrow(ax, 5, 3.5, 7, 3.5)
        
        draw_layer(ax, 2.5, 0, 5, 1.5, '#BA68C8', 'KL Loss + State: [B, 544]')
        draw_arrow(ax, 3, 2, 5, 1.5)
        draw_arrow(ax, 7, 2, 5, 1.5)
        
        # 4. Dreamer Agent (左下)
        ax = axes[1, 0]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title('Dreamer Agent', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        draw_layer(ax, 1, 8, 8, 1.5, '#F3E5F5', 'RSSM State [B, 544]')
        draw_arrow(ax, 5, 8, 5, 7)
        
        draw_layer(ax, 1, 5.5, 8, 2, '#E1BEE7', 'Imaginary Rollout\n(15 steps in latent space)')
        draw_arrow(ax, 5, 5.5, 5, 4.5)
        
        draw_layer(ax, 1, 2.5, 4, 2, '#CE93D8', 'Actor π(a|s)\n[B, 7]')
        draw_layer(ax, 5, 2.5, 4, 2, '#CE93D8', 'Critic V(s)\n[B, 1]')
        draw_arrow(ax, 5, 4.5, 3, 4.5)
        draw_arrow(ax, 5, 4.5, 7, 4.5)
        
        draw_layer(ax, 3, 0.5, 4, 1.5, '#BA68C8', 'Policy Gradient ∇J(π)')
        
        # 5. Motor Control (中下)
        ax = axes[1, 1]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title('AGV Motor Control', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        draw_layer(ax, 1, 7.5, 3.5, 2, '#FCE4EC', 'Target\nvx, vy, ω')
        draw_layer(ax, 5.5, 7.5, 3.5, 2, '#FCE4EC', 'Current State\nposition, velocity')
        
        draw_arrow(ax, 2.75, 7.5, 2.75, 6)
        draw_arrow(ax, 7.25, 7.5, 7.25, 6)
        
        draw_layer(ax, 1, 4.5, 8, 2, '#F8BBD9', 'Inverse Kinematics\nwheel_vel = f(twist)')
        draw_arrow(ax, 2.75, 4.5, 2.75, 3.5)
        draw_arrow(ax, 7.25, 4.5, 7.25, 3.5)
        
        draw_layer(ax, 3, 2, 4, 2, '#F48FB1', 'ω_L, ω_R\nLeft/Right Wheel')
        draw_arrow(ax, 5, 3.5, 5, 4)
        
        # 6. Safety Monitor (右下)
        ax = axes[1, 2]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title('Safety Monitor', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        draw_layer(ax, 1, 7.5, 8, 2, '#FFEBEE', 'Input: velocity, force, position')
        draw_arrow(ax, 5, 7.5, 5, 6.5)
        
        draw_layer(ax, 1, 5, 2.5, 2, '#FFCDD2', 'V-Limit')
        draw_layer(ax, 3.75, 5, 2.5, 2, '#FFCDD2', 'F-Limit')
        draw_layer(ax, 6.5, 5, 2.5, 2, '#FFCDD2', 'Boundary')
        
        draw_arrow(ax, 5, 6.5, 2.75, 7)
        draw_arrow(ax, 5, 6.5, 5, 7)
        draw_arrow(ax, 5, 6.5, 7.75, 7)
        
        draw_layer(ax, 2.5, 2.5, 5, 2, '#EF9A9A', 'Safety Logic (AND)')
        draw_arrow(ax, 2.75, 5, 5, 4.5)
        draw_arrow(ax, 5, 5, 5, 4.5)
        draw_arrow(ax, 7.75, 5, 5, 4.5)
        
        draw_layer(ax, 3, 0.5, 4, 1.5, '#E57373', 'E-STOP if unsafe')
        
        plt.tight_layout()
        plt.savefig('/tmp/supermodel_nn_structure.png', dpi=150, bbox_inches='tight', facecolor='white')
        print("✅ 神经网络结构图已保存到 /tmp/supermodel_nn_structure.png")
        plt.show()
        
    except ImportError as e:
        print(f"❌ 需要安装 matplotlib: pip install matplotlib")
        visualize_text()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SuperModel 神经网络可视化')
    parser.add_argument('--matplotlib', action='store_true', help='使用Matplotlib图形模式')
    args = parser.parse_args()
    
    if args.matplotlib:
        visualize_matplotlib()
    else:
        visualize_text()
