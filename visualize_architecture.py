#!/usr/bin/env python3
"""
SuperModel 超模态模型架构可视化程序
====================================
显示每一层神经网络的结构和数据流向

Usage:
    python3 visualize_architecture.py          # 文本模式
    python3 visualize_architecture.py --gui    # GUI图形模式
    python3 visualize_architecture.py --html   # 生成HTML报告
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def visualize_text():
    """纯文本架构可视化"""
    print("=" * 80)
    print("SuperModel 超模态模型架构可视化".center(80))
    print("=" * 80)
    print()
    
    # 1. 系统整体架构
    print("\n📊 一、系统整体架构")
    print("-" * 40)
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SuperModel 系统架构                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        外部环境 (Environment)                        │   │
│   │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────┐   │   │
│   │   │ 视觉   │  │ 听觉   │  │ 触觉   │  │ 力觉   │  │ IMU  │   │   │
│   │   │Vision │  │ Audio  │  │Tactile │  │ Force  │  │      │   │   │
│   │   └───┬───┘  └───┬───┘  └───┬───┘  └───┬───┘  └───┬───┘   │   │
│   │       └──────────┴──────────┴──────────┴──────────┘          │   │
│   └──────────────────────────────┼───────────────────────────────────┘   │
│                                  │                                        │
│   ┌──────────────────────────────┼───────────────────────────────────┐   │
│   │                        感知层 │ (Perception Layer)                 │   │
│   │   ┌─────────────────────────┴─────────────────────────────┐     │   │
│   │   │              传感器管理 (SensorManager)               │     │   │
│   │   └─────────────────────────┬─────────────────────────────┘     │   │
│   │                             │                                   │   │
│   │   ┌─────────┬─────────┬───┴────┬─────────┬─────────┐         │   │
│   │   │Vision   │ Audio   │ Tactile │ Force   │ IMU    │         │   │
│   │   │Encoder  │ Encoder │ Encoder │ Encoder  │Encoder │         │   │
│   │   └─────────┴─────────┴─────────┴─────────┴─────────┘         │   │
│   └──────────────────────────────┼────────────────────────────────────┘   │
│                                 │                                          │
│   ┌────────────────────────────┼────────────────────────────────────┐   │
│   │                     融合层 │ (Fusion Layer)                      │   │
│   │   ┌──────────────────────┴──────────────────────┐             │   │
│   │   │           跨模态融合 (CrossModalFusion)        │             │   │
│   │   │    ┌────────┬────────┬────────┬────────┐    │             │   │
│   │   │    │Early   │Middle  │Late    │Attention│    │             │   │
│   │   │    │Fusion  │Fusion  │Fusion │Fusion  │    │             │   │
│   │   │    └────────┴────────┴────────┴────────┘    │             │   │
│   │   └──────────────────────────────────────────────┘             │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
    """)
    
    # 2. 感知层神经网络结构
    print("\n🧠 二、感知层神经网络结构")
    print("-" * 40)
    print("""
┌─────────────────────────────────────────────────────────────────┐
│                      感知层 (Perception Layer)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   传感器编码器层                          │   │
│  │                                                          │   │
│  │   Vision Encoder      Audio Encoder      Tactile Encoder  │   │
│  │   ┌──────────┐      ┌──────────┐      ┌──────────┐     │   │
│  │   │ Conv2D   │      │  1D-CNN  │      │  1D-CNN  │     │   │
│  │   │ 64ch,3×3 │      │  64ch    │      │  64ch    │     │   │
│  │   │  ReLU    │      │  ReLU    │      │  ReLU    │     │   │
│  │   └────┬─────┘      └────┬─────┘      └────┬─────┘     │   │
│  │        │                   │                   │             │   │
│  │        ▼                   ▼                   ▼             │   │
│  │   ┌──────────┐      ┌──────────┐      ┌──────────┐     │   │
│  │   │ MaxPool  │      │ 1D-CNN  │      │  1D-CNN  │     │   │
│  │   │ 2×2      │      │  128ch   │      │  128ch   │     │   │
│  │   └────┬─────┘      └────┬─────┘      └────┬─────┘     │   │
│  │        │                   │                   │             │   │
│  │        └───────────────────┼───────────────────┘             │   │
│  │                            ▼                                  │   │
│  │                   ┌──────────────┐                          │   │
│  │                   │   特征融合   │                          │   │
│  │                   │ (Concat)   │                          │   │
│  │                   │ [512-dim]  │                          │   │
│  │                   └──────────────┘                          │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
    """)
    
    # 3. 融合层神经网络结构
    print("\n🔗 三、融合层神经网络结构 (CrossModalFusion)")
    print("-" * 40)
    print("""
┌─────────────────────────────────────────────────────────────────┐
│                   跨模态融合网络 (CrossModalFusion)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  输入:                                                              │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                    │
│  │Vision│ │Audio│ │Tact.│ │Force│ │ IMU │                    │
│  │[512] │ │[128]│ │[64] │ │[32] │ │[64] │                    │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘                    │
│     └───────┴───────┴───────┴───────┘                          │
│                         │                                          │
│                         ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Input Projection (Linear)                     │ │
│  │           Vision → [256]  Audio → [256]  ...              │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                            │                                       │
│                            ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Multi-Head Cross-Attention (×4 heads)         │ │
│  │                                                              │ │
│  │    ┌──────────────────────────────────────────────────┐    │ │
│  │    │  Head 1: Vision ↔ Audio (Q, K, V)              │    │ │
│  │    │  Head 2: Vision ↔ Tactile                       │    │ │
│  │    │  Head 3: Force ↔ IMU                            │    │ │
│  │    │  Head 4: All modalities fusion                   │    │ │
│  │    └──────────────────────────────────────────────────┘    │ │
│  │                            │                               │ │
│  │                            ▼                               │ │
│  │                   ┌──────────────┐                       │ │
│  │                   │    Add &    │                       │ │
│  │                   │   Norm      │                       │ │
│  │                   └──────────────┘                       │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                            │                                       │
│                            ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Feed-Forward Network                      │ │
│  │         Linear(256) → ReLU → Dropout → Linear(256)        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                            │                                       │
│                            ▼                                       │
│  输出:                                                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Unified Representation                │   │
│  │    State Rep [256]  │  Policy [128]  │  WorldModel [256] │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
    """)
    
    # 4. 认知层神经网络结构
    print("\n🧩 四、认知层神经网络结构 (World Model + Dreamer)")
    print("-" * 40)
    print("""
┌─────────────────────────────────────────────────────────────────┐
│                      认知层 (Cognition Layer)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    RSSM (循环状态空间模型)                   │   │
│  │                                                          │   │
│  │   obs_embed ──┐                                         │   │
│  │               │                                         │   │
│  │               ├──→ [Det-State] →─┐                      │   │
│  │               │                 │                       │   │
│  │   action ─────┴──→ [RSSM-Cell]──┴──→ [Stoch-State]    │   │
│  │                              │                            │   │
│  │                              └──→ [h_t] (hidden state)   │   │
│  │                                                          │   │
│  │   RSSM Cell内部:                                         │   │
│  │   ┌─────────────────────────────────────────────┐        │   │
│  │   │  Deterministic:  GRU(512)                  │        │   │
│  │   │  Stochastic:    Prior p(z_t|h_{t-1},a_{t-1})       │   │
│  │   │  Posterior:    q(z_t|h_t,a_{t-1})                 │        │   │
│  │   │  Kl Loss:      KL(p || q)                          │        │   │
│  │   └─────────────────────────────────────────────┘        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                   │
│           ┌────────────────┼────────────────┐                  │
│           │                │                │                  │
│           ▼                ▼                ▼                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │Observation  │  │   Reward    │  │   Continue   │          │
│  │  Decoder    │  │  Predictor  │  │  Predictor  │          │
│  │ [512-dim]  │  │  [1-dim]   │  │  [1-dim]   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Dreamer Agent                          │   │
│  │                                                          │   │
│  │   imagination_horizon = 15                                │   │
│  │                                                          │   │
│  │   h_t ──→ Actor(π) ──→ a_t ──→ RSSM ──→ h_{t+1}       │   │
│  │                   │                                      │   │
│  │                   ▼                                      │   │
│  │              Critic(V) ──→ value                        │   │
│  │                                                          │   │
│  │   loss = Σ [γ^t * reward + λ * value_loss]             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
    """)
    
    # 5. 执行层神经网络结构
    print("\n⚙️ 五、执行层神经网络结构")
    print("-" * 40)
    print("""
┌─────────────────────────────────────────────────────────────────┐
│                      执行层 (Execution Layer)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    任务规划 (HTN Planner)                   │   │
│  │    ┌─────────────┐                                       │   │
│  │    │  Root Task  │                                       │   │
│  │    └──────┬──────┘                                       │   │
│  │           │                                               │   │
│  │    ┌──────┴──────┐                                       │   │
│  │    │             │                                       │   │
│  │    ▼             ▼                                       │   │
│  │ ┌──────┐   ┌──────────┐                                │   │
│  │ │Subtask1│   │Subtask2 │                                │   │
│  │ └──────┘   └────┬─────┘                                │   │
│  │                 │                                        │   │
│  │                 ▼                                        │   │
│  │           ┌──────────────┐                             │   │
│  │           │   Primitive   │                             │   │
│  │           │   Actions     │                             │   │
│  │           └──────────────┘                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                   │
│                            ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    运动控制 (Motion Control)                 │   │
│  │                                                          │   │
│  │   Target Pose ──→ ┌──────────────┐ ──→ Wheel Velocities│   │
│  │                   │   MPC        │                       │   │
│  │                   │ (Model Pred) │                       │   │
│  │                   └──────────────┘                       │   │
│  │                            │                               │   │
│  │                            ▼                               │   │
│  │                   ┌──────────────┐                       │   │
│  │                   │   PID        │                       │   │
│  │                   │   Controller │                       │   │
│  │                   └──────────────┘                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    安全监控 (Safety)                        │   │
│  │    Velocity Limit ──→ ┌──────────────┐ ──→ E-Stop     │   │
│  │                       │   Monitor    │                  │   │
│  │    Force Limit ──────→ │              │ ──→ Alert       │   │
│  │                       └──────────────┘                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
    """)
    
    # 6. 数据流总结
    print("\n📈 六、数据流总结")
    print("-" * 40)
    print("""
输入 → 感知 → 融合 → 认知 → 规划 → 控制 → 执行
         │        │        │        │        │        │
         ▼        ▼        ▼        ▼        ▼        ▼
    [传感器]  [编码器] [注意力] [RSSM] [HTN] [MPC] [电机驱动]
    [原始]   [特征]   [统一]  [世界] [任务] [预测] [轮速]
     数据     表示     表示   模型  规划  控制  

维度变化:
  Vision:    [H,W,3]   → 512-dim
  Audio:     [T,1]     → 128-dim  
  Tactile:   [N,1]     → 64-dim
  Force:     [6,1]     → 32-dim
  IMU:       [6,1]     → 64-dim
  
  ─────────────────────────────────────────
  
  Concatenate → 832-dim → Projection → 256-dim
  
  Cross-Attention → 256-dim
  
  RSSM:
    obs_embed [512] + action [7] → h [512] + z [32]
    
  Actor: h [512] → a [7] (policy)
  Critic: h [512] → V [1] (value)
  
  MPC: target + state → control sequence [N×7]
""")
    
    print("\n" + "=" * 80)
    print("SuperModel 超模态模型架构可视化完成".center(80))
    print("=" * 80)


def visualize_mermaid():
    """生成 Mermaid 格式的架构图"""
    print("""
```mermaid
flowchart TB
    subgraph Environment["外部环境"]
        V[视觉 Vision]
        A[听觉 Audio]
        T[触觉 Tactile]
        F[力觉 Force]
        I[IMU]
    end
    
    subgraph Perception["感知层 Perception"]
        VE[Vision Encoder<br/>Conv2D → Pool → FC<br/>512-dim]
        AE[Audio Encoder<br/>1D-CNN → FC<br/>128-dim]
        TE[Tactile Encoder<br/>1D-CNN → FC<br/>64-dim]
        FE[Force Encoder<br/>MLP → FC<br/>32-dim]
        IE[IMU Encoder<br/>MLP → FC<br/>64-dim]
    end
    
    V --> VE
    A --> AE
    T --> TE
    F --> FE
    I --> IE
    
    subgraph Fusion["融合层 Fusion"]
        CMF[Cross-Modal Fusion<br/>Multi-Head Attention<br/>256-dim]
    end
    
    VE & AE & TE & FE & IE --> CMF
    
    subgraph Cognition["认知层 Cognition"]
        RSSM[RSSM Cell<br/>GRU(512) + Stoch(32)]
        WD[World Decoder<br/>obs_recon]
        RP[Reward Predictor]
    end
    
    CMF --> RSSM
    RSSM --> WD
    RSSM --> RP
    
    subgraph Execution["执行层 Execution"]
        HTN[HTN Planner<br/>Task Decomposition]
        MPC[MPC Controller<br/>Model Predictive]
        PID[PID Controller<br/>Joint Control]
        SM[Safety Monitor<br/>Limits + E-Stop]
    end
    
    RSSM --> HTN
    HTN --> MPC
    MPC --> PID
    PID --> SM
    
    SM --> Motor[Motor Driver<br/>ZLAC8015D]
```
""")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SuperModel 架构可视化')
    parser.add_argument('--gui', action='store_true', help='显示GUI图形')
    parser.add_argument('--html', action='store_true', help='生成HTML报告')
    parser.add_argument('--mermaid', action='store_true', help='生成Mermaid图')
    args = parser.parse_args()
    
    if args.mermaid:
        visualize_mermaid()
    elif args.html:
        print("Generating HTML report... (需要浏览器支持)")
        visualize_text()
        print("\n💡 提示: 使用 --gui 选项查看交互式GUI")
    elif args.gui:
        print("💡 启动GUI模式...")
        print("提示: 使用 --text 选项查看纯文本版本")
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
            
            fig, ax = plt.subplots(1, 1, figsize=(20, 16))
            ax.set_xlim(0, 20)
            ax.set_ylim(0, 16)
            ax.axis('off')
            ax.set_title('SuperModel 超模态模型架构', fontsize=20, fontweight='bold', pad=20)
            
            # 定义颜色
            colors = {
                'environment': '#E8F5E9',
                'perception': '#E3F2FD',
                'fusion': '#FFF3E0',
                'cognition': '#F3E5F5',
                'execution': '#FCE4EC',
                'hardware': '#ECEFF1'
            }
            
            def draw_box(ax, x, y, w, h, label, sublabel='', color='#FFFFFF', fontsize=10):
                box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", 
                                    facecolor=color, edgecolor='#333333', linewidth=2)
                ax.add_patch(box)
                if sublabel:
                    ax.text(x + w/2, y + h/2 + 0.2, label, ha='center', va='center', 
                           fontsize=fontsize, fontweight='bold')
                    ax.text(x + w/2, y + h/2 - 0.2, sublabel, ha='center', va='center', 
                           fontsize=fontsize-2, color='#666666')
                else:
                    ax.text(x + w/2, y + h/2, label, ha='center', va='center', 
                           fontsize=fontsize, fontweight='bold')
            
            def draw_arrow(ax, x1, y1, x2, y2, color='#333333'):
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', color=color, lw=2))
            
            # 绘制各层
            # 外部环境
            draw_box(ax, 0.5, 13, 19, 2.5, '外部环境 (Environment)', 'Vision | Audio | Tactile | Force | IMU', colors['environment'])
            
            # 感知层
            draw_box(ax, 0.5, 10, 3.5, 2.5, 'Vision\nEncoder', '512-dim', colors['perception'])
            draw_box(ax, 4.2, 10, 3.5, 2.5, 'Audio\nEncoder', '128-dim', colors['perception'])
            draw_box(ax, 7.9, 10, 3.5, 2.5, 'Tactile\nEncoder', '64-dim', colors['perception'])
            draw_box(ax, 11.6, 10, 3.5, 2.5, 'Force\nEncoder', '32-dim', colors['perception'])
            draw_box(ax, 15.3, 10, 3.5, 2.5, 'IMU\nEncoder', '64-dim', colors['perception'])
            
            # 感知层标签
            draw_box(ax, 0.5, 12.7, 18.3, 0.5, '', '', colors['perception'])
            ax.text(10, 12.95, '感知层 (Perception Layer)', ha='center', va='center', fontsize=12, fontweight='bold')
            
            # 箭头: 环境 → 感知
            for i, x in enumerate([2.25, 5.95, 9.65, 13.35, 17.05]):
                draw_arrow(ax, x, 13, x, 12.7)
            
            # 融合层
            draw_box(ax, 4, 7, 12, 2.5, 'Cross-Modal Fusion', 'Multi-Head Attention | 256-dim', colors['fusion'])
            draw_arrow(ax, 10, 10, 10, 9.5)
            
            # 认知层
            draw_box(ax, 1, 4, 8, 2.5, 'RSSM', 'Recurrent State Space Model\n512 + 32 dim', colors['cognition'])
            draw_box(ax, 11, 4, 8, 2.5, 'Dreamer Agent', 'Imaginary Rollout\nActor-Critic', colors['cognition'])
            
            # 认知层标签
            draw_box(ax, 0.5, 6.7, 19, 0.5, '', '', colors['cognition'])
            ax.text(10, 6.95, '认知层 (Cognition Layer)', ha='center', va='center', fontsize=12, fontweight='bold')
            
            # 箭头: 融合 → 认知
            draw_arrow(ax, 5, 7, 5, 6.7)
            draw_arrow(ax, 15, 7, 15, 6.7)
            
            # 执行层
            draw_box(ax, 0.5, 1.5, 4, 2, 'HTN Planner', 'Task Decompose', colors['execution'])
            draw_box(ax, 5, 1.5, 4, 2, 'MPC', 'Model Predictive', colors['execution'])
            draw_box(ax, 9.5, 1.5, 4, 2, 'PID Control', 'Joint Control', colors['execution'])
            draw_box(ax, 14, 1.5, 5, 2, 'Safety Monitor', 'Limits + E-Stop', colors['execution'])
            
            # 执行层标签
            draw_box(ax, 0.5, 3.7, 18.5, 0.5, '', '', colors['execution'])
            ax.text(10, 3.95, '执行层 (Execution Layer)', ha='center', va='center', fontsize=12, fontweight='bold')
            
            # 箭头: 认知 → 执行
            for x in [2.5, 6.5, 11.5, 16.5]:
                draw_arrow(ax, x, 4, x, 3.7)
            
            # 最终输出
            draw_box(ax, 7, 0.2, 6, 1, 'Motor Driver', 'ZLAC8015D → Wheel Motors', '#ECEFF1')
            draw_arrow(ax, 10, 1.5, 10, 1.2)
            
            plt.tight_layout()
            plt.savefig('/tmp/supermodel_architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
            print("✅ 架构图已保存到 /tmp/supermodel_architecture.png")
            print("💡 使用图像查看器打开查看")
            
        except ImportError as e:
            print(f"❌ GUI模式需要 matplotlib: {e}")
            print("💡 退回到文本模式...")
            visualize_text()
    else:
        visualize_text()
