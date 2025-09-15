#!/usr/bin/env python3
"""
生命游戏SFT训练脚本
训练Asterisk算子学习生命游戏规则
"""

import torch
import numpy as np
from life_game_solver import LifeGameSolver

def main():
    print("=== 开始生命游戏SFT训练 ===")

    # 创建求解器
    solver = LifeGameSolver(grid_size=64, hidden_dim=64)

    # 运行SFT训练（单步演化监督）
    print("生成100个生命游戏训练样本，进行单步演化监督训练...")
    accuracy = solver.train_sft(n_samples=100, epochs=1)

    # 图灵机模拟实验
    print("\n=== 最终图灵机模拟实验 ===")
    turing_results = solver.test_turing_machine_simulation(n_tests=50, max_steps=15)

    print(f"\n=== SFT训练完成 ===")
    print(f"单步演化准确率: {accuracy:.4f}")
    print(f"电路模拟准确率: {turing_results['circuit_accuracy']:.4f}")
    print(f"计算任务完成率: {turing_results['computation_success']:.4f}")

    # 保存训练好的模型
    torch.save({
        'model_state_dict': solver.asterisk.state_dict(),
        'grid_size': solver.grid_size,
        'hidden_dim': solver.hidden_dim,
        'circuit_accuracy': turing_results['circuit_accuracy'],
        'computation_success': turing_results['computation_success']
    }, 'life_game_sft_model.pth')

    print("模型已保存到 life_game_sft_model.pth")
    print(f"图灵机模拟准确率: {turing_results['circuit_accuracy']:.4f}")

if __name__ == "__main__":
    main()