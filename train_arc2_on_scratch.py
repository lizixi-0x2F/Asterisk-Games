#!/usr/bin/env python3
"""
ARC2 ASPP 直接图推理训练脚本
基于ASPP算子的知识图谱推理训练（无蒸馏）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import argparse
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from aspp_model.aspp_operator import ASPPOperator
from aspp_model.ffn_update import FFNUpdate


class KnowledgeEmbeddingLayer(nn.Module):
    """知识嵌入层：将token映射为图节点特征（单层嵌入）"""

    def __init__(self, vocab_size=12, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 单层token嵌入 (0-9: 颜色, 10: 分隔符, 11: 填充符)
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)

    def forward(self, tokens, positions, roles, relations):
        """
        将token转换为图节点特征

        Args:
            tokens: token序列 [seq_len]
            positions: 位置索引 [seq_len] (不再使用)
            roles: 语义角色 [seq_len] (不再使用)
            relations: 关系类型 [seq_len] (不再使用)

        Returns:
            节点特征 [seq_len, hidden_dim]
        """
        # 单层token嵌入
        node_features = self.token_embedding(tokens)

        return node_features


def build_knowledge_graph(nodes, grid_shapes):
    """
    构建知识图谱关系

    Args:
        nodes: 节点特征 [seq_len, hidden_dim]
        grid_shapes: 各网格的形状信息

    Returns:
        边索引列表 [(source, target), ...]
    """
    edges = []
    seq_len = nodes.shape[0]

    # 1. 空间相邻关系（网格内相邻像素）
    current_pos = 0
    for grid_shape in grid_shapes:
        height, width = grid_shape
        grid_size = height * width

        # 构建网格内的空间关系
        for i in range(height):
            for j in range(width):
                current_idx = current_pos + i * width + j

                # 8邻居连接
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width:
                            neighbor_idx = current_pos + ni * width + nj
                            edges.append((current_idx, neighbor_idx))

        current_pos += grid_size

    # 2. 语义相似关系（相同颜色的token）
    # 这里简化处理，实际可以根据具体需求实现

    # 3. 逻辑依赖关系（输入-输出对）
    # 这里简化处理，实际可以根据具体需求实现

    # 4. 添加自连接确保每个节点都有边
    for i in range(seq_len):
        edges.append((i, i))

    return edges


class ARC2Dataset(Dataset):
    """ARC2 数据集加载器（复用原有逻辑）"""

    def __init__(self, challenges_path: str, solutions_path: str, max_examples: int = None):
        super().__init__()

        # 加载数据
        with open(challenges_path, 'r') as f:
            self.challenges = json.load(f)

        with open(solutions_path, 'r') as f:
            self.solutions = json.load(f)

        # 过滤有效样本（确保挑战和解决方案匹配）
        self.valid_keys = []
        for key in self.challenges:
            if key in self.solutions and len(self.challenges[key]['train']) > 0:
                self.valid_keys.append(key)

        if max_examples:
            self.valid_keys = self.valid_keys[:max_examples]

        print(f"Loaded {len(self.valid_keys)} valid ARC examples")

    def __len__(self):
        return len(self.valid_keys)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        key = self.valid_keys[idx]

        # 获取训练对
        train_pairs = self.challenges[key]['train']
        exam_inputs = [pair['input'] for pair in train_pairs]
        exam_outputs = [pair['output'] for pair in train_pairs]

        # 获取测试输入和输出
        test_input = self.challenges[key]['test'][0]['input']
        test_output = self.solutions[key][0]  # 解决方案是列表的列表

        return {
            'exam_inputs': exam_inputs,
            'exam_outputs': exam_outputs,
            'test_input': test_input,
            'test_output': test_output,
            'task_id': key
        }


def collate_arc2_batch(batch: List[Dict]) -> Dict[str, Any]:
    """整理ARC2批次数据"""
    return {
        'exam_inputs': [item['exam_inputs'] for item in batch],
        'exam_outputs': [item['exam_outputs'] for item in batch],
        'test_inputs': [item['test_input'] for item in batch],
        'test_outputs': [item['test_output'] for item in batch],
        'task_ids': [item['task_id'] for item in batch]
    }


class ARC2DirectTrainer:
    """ARC2 直接图推理训练器"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # 初始化模型
        self._init_model()

        # 初始化优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=11)

    def _init_model(self):
        """初始化ASPP模型和知识嵌入层"""
        update_rule = FFNUpdate(hidden_dim=self.config['hidden_dim'])

        self.model = ASPPOperator(
            update_rule=update_rule,
            hidden_dim=self.config['hidden_dim'],
            device=self.device,
        )

        # 知识嵌入层
        self.knowledge_embedding = KnowledgeEmbeddingLayer(
            vocab_size=12,  # 0-11: 颜色(0-9) + 分隔符(10) + 填充符(11)
            hidden_dim=self.config['hidden_dim']
        ).to(self.device)

        self.model.to(self.device)
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")

    def train_direct_reasoning(self, dataloader: DataLoader, epoch: int) -> float:
        """直接图推理训练"""
        self.model.train()
        self.knowledge_embedding.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        progress_bar = tqdm(dataloader, desc=f"Direct Reasoning Epoch {epoch}")

        for _, batch in enumerate(progress_bar):
            exam_inputs = batch['exam_inputs']
            exam_outputs = batch['exam_outputs']
            test_inputs = batch['test_inputs']
            test_outputs = batch['test_outputs']

            self.optimizer.zero_grad()

            batch_loss = 0.0
            batch_correct = 0
            batch_tokens = 0
            loss_tensors = []

            for i in range(len(exam_inputs)):
                try:
                    # 1. 编码目标序列
                    target_tokens = self.model.encode_arc2_sequence(
                        exam_inputs[i], exam_outputs[i], test_inputs[i], test_outputs[i]
                    )

                    # 截断或填充目标序列
                    if len(target_tokens) > 8192:
                        target_tokens = target_tokens[:8192]
                    else:
                        target_tokens = target_tokens + [self.model.PAD_TOKEN] * (8192 - len(target_tokens))

                    target_tensor = torch.tensor(target_tokens, dtype=torch.long, device=self.device)

                    # 2. 知识嵌入：token -> 图节点
                    positions = torch.arange(len(target_tokens), device=self.device)
                    roles = self._get_token_roles(target_tokens)  # 简化实现
                    relations = torch.zeros_like(positions)  # 简化实现

                    # 确保token在有效范围内 (0-11)
                    valid_target_tensor = torch.clamp(target_tensor, 0, 11)
                    nodes = self.knowledge_embedding(valid_target_tensor, positions, roles, relations)

                    # 3. 构建知识图谱
                    grid_shapes = self._get_grid_shapes(exam_inputs[i], exam_outputs[i], test_inputs[i])
                    edges = build_knowledge_graph(nodes, grid_shapes)
                    edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t()

                    # 4. 应用ASPP算子推理
                    self.model.set_graph(len(target_tokens), edge_index)
                    reasoned_nodes = self.model.forward(nodes, K=5)

                    # 5. 输出投影
                    logits = self.model.output_projection(reasoned_nodes)

                    # 6. 计算损失
                    valid_token_mask = (target_tensor != self.model.PAD_TOKEN)
                    target_for_loss = target_tensor.clone()
                    target_for_loss[~valid_token_mask] = -100

                    loss = F.cross_entropy(
                        logits.view(-1, self.model.vocab_size),
                        target_for_loss.view(-1),
                        ignore_index=-100,
                        reduction='mean'
                    )

                    # 计算token准确率
                    pred_tokens = torch.argmax(logits, dim=-1)
                    correct = ((pred_tokens == target_tensor) & valid_token_mask).sum().item()
                    valid_tokens = valid_token_mask.sum().item()

                    batch_loss += loss.item()
                    batch_correct += correct
                    batch_tokens += valid_tokens
                    loss_tensors.append(loss)

                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # 反向传播
            if loss_tensors:
                total_loss_val = sum(loss_tensors) / len(loss_tensors)
                total_loss_val.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.knowledge_embedding.parameters(), max_norm=0.5)

                self.optimizer.step()

            # 计算批次准确率
            batch_acc = batch_correct / batch_tokens if batch_tokens > 0 else 0
            total_correct += batch_correct
            total_tokens += batch_tokens

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{batch_loss/len(exam_inputs):.8f}",
                'token_acc': f"{batch_acc:.4f}"
            })

            total_loss += batch_loss

        # 计算总体准确率
        total_acc = total_correct / total_tokens if total_tokens > 0 else 0
        print(f"Epoch {epoch} Token Accuracy: {total_acc:.4f} ({total_correct}/{total_tokens})")

        return total_loss / len(dataloader)

    def _get_token_roles(self, tokens):
        """简化实现：获取token的语义角色"""
        roles = torch.zeros(len(tokens), dtype=torch.long)
        # 这里简化实现，实际可以根据token位置判断角色
        return roles

    def _get_grid_shapes(self, exam_inputs, exam_outputs, test_input):
        """获取所有网格的形状信息"""
        shapes = []

        # 训练输入网格
        for grid in exam_inputs:
            shapes.append((len(grid), len(grid[0]) if grid else 0))

        # 训练输出网格
        for grid in exam_outputs:
            shapes.append((len(grid), len(grid[0]) if grid else 0))

        # 测试输入网格
        shapes.append((len(test_input), len(test_input[0]) if test_input else 0))

        return shapes

    def validate(self, dataloader: DataLoader):
        """验证模型"""
        self.model.eval()
        self.knowledge_embedding.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                exam_inputs = batch['exam_inputs']
                exam_outputs = batch['exam_outputs']
                test_inputs = batch['test_inputs']
                test_outputs = batch['test_outputs']

                batch_loss = 0.0

                for i in range(len(exam_inputs)):
                    try:
                        # 复用训练相同的逻辑
                        target_tokens = self.model.encode_arc2_sequence(
                            exam_inputs[i], exam_outputs[i], test_inputs[i], test_outputs[i]
                        )

                        if len(target_tokens) > 8192:
                            target_tokens = target_tokens[:8192]
                        else:
                            target_tokens = target_tokens + [self.model.PAD_TOKEN] * (8192 - len(target_tokens))

                        target_tensor = torch.tensor(target_tokens, dtype=torch.long, device=self.device)

                        positions = torch.arange(len(target_tokens), device=self.device)
                        roles = self._get_token_roles(target_tokens)
                        relations = torch.zeros_like(positions)

                        # 确保token在有效范围内 (0-11)
                        valid_target_tensor = torch.clamp(target_tensor, 0, 11)
                        nodes = self.knowledge_embedding(valid_target_tensor, positions, roles, relations)

                        grid_shapes = self._get_grid_shapes(exam_inputs[i], exam_outputs[i], test_inputs[i])
                        edges = build_knowledge_graph(nodes, grid_shapes)
                        edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t()

                        self.model.set_graph(len(target_tokens), edge_index)
                        reasoned_nodes = self.model.forward(nodes, K=5)

                        logits = self.model.output_projection(reasoned_nodes)

                        valid_token_mask = (target_tensor != self.model.PAD_TOKEN)
                        target_for_loss = target_tensor.clone()
                        target_for_loss[~valid_token_mask] = -100

                        loss = F.cross_entropy(
                            logits.view(-1, self.model.vocab_size),
                            target_for_loss.view(-1),
                            ignore_index=-100,
                            reduction='mean'
                        )

                        batch_loss += loss.item()

                        # 计算准确率
                        pred_tokens = torch.argmax(logits, dim=-1)
                        correct_predictions += ((pred_tokens == target_tensor) & valid_token_mask).sum().item()
                        total_predictions += valid_token_mask.sum().item()

                    except Exception as e:
                        print(f"Validation error processing sample {i}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

                total_loss += batch_loss

        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        return avg_loss, accuracy

    def save_checkpoint(self, epoch: int, loss: float, path: str):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'embedding_state_dict': self.knowledge_embedding.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='Train ASPP model on ARC2 data with direct reasoning')

    # 数据参数
    parser.add_argument('--train_challenges', type=str,
                       default='arc-prize-2025/arc-agi_training_challenges.json',
                       help='Path to training challenges')
    parser.add_argument('--train_solutions', type=str,
                       default='arc-prize-2025/arc-agi_training_solutions.json',
                       help='Path to training solutions')
    parser.add_argument('--val_challenges', type=str,
                       default='arc-prize-2025/arc-agi_evaluation_challenges.json',
                       help='Path to validation challenges')
    parser.add_argument('--val_solutions', type=str,
                       default='arc-prize-2025/arc-agi_evaluation_solutions.json',
                       help='Path to validation solutions')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--max_examples', type=int, default=None, help='Max examples for debugging')

    # 保存参数
    parser.add_argument('--save_dir', type=str, default='checkpoints_direct', help='Checkpoint directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Resume from checkpoint')

    args = parser.parse_args()

    # 创建配置
    config = {
        'hidden_dim': args.hidden_dim,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # 创建数据加载器
    train_dataset = ARC2Dataset(args.train_challenges, args.train_solutions, args.max_examples)
    val_dataset = ARC2Dataset(args.val_challenges, args.val_solutions, args.max_examples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_arc2_batch,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_arc2_batch,
        num_workers=4
    )

    # 初始化训练器
    trainer = ARC2DirectTrainer(config)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 训练循环
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss = trainer.train_direct_reasoning(train_loader, epoch)

        # 验证
        val_loss, val_acc = trainer.validate(val_loader)

        # 获取当前学习率
        current_lr = trainer.optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, LR={current_lr:.2e}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(epoch, val_loss, os.path.join(args.save_dir, 'best_model.pth'))

        # 定期保存检查点
        if epoch % 5 == 0:
            trainer.save_checkpoint(epoch, val_loss, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'))

    print("\n=== 训练完成 ===")


if __name__ == "__main__":
    main()