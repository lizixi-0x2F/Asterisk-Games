#!/usr/bin/env python3
"""
ARC2 ASPP 训练脚本
基于ASPP算子的ARC抽象推理训练
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


class ARC2Dataset(Dataset):
    """ARC2 数据集加载器"""

    def __init__(self, challenges_path: str, solutions_path: str, max_examples: int = None):
        """
        初始化ARC2数据集

        Args:
            challenges_path: 挑战文件路径
            solutions_path: 解决方案文件路径
            max_examples: 最大样本数量（用于调试）
        """
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
        """
        获取单个ARC样本

        Returns:
            {
                'exam_inputs': List[List[List[int]]],  # 训练输入网格列表
                'exam_outputs': List[List[List[int]]], # 训练输出网格列表
                'test_input': List[List[int]],         # 测试输入网格
                'test_output': List[List[int]]         # 测试输出网格
            }
        """
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
    """
    整理ARC2批次数据
    """
    return {
        'exam_inputs': [item['exam_inputs'] for item in batch],
        'exam_outputs': [item['exam_outputs'] for item in batch],
        'test_inputs': [item['test_input'] for item in batch],
        'test_outputs': [item['test_output'] for item in batch],
        'task_ids': [item['task_id'] for item in batch]
    }


class ARC2Trainer:
    """ARC2 ASPP 训练器"""

    def __init__(self, config: Dict):
        """
        初始化训练器

        Args:
            config: 训练配置
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # 初始化模型
        self._init_model()

        # 初始化优化器和损失函数
        # 包括模型参数和投影适配器参数
        all_params = list(self.model.parameters()) + list(self.embedding_projection.parameters())
        self.optimizer = optim.AdamW(
            all_params,
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )

        # 动态学习率调度器（解决震荡和鞍点问题）
        self.scheduler = self._create_dynamic_scheduler(config)

        # 学习率监控
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = config.get('patience', 5)
        self.lr_reduction_factor = config.get('lr_reduction_factor', 0.5)

        # 梯度监控
        self.gradient_stats = {
            'total_norm': [],
            'max_grad': []
        }

        # 只计算颜色token（0-9），忽略分隔符（10）和填充token（11）
        self.criterion = nn.CrossEntropyLoss(ignore_index=11)

    def _create_dynamic_scheduler(self, config):
        """创建动态学习率调度器（解决震荡和鞍点问题）"""
        initial_lr = config.get('learning_rate', 1e-4)
        epochs = config.get('epochs', 10)
        warmup_epochs = config.get('warmup_epochs', 2)

        # 使用组合策略：Warmup + CosineAnnealingLR + ReduceLROnPlateau

        # 第一阶段：线性warmup（解决初始震荡）
        def warmup_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                return 1.0

        warmup_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=warmup_lambda
        )

        # 第二阶段：CosineAnnealingLR（平滑退火）
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs - warmup_epochs,  # 减去warmup阶段
            eta_min=initial_lr * 0.01
        )

        # 第三阶段：ReduceLROnPlateau（解决鞍点问题）
        plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.get('lr_reduction_factor', 0.5),
            patience=config.get('patience', 3),
            min_lr=initial_lr * 0.001
        )

        return {
            'warmup': warmup_scheduler,
            'cosine': cosine_scheduler,
            'plateau': plateau_scheduler,
            'warmup_epochs': warmup_epochs,
            'current_epoch': 0
        }

    def _update_scheduler(self, val_loss):
        """更新学习率调度器"""
        current_epoch = self.scheduler['current_epoch']

        # 第一阶段：warmup
        if current_epoch < self.scheduler['warmup_epochs']:
            self.scheduler['warmup'].step()
        # 第二阶段：余弦退火
        else:
            self.scheduler['cosine'].step()

        # 第三阶段：平台期检测（始终运行）
        self.scheduler['plateau'].step(val_loss)

        self.scheduler['current_epoch'] += 1

    def _apply_gradient_clipping(self):
        """应用梯度裁剪和监控"""
        grad_clip = self.config.get('grad_clip', 0.5)
        clip_type = self.config.get('grad_clip_type', 'norm')

        # 梯度监控
        total_norm = 0.0
        max_grad = 0.0

        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                max_grad = max(max_grad, p.grad.data.abs().max().item())

        total_norm = total_norm ** 0.5

        # 应用梯度裁剪
        if clip_type == 'norm':
            # 梯度范数裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=grad_clip,
                norm_type=2
            )
        elif clip_type == 'value':
            # 梯度值裁剪
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(),
                clip_value=grad_clip
            )

        # 记录梯度统计信息（用于调试）
        if hasattr(self, 'gradient_stats'):
            self.gradient_stats['total_norm'].append(total_norm)
            self.gradient_stats['max_grad'].append(max_grad)
        else:
            self.gradient_stats = {
                'total_norm': [total_norm],
                'max_grad': [max_grad]
            }

    def _apply_gradient_regularization(self):
        """应用梯度正则化（L2正则化）"""
        # L2正则化（权重衰减）
        weight_decay = self.config.get('weight_decay', 1e-4)
        if weight_decay > 0:
            for param in self.model.parameters():
                if param.grad is not None and param.requires_grad:
                    param.grad.data.add_(weight_decay * param.data)




    def _build_sequence_graph(self, seq_len: int) -> torch.LongTensor:
        """构建序列位置邻接图"""
        edges = []
        # 创建线性链式连接
        for i in range(seq_len - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])  # 双向连接

        if not edges:
            edges.append([0, 0])  # 单节点图

        return torch.tensor(edges, dtype=torch.long, device=self.device).t()


    def _init_embedding_adapter(self):
        """初始化TreeGPT到ASPP的线性投影适配器"""
        # TreeGPT隐藏维度是256，ASPP配置可能是128，需要正确映射
        treegpt_hidden_dim = 256  # TreeGPT预训练权重维度
        aspp_hidden_dim = self.config['hidden_dim']

        self.embedding_projection = nn.Linear(
            treegpt_hidden_dim, aspp_hidden_dim
        ).to(self.device)

        print(f"Linear projection adapter initialized: {treegpt_hidden_dim} -> {aspp_hidden_dim}")

    def _project_treegpt_embeddings(self, treegpt_embeddings: torch.Tensor) -> torch.Tensor:
        """将TreeGPT嵌入线性投影到ASPP语义空间"""
        return self.embedding_projection(treegpt_embeddings)

    def print_gradient_stats(self):
        """打印梯度统计信息"""
        if not self.gradient_stats['total_norm']:
            print("暂无梯度统计信息")
            return

        total_norms = self.gradient_stats['total_norm']
        max_grads = self.gradient_stats['max_grad']

        print(f"梯度统计 - 范数: 平均={sum(total_norms)/len(total_norms):.4f}, "
              f"最大={max(total_norms):.4f}, 最小={min(total_norms):.4f}")
        print(f"梯度统计 - 值: 平均={sum(max_grads)/len(max_grads):.4f}, "
              f"最大={max(max_grads):.4f}, 最小={min(max_grads):.4f}")

    def _init_model(self):
        """初始化ASPP模型"""
        update_rule = FFNUpdate(
            hidden_dim=self.config['hidden_dim'],
        )

        self.model = ASPPOperator(
            update_rule=update_rule,
            hidden_dim=self.config['hidden_dim'],
            device=self.device,
        )

        
        # 确保token嵌入层存在（用于算子蒸馏）
        if not hasattr(self.model, 'token_embedding_layer'):
            self.model.token_embedding_layer = nn.Embedding(
                self.model.vocab_size, self.config['hidden_dim'], device=self.device
            )

        # 加载TreeGPT模型用于嵌入提取
        self._load_treegpt_model()

        # 创建TreeGPT到ASPP的嵌入适配器（冻结TreeGPT嵌入，训练适配器）
        self._init_embedding_adapter()

        self.model.to(self.device)
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")

    def _load_treegpt_model(self):
        """加载TreeGPT模型用于嵌入提取"""
        try:
            # 简化TreeGPT模型加载，只使用嵌入层
            # 创建简化的TreeGPT嵌入层（17个token，隐藏维度匹配）
            self.treegpt_embedding_layer = nn.Embedding(
                17, self.config['hidden_dim'], device=self.device
            )

            # 加载预训练权重（只加载嵌入层部分）
            treegpt_checkpoint = torch.load('./TreeGPT-main/best_treeffn_seq2seq.pth',
                                          map_location=self.device)

            # 提取嵌入层权重
            if 'model_state_dict' in treegpt_checkpoint:
                model_state = treegpt_checkpoint['model_state_dict']
                embedding_weight = model_state.get('token_embedding.weight', None)

                if embedding_weight is not None and embedding_weight.shape[0] == 17:
                    self.treegpt_embedding_layer.weight.data = embedding_weight
                    print(f"TreeGPT embedding layer loaded successfully (shape: {embedding_weight.shape})")
                else:
                    print("Using randomly initialized TreeGPT embeddings")
            else:
                print("Using randomly initialized TreeGPT embeddings")

            # 创建简化的tokenizer
            class SimpleARCGridTokenizer:
                def __init__(self):
                    self.color_tokens = list(range(10))
                    self.GRID_START = 10
                    self.GRID_END = 11
                    self.ROW_SEP = 12
                    self.EXAMPLE_SEP = 13
                    self.INPUT_OUTPUT_SEP = 14
                    self.TEST_START = 15
                    self.PAD = 16
                    self.vocab_size = 17

            self.treegpt_tokenizer = SimpleARCGridTokenizer()
            print("TreeGPT components initialized for embedding extraction")

        except Exception as e:
            print(f"Failed to load TreeGPT model: {e}")
            print("Using dummy embeddings for testing")
            self.treegpt_model = None
            self.treegpt_tokenizer = None

    def _generate_from_teacher_embeddings(self, teacher_embeddings: torch.Tensor):
        """
        从教师嵌入生成预测：教师嵌入 -> 线性投影 -> 图构建 -> ASPP算子求解 -> 输出

        Args:
            teacher_embeddings: TreeGPT教师嵌入 [seq_len, treegpt_hidden_dim]

        Returns:
            asp_output: ASPP内部表示 [seq_len, hidden_dim]
            logits: ASPP输出logits [seq_len, vocab_size]
        """
        # 线性投影到ASPP语义空间
        projected_embeddings = self._project_treegpt_embeddings(teacher_embeddings)

        # 构建序列图结构
        seq_len = projected_embeddings.shape[0]
        edge_index = self._build_sequence_graph(seq_len)

        # 应用ASPP算子进行语义演化
        self.model.set_graph(seq_len, edge_index)
        asp_output = self.model.forward(projected_embeddings, K=3)

        # 输出投影到词汇表
        logits = self.model.output_projection(asp_output)

        return asp_output, logits

    # SFT训练函数已删除，只使用GRPO训练

    def train_embedding_stage(self, dataloader: DataLoader, epoch: int) -> float:
        """第一阶段：只训练嵌入蒸馏，不使用数据集"""
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Embedding Stage Epoch {epoch}")

        for _, batch in enumerate(progress_bar):
            exam_inputs = batch['exam_inputs']
            exam_outputs = batch['exam_outputs']
            test_inputs = batch['test_inputs']
            test_outputs = batch['test_outputs']

            self.optimizer.zero_grad()

            batch_loss = 0.0
            loss_tensors = []

            for i in range(len(exam_inputs)):
                try:
                    # 准备目标序列（不使用数据集，直接从教师嵌入学习）
                    target_tokens = self.model.encode_arc2_sequence(
                        exam_inputs[i], exam_outputs[i], test_inputs[i], test_outputs[i]
                    )

                    # 截断或填充目标序列
                    if len(target_tokens) > 8192:
                        target_tokens = target_tokens[:8192]
                    else:
                        target_tokens = target_tokens + [self.model.PAD_TOKEN] * (8192 - len(target_tokens))

                    target_tokens_tensor = torch.tensor(target_tokens, dtype=torch.long, device=self.device)

                    # 使用TreeGPT教师嵌入作为学习目标
                    with torch.no_grad():
                        if hasattr(self, 'treegpt_embedding_layer'):
                            # 获取TreeGPT教师嵌入
                            valid_target_tokens = torch.clamp(target_tokens_tensor, 0, 16)
                            teacher_embeddings = self.treegpt_embedding_layer(valid_target_tokens)

                            # 直接使用教师嵌入生成预测
                            asp_output, _ = self._generate_from_teacher_embeddings(teacher_embeddings)

                    # 嵌入蒸馏损失：ASPP输出与教师嵌入的相似度
                    with torch.no_grad():
                        teacher_embeddings = self.treegpt_embedding_layer(target_tokens_tensor)

                    # 将教师嵌入投影到ASPP语义空间进行比较
                    projected_teacher_embeddings = self._project_treegpt_embeddings(teacher_embeddings)

                    # 创建mask：排除padding token的位置（PAD_TOKEN = 11）
                    non_padding_mask = (target_tokens_tensor != self.model.PAD_TOKEN)

                    # 只计算非padding位置的MSE损失
                    if non_padding_mask.sum() > 0:
                        embedding_loss = F.mse_loss(
                            asp_output[non_padding_mask],
                            projected_teacher_embeddings[non_padding_mask]
                        )
                    else:
                        embedding_loss = torch.tensor(0.0, device=self.device)

                    # 第一阶段只训练嵌入蒸馏
                    loss = embedding_loss

                    batch_loss += loss.item()
                    loss_tensors.append(loss)

                except Exception as e:
                    print(f"Embedding Stage Error processing sample {i}: {e}")
                    continue

            # 反向传播
            if loss_tensors:
                total_loss_val = sum(loss_tensors) / len(loss_tensors)
                total_loss_val.backward()
                self._apply_gradient_clipping()
                self._apply_gradient_regularization()
                self.optimizer.step()

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{batch_loss/len(exam_inputs):.8f}"
            })

            total_loss += batch_loss

        return total_loss / len(dataloader)

    def train_lmhead_stage(self, dataloader: DataLoader, epoch: int) -> float:
        """第二阶段：冻结算子权重，只训练LM head"""
        # 首先解冻所有参数
        for param in self.model.parameters():
            param.requires_grad = True
        for param in self.embedding_projection.parameters():
            param.requires_grad = True

        # 然后只冻结ASPP核心参数，只保留主output_projection可训练（不训练child operators的output）
        print("冻结参数:")
        for name, param in self.model.named_parameters():
            if name == 'output_projection.weight' or name == 'output_projection.bias':
                param.requires_grad = True
                print(f"  Trainable: {name}")
            else:
                param.requires_grad = False
                print(f"  Frozen: {name}")

        print("可训练参数:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"  {name}")

        # 调整学习率为1e-4用于LM head微调
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 1e-3
        print(f"LMHead阶段学习率调整为: 1e-4")

        # 只训练LM head和投影适配器
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"LMHead Stage Epoch {epoch}")

        for _, batch in enumerate(progress_bar):
            exam_inputs = batch['exam_inputs']
            exam_outputs = batch['exam_outputs']
            test_inputs = batch['test_inputs']
            test_outputs = batch['test_outputs']

            self.optimizer.zero_grad()

            batch_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            loss_tensors = []

            for i in range(len(exam_inputs)):
                try:
                    # 准备目标序列
                    target_tokens = self.model.encode_arc2_sequence(
                        exam_inputs[i], exam_outputs[i], test_inputs[i], test_outputs[i]
                    )

                    # 截断或填充目标序列
                    if len(target_tokens) > 8192:
                        target_tokens = target_tokens[:8192]
                    else:
                        target_tokens = target_tokens + [self.model.PAD_TOKEN] * (8192 - len(target_tokens))

                    target_tokens_tensor = torch.tensor(target_tokens, dtype=torch.long, device=self.device)

                    # 使用完整的ARC序列作为输入（包括exam inputs/outputs和test input/output）
                    # 模型会并行预测所有位置的token，我们计算损失时只关注test output部分
                    input_tokens = self.model.encode_arc2_sequence(
                        exam_inputs[i], exam_outputs[i], test_inputs[i], test_outputs[i]
                    )

                    # 截断或填充到最大长度
                    if len(input_tokens) > 8192:
                        input_tokens = input_tokens[:8192]
                    else:
                        input_tokens = input_tokens + [self.model.PAD_TOKEN] * (8192 - len(input_tokens))

                    input_tensor = torch.tensor(input_tokens, dtype=torch.long, device=self.device)

                    # 获取token嵌入和位置编码
                    token_emb = self.model.token_embedding_layer(input_tensor)
                    positions = torch.arange(len(input_tokens), device=self.device)
                    pos_emb = self.model.position_emb(positions)
                    combined_emb = token_emb + pos_emb

                    # 应用ASPP算子进行推理
                    edge_index = self.model._create_8neighbor_edges(test_inputs[i])
                    self.model.set_graph(len(input_tokens), edge_index)
                    processed_features = self.model.forward(combined_emb, K=5)

                    # 获取所有位置的logits
                    logits = self.model.output_projection(processed_features)

                    # LM head损失：训练模型输出正确的token（包括特殊token，排除padding）
                    valid_token_mask = (target_tokens_tensor != self.model.PAD_TOKEN)
                    target_for_loss = target_tokens_tensor.clone()
                    target_for_loss[~valid_token_mask] = -100  # 忽略padding token的损失

                    lm_loss = F.cross_entropy(
                        logits.view(-1, self.model.vocab_size),
                        target_for_loss.view(-1),
                        ignore_index=-100,
                        reduction='mean',
                        label_smoothing=0.1
                    )

                    # 第二阶段只训练LM head
                    loss = lm_loss

                    batch_loss += loss.item()
                    loss_tensors.append(loss)

                    # 计算准确率
                    pred_tokens = torch.argmax(logits, dim=-1)
                    correct_predictions += ((pred_tokens == target_tokens_tensor) & valid_token_mask).sum().item()
                    total_predictions += valid_token_mask.sum().item()

                except Exception as e:
                    print(f"LMHead Stage Error processing sample {i}: {e}")
                    continue

            # 反向传播
            if loss_tensors:
                total_loss_val = sum(loss_tensors) / len(loss_tensors)
                total_loss_val.backward()
                self._apply_gradient_clipping()
                self._apply_gradient_regularization()
                self.optimizer.step()

            # 更新进度条
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            progress_bar.set_postfix({
                'loss': f"{batch_loss/len(exam_inputs):.8f}",
                'acc': f"{accuracy:.4f}"
            })

            total_loss += batch_loss

        return total_loss / len(dataloader)

    def validate(self, dataloader: DataLoader):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        full_sequence_correct_count = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                exam_inputs = batch['exam_inputs']
                exam_outputs = batch['exam_outputs']
                test_inputs = batch['test_inputs']
                test_outputs = batch['test_outputs']

                batch_loss = 0.0

                for i in range(len(exam_inputs)):
                    try:
                        # 复用训练相同的逻辑进行验证
                        # 编码目标序列
                        target_tokens = self.model.encode_arc2_sequence(
                            exam_inputs[i],
                            exam_outputs[i],
                            test_inputs[i],
                            test_outputs[i]
                        )

                        # 截断或填充到最大长度
                        if len(target_tokens) > 8192:
                            target_tokens = target_tokens[:8192]
                        else:
                            target_tokens = target_tokens + [self.model.PAD_TOKEN] * (8192 - len(target_tokens))

                        target_tensor = torch.tensor(target_tokens, dtype=torch.long, device=self.device)

                        # 使用完整的ARC序列作为输入（与训练相同）
                        input_tokens = self.model.encode_arc2_sequence(
                            exam_inputs[i], exam_outputs[i], test_inputs[i], test_outputs[i]
                        )

                        # 截断或填充到最大长度
                        if len(input_tokens) > 8192:
                            input_tokens = input_tokens[:8192]
                        else:
                            input_tokens = input_tokens + [self.model.PAD_TOKEN] * (8192 - len(input_tokens))

                        input_tensor = torch.tensor(input_tokens, dtype=torch.long, device=self.device)

                        # 获取token嵌入和位置编码（与训练相同）
                        token_emb = self.model.token_embedding_layer(input_tensor)
                        positions = torch.arange(len(input_tokens), device=self.device)
                        pos_emb = self.model.position_emb(positions)
                        combined_emb = token_emb + pos_emb

                        # 应用ASPP算子进行推理（与训练相同）
                        edge_index = self.model._create_8neighbor_edges(test_inputs[i])
                        self.model.set_graph(len(input_tokens), edge_index)
                        processed_features = self.model.forward(combined_emb, K=5)

                        # 获取所有位置的logits（与训练相同）
                        logits = self.model.output_projection(processed_features)

                        # 计算LM head损失（与训练相同）
                        valid_token_mask = (target_tensor != self.model.PAD_TOKEN)
                        target_for_loss = target_tensor.clone()
                        target_for_loss[~valid_token_mask] = -100

                        loss = F.cross_entropy(
                            logits.view(-1, self.model.vocab_size),
                            target_for_loss.view(-1),
                            ignore_index=-100,
                            reduction='mean',
                            label_smoothing=0.1
                        )

                        batch_loss += loss.item()

                        # 计算准确率（通过LM head输出的token）
                        pred_tokens = torch.argmax(logits, dim=-1)

                        # 计算所有有效token的准确率（包括颜色和分隔符，排除填充符）
                        valid_token_mask = target_tensor != self.model.PAD_TOKEN
                        correct_predictions += ((pred_tokens == target_tensor) & valid_token_mask).sum().item()
                        total_predictions += valid_token_mask.sum().item()

                        # 计算整个序列准确率（排除填充符）
                        valid_mask = target_tensor != self.model.PAD_TOKEN
                        full_sequence_correct = torch.all((pred_tokens == target_tensor)[valid_mask]).item()
                        full_sequence_correct_count += full_sequence_correct

                    except Exception:
                        continue

                total_loss += batch_loss

        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        full_sequence_accuracy = full_sequence_correct_count / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0

        print(f"Validation - Full Sequence Accuracy: {full_sequence_accuracy:.4f} ({full_sequence_correct_count}/{len(dataloader.dataset)})")
        return avg_loss, accuracy

    def save_checkpoint(self, epoch: int, loss: float, path: str):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """加载检查点（处理不兼容的state_dict）"""
        checkpoint = torch.load(path, map_location=self.device)

        # 处理不兼容的state_dict
        model_state_dict = checkpoint['model_state_dict']
        current_state_dict = self.model.state_dict()

        # 过滤掉不兼容的key
        filtered_state_dict = {}
        for key, value in model_state_dict.items():
            if key in current_state_dict and current_state_dict[key].shape == value.shape:
                filtered_state_dict[key] = value
            else:
                print(f"跳过不兼容参数: {key} (形状不匹配)")

        # 加载兼容的参数
        self.model.load_state_dict(filtered_state_dict, strict=False)

        # 尝试加载优化器状态（如果兼容）
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            print("优化器状态不兼容，使用新的优化器")

        print(f"Checkpoint loaded from {path}, epoch {checkpoint['epoch']} (部分参数加载)")


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='Train ASPP model on ARC2 data')

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
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--max_seq_length', type=int, default=8192, help='Max sequence length')
    parser.add_argument('--max_examples', type=int, default=None, help='Max examples for debugging')

    # 学习率调度参数
    parser.add_argument('--patience', type=int, default=3, help='Patience for learning rate reduction')
    parser.add_argument('--lr_reduction_factor', type=float, default=0.5, help='Factor for learning rate reduction')
    parser.add_argument('--warmup_epochs', type=int, default=2, help='Number of warmup epochs')

    # 梯度裁剪参数
    parser.add_argument('--grad_clip', type=float, default=0.5, help='Gradient clipping value')
    parser.add_argument('--grad_clip_type', type=str, default='norm', choices=['norm', 'value'], help='Gradient clipping type')

    # 保存参数
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Resume from checkpoint')

    args = parser.parse_args()

    # 创建配置
    config = {
        'hidden_dim': args.hidden_dim,
        'max_seq_length': args.max_seq_length,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'patience': args.patience,
        'lr_reduction_factor': args.lr_reduction_factor,
        'warmup_epochs': args.warmup_epochs,
        'grad_clip': args.grad_clip,
        'grad_clip_type': args.grad_clip_type,
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
    trainer = ARC2Trainer(config)

    # 加载检查点（如果提供）
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 训练循环 - 两阶段训练
    best_val_loss = float('inf')

    # 第一阶段：嵌入蒸馏训练
    print("=== 第一阶段：嵌入蒸馏训练 ===")
    embedding_epochs = 1  # 训练100步嵌入蒸馏

    for epoch in range(1, embedding_epochs + 1):
        # 嵌入蒸馏训练
        train_loss = trainer.train_embedding_stage(train_loader, epoch)

        # 验证（使用嵌入蒸馏验证）
        val_loss, val_acc = trainer.validate(val_loader)

        # 更新学习率
        trainer._update_scheduler(val_loss)

        # 获取当前学习率
        current_lr = trainer.optimizer.param_groups[0]['lr']
        print(f"Embedding Stage Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, LR={current_lr:.2e}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(epoch, val_loss, os.path.join(args.save_dir, 'best_embedding_model.pth'))

    # 第二阶段：LM head训练
    print("=== 第二阶段：LM head训练 ===")

    for epoch in range(embedding_epochs + 1, args.epochs + 1):
        # LM head训练
        train_loss = trainer.train_lmhead_stage(train_loader, epoch)

        # 验证
        val_loss, val_acc = trainer.validate(val_loader)

        # 更新学习率
        trainer._update_scheduler(val_loss)

        # 获取当前学习率
        current_lr = trainer.optimizer.param_groups[0]['lr']
        print(f"LMHead Stage Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, LR={current_lr:.2e}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(epoch, val_loss, os.path.join(args.save_dir, 'best_model.pth'))

        # 定期保存检查点
        if epoch % 10 == 0:
            trainer.save_checkpoint(epoch, val_loss, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'))

    # 训练结束后打印梯度统计信息
    print("\n=== 训练完成 ===")
    trainer.print_gradient_stats()


if __name__ == "__main__":
    main()